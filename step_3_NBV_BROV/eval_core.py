"""평가 코어 — `evaluate_nbv.py`와 `tools/corruption_sweep.py`가 공유한다.

이 모듈은 **AppLauncher를 띄우지 않는다**. IsaacLab 스크립트는 `AppLauncher`가
먼저 돌아야 `isaaclab.*` / `omni.*` import가 성립하므로, 진입점 스크립트가
AppLauncher를 띄운 **다음에** 이 모듈을 import해야 한다. 그래서 여기에는
argparse도 AppLauncher도 두지 않는다 — 두 진입점이 각자 자기 인자를 갖는다.

여기 모인 것은 "정책 하나를 N 에피소드 돌려 기록을 남기는" 절차와, 그 기록을
읽는 두 헬퍼다. ②a 오염 스윕은 오염 세기만 바꿔가며 같은 절차를 반복하므로
복제하지 않고 그대로 쓴다 — 지표 정의가 두 벌이 되면 비교가 무의미해진다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from algorithm.algo_nbv_continuous import Actor
from envs.nbv_baselines import BaselinePolicy


def _quat_angle(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """두 쿼터니언 사이 회전각 [rad]. 부호 모호성을 없애려 |dot|을 쓴다."""
    dot = (q_a * q_b).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


class Policy:
    """액션 생성기. 환경 난수 스트림을 오염시키지 않는 것이 핵심 계약."""

    def __init__(self, name: str, env, device, seed: int, stochastic: bool,
                 ablate_map: str = "none"):
        self.name = name
        self.kind = "ckpt"
        self._device = device
        # 환경과 분리된 Generator — 공정 비교의 전제(모듈 docstring 참조)
        self._gen = torch.Generator(device=device).manual_seed(seed)
        self._stochastic = stochastic
        # ── 지도 절제 (2026-09-09) ──────────────────────────────────────
        # "정책이 정말 지도를 읽고 다음 시점을 고르는가, 아니면 자기 좌표만 보고
        # 좋은 훑기 궤적을 재생하는가"를 가르는 시험이다. actor의 스칼라 입력은
        # 자기 구면좌표 3개뿐이라(`env.py::_get_observations`), "무엇을 이미
        # 봤는지"는 **오직 voxel 격자에서만** 온다. 그래서 voxel만 무력화하면
        # 지도 의존도가 분리된다.
        #   shuffle: env끼리 voxel을 뒤섞는다 — 입력 분포는 그대로(진짜 지도)이고
        #            자기 상태와의 **대응만** 깨진다. zero보다 공정한 시험이다.
        #   zero:    지도가 아예 없는 경우.
        self._ablate_map = ablate_map

        self._baseline = None
        if name in ("random", "hold", "orbit", "approach", "sweep"):
            self._baseline = BaselinePolicy(name, seed=seed, device=device)
            self.kind = name
            self.actor = None
            self._sweep_t = 0
            return

        ckpt = torch.load(name, map_location=device)
        cfg = env.cfg
        self.actor = Actor(
            img_ch=2, scalar_dim=3, action_dim=cfg.action_space,
            H=cfg.visual.h, W=cfg.visual.w,
        ).to(device)
        # ── 학습/평가 env 수 일치 검사 (2026-09-03) ────────────────────
        # tiled rendering에서 **env 수가 화면 밝기를 바꾼다**(같은 32개 물체로
        # 32 env 평균 160.0 vs 256 env 평균 122.2). 광원은 매 리셋 고정값
        # (light_level=7)이라 절대 밝기가 정책이 붙잡을 수 있는 단서이고,
        # 따라서 학습과 다른 env 수로 평가하면 정책이 학습 때 본 적 없는
        # 밝기를 보게 된다. coverage는 depth 기반이라 무관하지만, actor의
        # 시각 입력은 직접 영향을 받는다.
        #
        # 조용히 어기기 쉬운 규칙이라 여기서 경고한다. 근본 해결은 조명
        # 레벨 DR이며, 그것은 "파이프라인 구성 후 다변수화" 단계의 1순위다.
        _targs = ckpt.get("args") or {}

        # 렌더 경로도 같아야 한다 — Camera와 TiledCamera는 화면 밝기가 다르다.
        trained_path = _targs.get("camera_path")
        cur_path = "tiled" if getattr(env.cfg, "use_tiled_camera", True) else "per_env"
        if trained_path is not None and trained_path != cur_path:
            print(f"[eval] ⚠ 학습 렌더 경로({trained_path}) ≠ 평가 경로({cur_path})"
                  f" — 두 경로는 화면 밝기가 다르므로 정책이 학습 때 본 적 없는"
                  f" 입력을 받는다. --camera_path {trained_path} 로 맞출 것.")

        # 관측 프레임이 어긋나면 정책이 학습 때 본 적 없는 입력을 받는다 —
        # 렌더 경로 불일치보다 조용하고 더 치명적이다(값이 그럴듯하게 나온다).
        trained_ego = _targs.get("egocentric_vox")
        cur_ego = bool(getattr(env.cfg, "vox_egocentric", False))
        if trained_ego is not None and bool(trained_ego) != cur_ego:
            print(f"[eval] ⚠ 학습 voxel 프레임(자기중심={bool(trained_ego)}) ≠ 평가"
                  f"(자기중심={cur_ego}) — 관측 프레임이 다르다."
                  f" {'--egocentric_vox 를 줄 것' if trained_ego else '--egocentric_vox 를 뺄 것'}.")

        trained_envs = _targs.get("num_envs")
        if trained_envs is not None and trained_envs != env.num_envs:
            print(f"[eval] ⚠ 학습 env 수({trained_envs}) ≠ 평가 env 수"
                  f"({env.num_envs}) — tiled 렌더는 env 수에 따라 화면 밝기가"
                  f" 달라진다. 같은 값으로 맞출 것.")

        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.trained_iters = ckpt.get("it", -1)

    @torch.no_grad()
    def act(self, obs, n_env: int, a_dim: int) -> torch.Tensor:
        # 베이스라인 구현은 `envs/nbv_baselines.py` 에 있다 — Gazebo 배포 루프가
        # **같은 코드**를 쓴다(2026-09-10 분리). 두 곳이 갈라지면 "같은 결정 수
        # random 대비" 라는 판정 기준이 무의미해진다.
        if self._baseline is not None:
            return self._baseline.act(n_env, a_dim)
        vox = obs["vox_actor"]
        if self._ablate_map == "shuffle":
            perm = torch.randperm(vox.shape[0], generator=self._gen, device=self._device)
            vox = vox[perm]
        elif self._ablate_map == "zero":
            vox = torch.zeros_like(vox)

        if self._stochastic:
            a, _u, _lp, _e = self.actor.sample(
                vox, obs["img_semantic"], obs["extra_info"])
            return a
        return self.actor.greedy(vox, obs["img_semantic"], obs["extra_info"])


def run_policy(env, policy: Policy, n_episodes: int, seed: int, out_dir: Path) -> dict:
    """한 정책을 n_episodes 만큼 돌리고 에피소드/스텝 기록을 남긴다."""
    device = env.device
    E, A = env.num_envs, env.cfg.action_space

    # 정책마다 동일 에피소드를 보장하기 위해 환경 난수를 되심는다
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset()

    # ── 관측 가능 표면 마스크 (2026-09-10, 계획 §11.7-4) ─────────────────────
    # sweep 정책은 feasible box의 (θ,φ,ψ) 격자를 결정론적으로 지나가므로, 에피소드가
    # 끝날 때 남은 `weight>0 ∧ surf`의 **합집합**이 "이 box에서 볼 수 있는 표면"이다.
    # 리셋이 볼륨을 지우기 전에 잡아야 하므로 `_reset_idx`를 감싼다. 초기 reset()
    # 뒤에 거는 이유: 그 전에는 앞 정책이 남긴 지도가 섞인다. 고정 자세 단일
    # 물체(randomize_object_pose=False)에서만 의미가 있다 — 전 env가 같은 voxel 프레임.
    seen_union = None
    orig_reset_idx = env._reset_idx
    if policy.kind == "sweep":
        seen_union = torch.zeros_like(env._surf_vol[0])

        def _accumulate(ids):
            ids = torch.as_tensor(ids, device=device)
            seen_union.logical_or_(((env._weight_vol[ids] > 0) & env._surf_vol[ids]).any(dim=0))

        def _reset_idx_hooked(ids):
            _accumulate(ids)
            return orig_reset_idx(ids)
        env._reset_idx = _reset_idx_hooked

    ep_rows: list[dict] = []
    step_rows: list[dict] = []
    # 에피소드별 coverage 궤적 — 정책마다 에피소드 길이가 달라(성공하면 일찍
    # 끝난다) **종료 시점 coverage끼리의 비교는 구조적으로 편향된다**: 일찍
    # 성공한 정책은 그 시점에서 누적이 멈추고, 실패하는 정책은 25결정까지
    # 계속 쌓는다. 같은 결정 수에서 비교하려면 궤적이 필요하다.
    ep_curves: list[list[tuple[float, float]]] = []   # [(cov_q, cov_bin), ...]
    cur_curve: list[list[tuple[float, float]]] = [[] for _ in range(E)]
    ep_len = torch.zeros(E, dtype=torch.long, device=device)
    ep_ret = torch.zeros(E, device=device)
    psi_sum = [0.0] * E                      # 에피소드 평균 관측 반경용
    decision = 0

    while len(ep_rows) < n_episodes:
        act = policy.act(obs, E, A)

        # 목표는 액션 적용 직후 갱신되므로, 추종 오차를 재려면 step() **이후**의
        # 목표(= 이번 결정의 목표)와 step() 이후의 실제 pose를 비교해야 한다.
        obs, reward, terminated, truncated, _ = env.step(act)
        decision += 1
        ep_len += 1
        ep_ret += reward

        p_err = torch.norm(
            env._robot.data.root_pos_w - env._guidance.p_target, dim=-1)
        q_err = _quat_angle(env._robot.data.root_quat_w, env._guidance.q_target)
        # ②a: 채점은 진실 스트림(오염이 꺼져 있으면 보상용과 동일 텐서).
        cov_now = env._coverage_for_scoring()
        cov_bin_now = env._coverage_bin_for_scoring()
        term_q, term_bin = env._terminal_coverage_for_scoring()

        step_rows.append(dict(
            decision=decision,
            pos_err_m=p_err.mean().item(),
            pos_err_max_m=p_err.max().item(),
            att_err_deg=math.degrees(q_err.mean().item()),
            coverage=cov_now.mean().item(),
            coverage_binary=cov_bin_now.mean().item(),
            psi=env._sph_psi.mean().item(),
            phi_deg=math.degrees(env._sph_phi.mean().item()),
            # 클램프 한계에 붙어 있는 env 비율 — 포화 여부의 직접 지표
            psi_at_max=(env._sph_psi > env.cfg.psi_max - 1e-3).float().mean().item(),
            phi_at_max=(env._sph_phi > env.cfg.phi_max - 1e-3).float().mean().item(),
            action_abs_mean=act.abs().mean().item(),
            # tanh 포화: |a|가 1에 붙어 있으면 mu가 발산했다는 신호
            action_saturated=(act.abs() > 0.99).float().mean().item(),
        ))

        for i in range(E):
            psi_sum[i] += env._sph_psi[i].item()

        done = terminated | truncated
        # 종료된 env의 `cov_now`는 이미 리셋된 0이므로 `terminal_*`를 쓴다.
        for i in range(E):
            if done[i]:
                cur_curve[i].append((term_q[i].item(), term_bin[i].item()))
            else:
                cur_curve[i].append((cov_now[i].item(), cov_bin_now[i].item()))

        for eid in done.nonzero(as_tuple=True)[0].tolist():
            if len(ep_rows) >= n_episodes:
                break
            covq = term_q[eid].item()
            covb = term_bin[eid].item()
            # 로봇의 자가채점(믿음 스트림) — 진실과의 차이가 "얼마나 잘못 알고 있나"
            covq_belief = env.terminal_coverage_q[eid].item()
            ep_rows.append(dict(
                episode=len(ep_rows),
                # env 번호 = 물체 번호(물체는 env마다 고정 배정된다). 같은 물체의
                # 여러 궤적을 묶어 "이 물체에서 잘 고르면 얼마나 더 볼 수 있나"를
                # 재려면 반드시 필요하다 — 적응성 여유 측정의 축이다.
                env_id=eid,
                outcome="success" if terminated[eid].item() else "timeout",
                length=int(ep_len[eid].item()),
                ep_return=ep_ret[eid].item(),
                coverage=covq,
                coverage_binary=covb,
                coverage_belief=covq_belief,
                # 에피소드 동안의 평균 관측 반경. 예전에는 품질비를
                # Beer-Lambert로 역산했는데, 그 식은 전역 Q_sat 정규화를
                # 전제하므로 (A) voxel별 정규화 도입 후 무효다. 실제 psi를
                # 직접 누적해 쓴다 — 근접 전략 여부의 **독립** 증거이기도 하다.
                mean_obs_dist_m=(psi_sum[eid] / max(int(ep_len[eid].item()), 1)),
                gt_never=env._diag_gt_never[eid].item(),
                gt_partial=env._diag_gt_partial[eid].item(),
                gt_full=env._diag_gt_full[eid].item(),
            ))
            ep_curves.append(cur_curve[eid])
            ep_len[eid] = 0
            ep_ret[eid] = 0.0
            psi_sum[eid] = 0.0

        for i in done.nonzero(as_tuple=True)[0].tolist():
            cur_curve[i] = []

    tag = Path(policy.name).stem if policy.kind == "ckpt" else policy.name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / f"{tag}_episodes.csv", ep_rows)
    _write_csv(out_dir / f"{tag}_steps.csv", step_rows)

    def m(key, rows=ep_rows):
        vals = [r[key] for r in rows if isinstance(r[key], float) and math.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    if seen_union is not None:
        _accumulate(torch.arange(E, device=device))   # 미완 에피소드의 살아있는 지도
        env._reset_idx = orig_reset_idx
        np.save(out_dir / "observable_mask.npy", seen_union.cpu().numpy())
        # 같은 고정 자세라도 env마다 GT 표면 voxel이 ±2% 다르다(CAD의 평면이 voxel
        # 경계에 놓여 float 잡음으로 뒤집힘) → 합집합 크기는 분모가 아니다. 분모는
        # env별 `surf_e ∧ mask`이고, 그 비율이 "관측 가능 비율"이다.
        surf = env._surf_vol
        frac = ((seen_union & surf).flatten(1).sum(1).float() / surf.flatten(1).sum(1).clamp(min=1))
        print(f"[eval] 관측 가능 표면(env별 mask∧surf / surf): 평균 {frac.mean():.3f}"
              f" 최소 {frac.min():.3f} 최대 {frac.max():.3f} | 합집합 {int(seen_union.sum())} voxel"
              f" → {out_dir / 'observable_mask.npy'} (--observable_mask 로 분모에 적용)")

    return dict(
        policy=tag,
        episodes=len(ep_rows),
        cov_curves=ep_curves,
        mean_success_length=float(np.mean(
            [r["length"] for r in ep_rows if r["outcome"] == "success"]
        )) if any(r["outcome"] == "success" for r in ep_rows) else float("nan"),
        success_rate=float(np.mean([r["outcome"] == "success" for r in ep_rows])),
        coverage=m("coverage"),
        coverage_std=float(np.std([r["coverage"] for r in ep_rows])),
        n_episodes=len(ep_rows),
        coverage_binary=m("coverage_binary"),
        coverage_belief=m("coverage_belief"),
        mean_obs_dist_m=m("mean_obs_dist_m"),
        gt_never=m("gt_never"), gt_partial=m("gt_partial"), gt_full=m("gt_full"),
        mean_length=float(np.mean([r["length"] for r in ep_rows])),
        mean_return=m("ep_return"),
        # 제어 성능 (step_3 고유)
        pos_err_m=m("pos_err_m", step_rows),
        att_err_deg=m("att_err_deg", step_rows),
        psi_at_max=m("psi_at_max", step_rows),
        phi_at_max=m("phi_at_max", step_rows),
        action_saturated=m("action_saturated", step_rows),
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _max_len(r: dict) -> int:
    return max((len(c) for c in r.get("cov_curves", []) if c), default=0)


def _cov_at(curves: list, k: int, idx: int) -> float:
    """결정 k 시점의 평균 coverage. 이미 끝난 에피소드는 종료값으로 **동결**한다.

    동결이 옳은 이유: 성공해서 끝난 에피소드는 "그 시점에 목표를 달성하고
    관측을 멈춘 것"이므로, 이후 결정에서 값이 더 오르지 않는 것이 사실이다.
    반대로 종료된 에피소드를 평균에서 빼면(= 생존자만 평균) 성공적인 궤적이
    표본에서 사라져 **잘하는 정책일수록 곡선이 내려가는** 착시가 생긴다.
    실제로 2026-09-02 평가의 steps.csv가 그 착시를 보였다(결정 7→8에서
    0.550→0.502로 하락).
    """
    vals = [c[min(k, len(c)) - 1][idx] for c in curves if c]
    return float(np.mean(vals)) if vals else float("nan")



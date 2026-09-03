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


def _quat_angle(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """두 쿼터니언 사이 회전각 [rad]. 부호 모호성을 없애려 |dot|을 쓴다."""
    dot = (q_a * q_b).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


class Policy:
    """액션 생성기. 환경 난수 스트림을 오염시키지 않는 것이 핵심 계약."""

    def __init__(self, name: str, env, device, seed: int, stochastic: bool):
        self.name = name
        self.kind = "ckpt"
        self._device = device
        # 환경과 분리된 Generator — 공정 비교의 전제(모듈 docstring 참조)
        self._gen = torch.Generator(device=device).manual_seed(seed)
        self._stochastic = stochastic

        if name in ("random", "hold", "orbit", "approach"):
            self.kind = name
            self.actor = None
            return

        ckpt = torch.load(name, map_location=device)
        cfg = env.cfg
        self.actor = Actor(
            img_ch=2, scalar_dim=3, action_dim=cfg.action_space,
            H=cfg.visual.h, W=cfg.visual.w,
        ).to(device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.trained_iters = ckpt.get("it", -1)

    @torch.no_grad()
    def act(self, obs, n_env: int, a_dim: int) -> torch.Tensor:
        if self.kind == "hold":
            return torch.zeros(n_env, a_dim, device=self._device)
        if self.kind == "approach":
            # **근접 고착 베이스라인** — psi를 하한까지 밀어붙이고 거기 머문다.
            # 2026-09-02 Stage 2 평가에서 학습 정책이 실제로 수렴한 행동이고,
            # 당시 전역 Q_sat 정규화에서는 이것만으로 orbit을 이겼다. (A)
            # voxel별 정규화가 그 구멍을 막았는지 재는 **회귀 테스트**다:
            # approach가 orbit을 넘으면 정규화가 새고 있다는 뜻이다.
            a = torch.zeros(n_env, a_dim, device=self._device)
            a[:, 2] = -1.0
            return a
        if self.kind == "orbit":
            # 방위각만 최대 속도로 — step_1의 Manual Orbit 대응
            a = torch.zeros(n_env, a_dim, device=self._device)
            a[:, 0] = 1.0
            return a
        if self.kind == "random":
            return torch.rand(
                (n_env, a_dim), generator=self._gen, device=self._device
            ) * 2.0 - 1.0
        if self._stochastic:
            a, _u, _lp, _e = self.actor.sample(
                obs["vox_actor"], obs["img_semantic"], obs["extra_info"])
            return a
        return self.actor.greedy(
            obs["vox_actor"], obs["img_semantic"], obs["extra_info"])


def run_policy(env, policy: Policy, n_episodes: int, seed: int, out_dir: Path) -> dict:
    """한 정책을 n_episodes 만큼 돌리고 에피소드/스텝 기록을 남긴다."""
    device = env.device
    E, A = env.num_envs, env.cfg.action_space

    # 정책마다 동일 에피소드를 보장하기 위해 환경 난수를 되심는다
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset()

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
        cov_now = env._coverage_for_reward()

        step_rows.append(dict(
            decision=decision,
            pos_err_m=p_err.mean().item(),
            pos_err_max_m=p_err.max().item(),
            att_err_deg=math.degrees(q_err.mean().item()),
            coverage=cov_now.mean().item(),
            coverage_binary=env.curr_coverage.mean().item(),
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
                cur_curve[i].append((env.terminal_coverage_q[i].item(),
                                     env.terminal_coverage[i].item()))
            else:
                cur_curve[i].append((cov_now[i].item(),
                                     env.curr_coverage[i].item()))

        for eid in done.nonzero(as_tuple=True)[0].tolist():
            if len(ep_rows) >= n_episodes:
                break
            covq = env.terminal_coverage_q[eid].item()
            covb = env.terminal_coverage[eid].item()
            ep_rows.append(dict(
                episode=len(ep_rows),
                outcome="success" if terminated[eid].item() else "timeout",
                length=int(ep_len[eid].item()),
                ep_return=ep_ret[eid].item(),
                coverage=covq,
                coverage_binary=covb,
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
        coverage_binary=m("coverage_binary"),
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



"""step_3 NBV 정책 평가 — 학습 곡선이 아니라 **실제 거동**을 본다.

`step_1_NBV/evaluate/evaluate_recon.py`의 설계를 따르되, step_3 고유의 질문
두 가지를 추가로 잰다.

왜 필요한가
-----------
2026-08-27 본 학습(`stage2_quality_run01`)이 남긴 미결 질문들은 학습 로그로는
풀리지 않는다:

1. **학습된 정책이 랜덤보다 나은가?** 지금까지 이 비교를 한 적이 없다. coverage가
   올랐다고 해서 시점 선택을 배웠다는 뜻은 아니다 — 액션이 커져서 많이 움직인
   부작용일 수 있다. 같은 에피소드에 대해 여러 정책을 돌려야 판별된다.
2. **`psi=4.5`(=psi_max), `phi=80°`(=phi_max) 포화가 학습된 전략인가,
   tanh 포화로 액션이 극단에 고정된 부산물인가?** 스텝별 궤적을 봐야 한다.
3. **DP 컨트롤러가 5초 안에 목표에 도달하는가?** step_1은 순간이동이라 없던
   질문이다. 도달하지 못하면 "목표 pose에 정착한 뒤 관측한다"는 step_3의 설계
   전제 자체가 깨진다 — 그러면 coverage 수치의 해석이 달라진다.

공정 비교
---------
정책마다 **같은 에피소드**(같은 바위 자세, 같은 스폰)를 주기 위해 정책 실행 전
시드를 다시 심는다. 랜덤 정책은 **별도 Generator**를 써서 환경의 난수 스트림을
건드리지 않는다 — 같은 스트림에서 뽑으면 이후 바위 자세/스폰이 어긋나 비교가
무의미해진다.

사용법
------
python.sh -u evaluate_nbv.py --headless --num_envs 16 --num_episodes 48 \
    --policies checkpoints/nbv_step3_00060.pt,random,hold,orbit \
    --out_dir eval_out/run01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="step_3 NBV 정책 평가")
parser.add_argument("--policies", type=str, default="random,hold,orbit",
                    help="쉼표 구분. 체크포인트 경로 또는 random/hold/orbit "
                         "(hold=액션 0, orbit=최대 속도 방위각 공전)")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=48,
                    help="정책당 평가 에피소드 수")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--out_dir", type=str, default="eval_out")
parser.add_argument("--max_decisions", type=int, default=0,
                    help="에피소드 최대 결정 수 (0=env_cfg 기본값 10). 상한 측정용")
parser.add_argument("--ceiling", action="store_true",
                    help="상한 측정 모드: coverage 종료를 비활성화(임계값 1.1)해서\n"
                         "에피소드가 끝까지 가도록 하고, 결정별 coverage 포화\n"
                         "곡선을 얻는다. 학습으로 얻을 여지가 있는지 판단하는 근거")
parser.add_argument("--stochastic", action="store_true",
                    help="체크포인트 정책을 greedy(tanh(mu)) 대신 샘플링으로 실행")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.env_cfg import NBVBROVEnvCfg
from envs.env import NBVBROVEnv
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

        if name in ("random", "hold", "orbit"):
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
    ep_len = torch.zeros(E, dtype=torch.long, device=device)
    ep_ret = torch.zeros(E, device=device)
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

        done = terminated | truncated
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
                # quality/binary 비 → 평균 관측거리 (Beer-Lambert 역산)
                mean_obs_dist_m=(
                    1.0 - math.log(max(covq / covb, 1e-6)) / env._quality_mu[eid].item()
                ) if covb > 1e-6 else float("nan"),
                gt_never=env._diag_gt_never[eid].item(),
                gt_partial=env._diag_gt_partial[eid].item(),
                gt_full=env._diag_gt_full[eid].item(),
            ))
            ep_len[eid] = 0
            ep_ret[eid] = 0.0

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


def main() -> int:
    cfg = NBVBROVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    # 평가는 고정 난이도에서 — 커리큘럼이 돌면 정책 간 종료 기준이 달라져
    # 비교 자체가 성립하지 않는다.
    cfg.curriculum_enabled = False

    if args.max_decisions > 0:
        cfg.episode_length_s = args.max_decisions * (cfg.sim.dt * cfg.decimation)
    if args.ceiling:
        # 도달 불가 임계값 — coverage 달성으로 조기 종료되면 포화 곡선이 끊긴다
        cfg.coverage_terminal = 1.1
        print("[eval] 상한 측정 모드: coverage 종료 비활성화")

    env = NBVBROVEnv(cfg)
    out_dir = Path(args.out_dir)
    print(f"[eval] envs={args.num_envs} 에피소드/정책={args.num_episodes} "
          f"종료 임계값={cfg.coverage_terminal} (커리큘럼 off)")

    results = []
    try:
        for name in args.policies.split(","):
            name = name.strip()
            if not name:
                continue
            pol = Policy(name, env, env.device, args.seed, args.stochastic)
            print(f"\n[eval] ── {name} ──")
            res = run_policy(env, pol, args.num_episodes, args.seed, out_dir)
            results.append(res)
            print(f"[eval] {res['policy']:>22s}  cov={res['coverage']:.3f} "
                  f"bin={res['coverage_binary']:.3f} succ={res['success_rate']:.2f} "
                  f"len={res['mean_length']:.1f}")
    finally:
        env.close()

    _report(results, out_dir, cfg)
    return 0


def _report(results: list[dict], out_dir: Path, cfg) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'정책':>22s} {'cov_q':>7s} {'cov_bin':>8s} {'성공률':>7s} {'길이':>6s} "
          f"{'거리(m)':>8s} {'gt_full':>8s} {'추종오차(m)':>11s} {'psi포화':>8s} {'액션포화':>9s}")
    print("-" * 100)
    for r in results:
        print(f"{r['policy']:>22s} {r['coverage']:>7.3f} {r['coverage_binary']:>8.3f} "
              f"{r['success_rate']:>7.2f} {r['mean_length']:>6.1f} "
              f"{r['mean_obs_dist_m']:>8.2f} {r['gt_full']:>8.3f} "
              f"{r['pos_err_m']:>11.3f} {r['psi_at_max']:>8.2f} {r['action_saturated']:>9.2f}")
    print("-" * 100)

    base = next((r for r in results if r["policy"] == "random"), None)
    ckpts = [r for r in results if r["policy"] not in ("random", "hold", "orbit")]
    if base and ckpts:
        print("\n[판정] 학습 정책 vs 랜덤")
        for r in ckpts:
            d = r["coverage"] - base["coverage"]
            rel = d / base["coverage"] * 100 if base["coverage"] else float("nan")
            verdict = ("랜덤보다 유의하게 우수" if d > 2 * base["coverage_std"] else
                       "랜덤과 구분 불가 — 시점 선택을 배우지 못했다는 뜻")
            print(f"  {r['policy']}: cov {r['coverage']:.3f} vs 랜덤 {base['coverage']:.3f} "
                  f"({d:+.3f}, {rel:+.1f}%) → {verdict}")
            print(f"    (랜덤 에피소드 표준편차 {base['coverage_std']:.3f} 기준)")

    worst = max(results, key=lambda r: r["pos_err_m"]) if results else None
    if worst:
        print(f"\n[제어] 결정당 위치 추종오차 최대 {worst['pos_err_m']:.3f} m ({worst['policy']}).")
        print("  이 값이 크면 '목표 pose에 정착한 뒤 관측한다'는 전제가 깨진 것이라")
        print("  coverage 수치를 시점 선택 능력으로 해석할 수 없다.")
    print(f"\n[eval] 결과: {out_dir}/summary.json, *_episodes.csv, *_steps.csv")


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)

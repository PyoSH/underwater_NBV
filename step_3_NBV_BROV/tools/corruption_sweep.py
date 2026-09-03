"""②a 허용 오차 스윕 — "정책이 depth/pose 오차를 얼마까지 견디는가".

무엇을 정하는 실험인가
----------------------
배포 depth 파이프라인(TRIDENT+DVL 앵커든 다른 것이든)이 합격인지 판정하려면
**합격선이 먼저 있어야 한다**. 그 선은 센서가 아니라 정책의 성질이므로, 여기서는
센서를 전혀 쓰지 않고 파라메트릭 오염만으로 잰다(`envs/depth_corruption.py`).
산출물은 한 줄이다:

    "AbsRel 0.XX까지는 열화 15% 이내" = 요구사양

그 다음에야 ②b(센서 자격시험)가 "TRIDENT+앵커의 AbsRel이 그 선 안인가"를 묻는다.
이 순서 덕분에 센서가 탈락해도 계획 본체는 무사하고, 센서를 교체해도 이 표는
그대로 쓰인다.

왜 축이 파라미터가 아니라 realized AbsRel인가
---------------------------------------------
`scale_sigma=0.1`은 센서 스펙과 대응되지 않는다. 오염을 적용하면서 실제
|d̂−d|/d를 측정해 두고 그것을 가로축으로 쓰면, TRIDENT 실측치(정렬 없음 0.257 /
앵커 후 0.130 전망)를 같은 축에 얹어 바로 읽을 수 있다.

주의: 채점은 언제나 **진실 대비**다. 오염되는 것은 로봇이 무엇을 봤다고 믿는가
뿐이고 GT surface voxel은 건드리지 않는다 — 그래야 열화가 측정된다.

사용법
------
/isaac-sim/python.sh -u tools/corruption_sweep.py --headless --enable_cameras \
    --policy checkpoints/stage4_A_run01/nbv_step3_00140.pt \
    --mesh_pool ../robots/data/gso_usd/manifest.json \
    --num_envs 16 --num_episodes 32 --max_decisions 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="②a 허용 오차 스윕")
parser.add_argument("--policy", type=str, default="orbit",
                    help="체크포인트 경로 또는 random/hold/orbit/approach. "
                         "학습 정책이 없을 때 orbit으로 하네스만 검증할 수 있다")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=32)
parser.add_argument("--max_decisions", type=int, default=25)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--out_dir", type=str, default="eval_out/corruption_a")
parser.add_argument("--mesh_pool", type=str, default=None)
parser.add_argument("--mesh_pool_split", type=str, default="holdout",
                    choices=("train", "holdout", "all"),
                    help="허용선은 **미학습 물체**에서 재야 배포 조건과 같다")
parser.add_argument("--mesh_pool_offset", type=int, default=0)
parser.add_argument("--axes", type=str, default="scale,noise,pose,combo",
                    help="쉼표 구분. scale/noise/pose/mu/combo")
parser.add_argument("--tolerance", type=float, default=0.15,
                    help="허용 열화 비율. 계획 §5의 distillation 발동 기준과 동일")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from envs.env_cfg import NBVBROVEnvCfg          # noqa: E402
from envs.env import NBVBROVEnv                 # noqa: E402
from eval_core import Policy, run_policy        # noqa: E402


def levels_for(axes: list[str]) -> list[tuple[str, dict]]:
    """오염 조건 사다리. 축마다 **하나씩만** 흔들어 원인을 분리한다.

    combo 두 개는 실제 후보 파이프라인의 예상치다:
      - `combo_anchored`: TRIDENT + DVL 스케일 앵커 (스케일 오차가 대부분 제거된 상태)
      - `combo_raw`: 앵커 없는 TRIDENT 단독 (스케일 오차가 지배)
    이 둘이 허용선의 어느 쪽에 떨어지는지가 곧 ②b의 답이다.
    """
    L: list[tuple[str, dict]] = [("clean", {})]
    if "scale" in axes:
        for sg in (0.05, 0.10, 0.20, 0.30):
            L.append((f"scale{sg:.2f}", dict(scale_sigma=sg)))
    if "noise" in axes:
        for rn in (0.05, 0.10, 0.20):
            L.append((f"noise{rn:.2f}", dict(rel_noise=rn)))
    if "pose" in axes:
        for pd in (0.005, 0.01, 0.02, 0.04):
            L.append((f"pose{pd*100:.1f}cm", dict(pos_drift_std=pd,
                                                  yaw_drift_std=pd * 0.5)))
    if "mu" in axes:
        for lo, hi in ((0.85, 1.2), (0.7, 1.4), (0.5, 2.0)):
            L.append((f"mu{lo}-{hi}", dict(mu_factor_range=(lo, hi))))
    if "combo" in axes:
        L.append(("combo_anchored", dict(scale_sigma=0.10, rel_noise=0.05,
                                         pos_drift_std=0.01, yaw_drift_std=0.005,
                                         mu_factor_range=(0.7, 1.4))))
        L.append(("combo_raw", dict(scale_sigma=0.25, rel_noise=0.08,
                                    pos_drift_std=0.02, yaw_drift_std=0.01,
                                    mu_factor_range=(0.5, 2.0))))
    return L


_FIELDS = ("scale_sigma", "rel_noise", "pos_drift_std", "yaw_drift_std",
           "mu_factor_range")
_DEFAULTS = dict(scale_sigma=0.0, rel_noise=0.0, pos_drift_std=0.0,
                 yaw_drift_std=0.0, mu_factor_range=(1.0, 1.0))


def main() -> int:
    cfg = NBVBROVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum_enabled = False
    cfg.episode_length_s = args.max_decisions * (cfg.sim.dt * cfg.decimation)
    if args.mesh_pool:
        cfg.mesh_pool_manifest = args.mesh_pool
        cfg.mesh_pool_split = args.mesh_pool_split
        cfg.mesh_pool_offset = args.mesh_pool_offset
    # 오염기는 항상 만들어 두고 조건마다 필드만 바꾼다 — 씬을 다시 짓지 않는
    # 것이 핵심이다(씬 생성이 이 실험에서 가장 비싼 부분이고, 물체 배정이
    # 바뀌면 조건 간 비교가 무너진다).
    cfg.corruption.enabled = True

    env = NBVBROVEnv(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[②a] 정책={args.policy} 물체={getattr(env, '_n_mesh_objects', 1)}종"
          f"({args.mesh_pool_split}) 결정≤{args.max_decisions} "
          f"에피소드/조건={args.num_episodes}")

    results = []
    try:
        for name, params in levels_for(args.axes.split(",")):
            for k in _FIELDS:                      # 조건 간 누수 방지
                setattr(env.cfg.corruption, k, params.get(k, _DEFAULTS[k]))
            env.cfg.corruption.enabled = bool(params)
            env._corruptor.clear_stats()
            print(f"\n[②a] ── {name}: {env._corruptor.describe()} ──")

            r = run_policy(env, Policy(args.policy, env, env.device, args.seed),
                           args.num_episodes, args.seed, out_dir / name)
            r["level"] = name
            r["params"] = {k: getattr(env.cfg.corruption, k) for k in _FIELDS}
            r["absrel"] = env._corruptor.realized_absrel
            r.pop("cov_curves", None)
            results.append(r)
            print(f"[②a] {name:>16s}  AbsRel={r['absrel']:.3f}  "
                  f"cov_q={r['coverage']:.3f}  성공률={r['success_rate']:.2f}")
    finally:
        (out_dir / "sweep.json").write_text(json.dumps(results, indent=2))
        env.close()

    # ── 보고 ────────────────────────────────────────────────────────────
    base = results[0]
    print("\n" + "=" * 84)
    print(f"{'조건':>16s}{'AbsRel':>9s}{'cov_q':>8s}{'열화':>8s}"
          f"{'cov_bin':>9s}{'성공률':>8s}{'psi평균':>9s}{'판정':>8s}")
    print("-" * 84)
    for r in results:
        drop = (base["coverage"] - r["coverage"]) / max(base["coverage"], 1e-6)
        ok = drop <= args.tolerance
        print(f"{r['level']:>16s}{r['absrel']:>9.3f}{r['coverage']:>8.3f}"
              f"{drop*100:>7.1f}%{r['coverage_binary']:>9.3f}"
              f"{r['success_rate']:>8.2f}{r['mean_obs_dist_m']:>9.2f}"
              f"{'합격' if ok else '탈락':>8s}")
    print("-" * 84)

    # 허용선: 합격한 조건들 중 realized AbsRel 최댓값
    passed = [r for r in results[1:]
              if (base["coverage"] - r["coverage"]) / max(base["coverage"], 1e-6)
              <= args.tolerance]
    if passed:
        line = max(p["absrel"] for p in passed)
        print(f"\n[요구사양] 열화 {args.tolerance*100:.0f}% 이내를 지키는 "
              f"realized AbsRel 상한 = **{line:.3f}**")
        print(f"  → ②b: 배포 depth 파이프라인의 AbsRel이 이 값 이하여야 한다.")
        print(f"  참고 실측(2026-09-02): TRIDENT 단독 0.257 / 장면별 스케일 보정 후 0.130")
    else:
        print(f"\n[요구사양] 가장 약한 오염에서도 열화 {args.tolerance*100:.0f}% 초과 "
              f"— 어떤 단안 depth로도 이 정책을 그대로 배포할 수 없다는 뜻이다.")
        print(f"  → 계획 §5 distillation을 발동하거나 기하를 재설계할 것.")
    print(f"\n[②a] 결과: {out_dir}/sweep.json")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)

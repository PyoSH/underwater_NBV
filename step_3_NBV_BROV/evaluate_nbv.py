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
parser.add_argument("--policies", type=str, default="random,hold,orbit,approach",
                    help="쉼표 구분. 체크포인트 경로 또는 random/hold/orbit/approach "
                         "(hold=액션 0, orbit=최대 속도 방위각 공전, "
                         "approach=psi 하한까지 접근 후 고착 — 정규화 누수 회귀 테스트)")
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
parser.add_argument("--mesh_pool", type=str, default=None,
                    help="메쉬 풀 manifest 경로. 홀드아웃 평가(= 배포 리허설)에 쓴다")
parser.add_argument("--mesh_pool_limit", type=int, default=0)
parser.add_argument("--mesh_pool_offset", type=int, default=0,
                    help="풀 선택 창을 회전시킨다. 한 실행에서 보는 물체는 "
                         "min(num_envs, 풀)개뿐이라 풀 전체를 훑으려면 여러 번 필요")
parser.add_argument("--mesh_pool_split", type=str, default="holdout",
                    choices=("train", "holdout", "all"),
                    help="기본 holdout — 수조 표적은 정의상 미학습 물체다")
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


from eval_core import Policy, run_policy, _max_len, _cov_at


def main() -> int:
    cfg = NBVBROVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.mesh_pool:
        cfg.mesh_pool_manifest = args.mesh_pool
        cfg.mesh_pool_limit = args.mesh_pool_limit
        cfg.mesh_pool_split = args.mesh_pool_split
        cfg.mesh_pool_offset = args.mesh_pool_offset
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
        # 궤적은 요약이 아니라 원시 데이터라 CSV 쪽에 이미 있다 — json에는
        # 결정별 평균 곡선만 압축해 남긴다.
        json.dump([
            {**{k: v for k, v in r.items() if k != "cov_curves"},
             "cov_q_by_decision": [round(_cov_at(r["cov_curves"], k, 0), 4)
                                   for k in range(1, _max_len(r) + 1)],
             "cov_bin_by_decision": [round(_cov_at(r["cov_curves"], k, 1), 4)
                                     for k in range(1, _max_len(r) + 1)]}
            for r in results
        ], f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'정책':>22s} {'cov_q':>7s} {'cov_bin':>8s} {'성공률':>7s} {'길이':>6s} "
          f"{'psi평균':>8s} {'gt_full':>8s} {'추종오차(m)':>11s} {'psi포화':>8s} {'액션포화':>9s}")
    print("-" * 100)
    for r in results:
        print(f"{r['policy']:>22s} {r['coverage']:>7.3f} {r['coverage_binary']:>8.3f} "
              f"{r['success_rate']:>7.2f} {r['mean_length']:>6.1f} "
              f"{r['mean_obs_dist_m']:>8.2f} {r['gt_full']:>8.3f} "
              f"{r['pos_err_m']:>11.3f} {r['psi_at_max']:>8.2f} {r['action_saturated']:>9.2f}")
    print("-" * 100)

    # ── 같은 결정 수에서의 비교 ──────────────────────────────────────────
    # 종료 시점 coverage끼리 비교하면 "일찍 성공한 정책"이 손해를 본다
    # (성공하면 누적이 멈추고, 실패하는 정책은 25결정까지 계속 쌓는다).
    # 판정의 1차 근거는 반드시 이 표여야 한다.
    ks = [k for k in (3, 5, 7, 10, 15, 20, 25) if k <= max(_max_len(r) for r in results)]
    if ks:
        print("\n[동일 결정 수 비교] cov_q / cov_bin — 종료된 에피소드는 종료값으로 동결")
        print(f"{'정책':>22s}" + "".join(f"{'@'+str(k):>16s}" for k in ks))
        for r in results:
            row = "".join(f"{_cov_at(r['cov_curves'], k, 0):>8.3f}"
                          f"{_cov_at(r['cov_curves'], k, 1):>8.3f}" for k in ks)
            print(f"{r['policy']:>22s}{row}")

    base = next((r for r in results if r["policy"] == "random"), None)
    ckpts = [r for r in results
             if r["policy"] not in ("random", "hold", "orbit", "approach")]
    orbit = next((r for r in results if r["policy"] == "orbit"), None)
    appr = next((r for r in results if r["policy"] == "approach"), None)
    if orbit and appr:
        d = appr["coverage"] - orbit["coverage"]
        print(f"\n[정규화 누수 점검] approach {appr['coverage']:.3f} vs orbit "
              f"{orbit['coverage']:.3f} ({d:+.3f})")
        print("  approach는 psi 하한에 붙어만 있는 정책이다. 이것이 orbit을 넘으면")
        print("  '가까이 가기'가 '돌아보기'를 대체할 수 있다는 뜻 = voxel별 정규화 누수."
              if d > 0 else
              "  → 근접만으로는 공전을 이기지 못한다 = voxel별 정규화가 유지되고 있다.")
    if base and ckpts:
        # 비교 지점: 학습 정책의 평균 성공 길이 — "정책이 과제를 끝냈다고
        # 판단한 시점"이라 정책에 유리하지도 불리하지도 않은 자연스러운 기준.
        print("\n[판정] 학습 정책 vs 랜덤")
        for r in ckpts:
            k = int(round(r.get("mean_success_length", float("nan"))
                          if math.isfinite(r.get("mean_success_length", float("nan")))
                          else r["mean_length"]))
            k = max(1, k)
            a, b = _cov_at(r["cov_curves"], k, 0), _cov_at(base["cov_curves"], k, 0)
            d = a - b
            rel = d / b * 100 if b else float("nan")
            sig = d > 2 * base["coverage_std"]
            print(f"  ① 효율 — 결정 {k}회 시점 cov_q {a:.3f} vs 랜덤 {b:.3f} "
                  f"({d:+.3f}, {rel:+.1f}%) → "
                  f"{'랜덤보다 유의하게 우수' if sig else '랜덤과 구분 불가'}")
            print(f"     성공률 {r['success_rate']:.2f} vs 랜덤 {base['success_rate']:.2f}, "
                  f"성공까지 {r.get('mean_success_length', float('nan')):.1f}결정 vs "
                  f"{base.get('mean_success_length', float('nan')):.1f}결정")
            print(f"     (랜덤 에피소드 표준편차 {base['coverage_std']:.3f} 기준)")

            # ② 이득의 출처: 표면을 더 봤는가(시점 선택) vs 가까이 갔는가(근접)
            ab, bb = _cov_at(r["cov_curves"], k, 1), _cov_at(base["cov_curves"], k, 1)
            ob = _cov_at(orbit["cov_curves"], k, 1) if orbit else float("nan")
            ratio = r["coverage"] / r["coverage_binary"] if r["coverage_binary"] else float("nan")
            print(f"  ② 이득의 출처 — 같은 시점 cov_bin(관측 표면 비율) "
                  f"{ab:.3f} vs 랜덤 {bb:.3f} / orbit {ob:.3f}")
            if math.isfinite(ab) and math.isfinite(ob) and ab <= ob + 0.02:
                print(f"     → 표면은 orbit 이상으로 보지 못했다. cov_q 이득은 "
                      f"**근접**(품질비 {ratio:.2f}, psi 포화)에서 온 것이다.")
                print(f"     cov_q는 Q_sat=exp(-mu*psi_min)이라는 **전역 상수**로만 "
                      f"정규화돼 있어, psi_min보다 가까운 voxel은 1을 넘는 점수를 받고")
                print(f"     그 초과분이 '못 본 voxel'을 상쇄한다 — voxel별 정규화(A)로 "
                      f"막아야 하는 바로 그 경로다.")
            else:
                print(f"     → 표면 관측 자체가 늘었다 = 시점 선택을 배웠다는 증거.")

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

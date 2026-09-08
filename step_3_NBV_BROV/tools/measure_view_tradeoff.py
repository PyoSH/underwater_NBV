"""한 시점에서 관측되는 GT surface voxel 수를 거리별로 잰다.

왜 필요한가
-----------
정책이 psi 상한으로 물러나 "넓게 대충 보기"에 고착한다(run03·run04 실측:
psi 포화 0.75, gt_full 0.001, 그런데 cov_bin은 정상). 이 행동이 이득인지는
**교환비**가 정한다:

    물러날 때 얻는 것 = 한 시점당 관측 voxel 수의 배수
    물러날 때 잃는 것 = voxel당 품질비  exp(-mu * dd)

두 번째는 해석적으로 계산되지만(mu=0.233, psi 1.0->2.5에서 0.705), 첫 번째는
화각·물체 크기·자기 가려짐이 얽혀 있어 **재야 한다**. 2026-09-08에 화면
면적비 6.3배를 그대로 쓴 추정을 했다가, 물체가 유한해 상한이 걸린다는 점을
빠뜨린 것을 발견했다. 그래서 직접 센다.

이 값이 정해지면 보상의 품질 볼록화 지수 p를 산술로 고를 수 있다:

    물러남이 이득  <=>  (voxel 수 배수) * (품질비)^p  >  1
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--mesh_pool", type=str, required=True)
parser.add_argument("--max_solidity", type=float, default=1.0)
parser.add_argument("--quality_model", type=str, default="exp",
                    choices=("exp", "pixel", "nbuv"))
parser.add_argument("--psi_list", type=str, default="1.0,1.3,1.6,2.0,2.5")
parser.add_argument("--n_view", type=int, default=8,
                    help="거리마다 볼 방위 수 (theta 균등)")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, ".")
from envs.env_cfg import NBVBROVEnvCfg   # noqa: E402
from envs.env import NBVBROVEnv          # noqa: E402


def main() -> int:
    cfg = NBVBROVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum_enabled = False
    cfg.use_tiled_camera = False
    cfg.mesh_pool_manifest = args.mesh_pool
    cfg.mesh_pool_split = "train"
    cfg.mesh_pool_max_solidity = args.max_solidity
    cfg.mesh_pool_require_texture = False
    cfg.quality_model = args.quality_model
    # 다양성 시험은 한 에피소드 안에서 누적을 봐야 한다. 종료 임계값을 넘은
    # env가 리셋되면 평균이 떨어져 "누적이 줄어드는" 착시가 난다(2026-09-08
    # nbuv 측정에서 1.6 m 6방위 0.636→0.628). 도달 불가 값으로 끈다.
    cfg.coverage_terminal = 1.1
    env = NBVBROVEnv(cfg)

    psis = [float(x) for x in args.psi_list.split(",")]
    mu = float(env._quality_mu.mean())
    env.reset()          # mu는 리셋에서 확정된다(생성 직후는 플레이스홀더 0.1)
    mu = float(env._quality_mu.mean())
    print(f"\n[tradeoff] model={cfg.quality_model} mu={mu:.3f}  물체 {getattr(env,'_n_mesh_objects',1)}종  "
          f"env {env.num_envs}개  방위 {args.n_view}개/거리")
    print(f"\n{'psi[m]':>7}{'관측voxel':>11}{'표면대비':>9}{'평균품질':>10}"
          f"{'voxel배수':>11}{'품질비':>9}")

    base_n = base_q = None
    rows = []
    for psi in psis:
        n_tot, q_tot, cnt = 0.0, 0.0, 0
        for k in range(args.n_view):
            # 매 시점마다 완전 초기화 — 누적이 아니라 **한 시점의 관측량**을 잰다.
            env.reset()
            th = 2 * math.pi * k / args.n_view
            env._sph_theta[:] = th
            env._sph_phi[:] = (cfg.phi_min + cfg.phi_max) / 2
            env._sph_psi[:] = psi
            act = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
            env.step(act)          # 목표 자세로 이동 + 관측 1회 융합
            surf = env._surf_vol
            seen = (env._weight_vol > 0) & surf
            n = seen.sum(dim=(1, 2, 3)).float()
            qn = (env._quality_vol / env._q_star).clamp(0, 1)
            q = (qn * seen.float()).sum(dim=(1, 2, 3)) / n.clamp(min=1)
            n_tot += float(n.mean()); q_tot += float(q.mean()); cnt += 1
        n_avg, q_avg = n_tot / cnt, q_tot / cnt
        tot_surf = float(env._total_surf_voxels.mean())
        if base_n is None:
            base_n, base_q = n_avg, q_avg
        rows.append((psi, n_avg, n_avg / tot_surf, q_avg, n_avg / base_n, q_avg / base_q))
        print(f"{psi:>7.1f}{n_avg:>11.1f}{n_avg/tot_surf:>9.3f}{q_avg:>10.3f}"
              f"{n_avg/base_n:>11.2f}{q_avg/base_q:>9.3f}", flush=True)

    print(f"\n[tradeoff] psi=1.0 대비 물러남의 손익 (이득 = voxel배수 x 품질비^p)")
    print(f"{'psi':>6}" + "".join(f"{'p='+str(p):>10}" for p in (1, 2, 3, 4)))
    for psi, _, _, _, nr, qr in rows:
        print(f"{psi:>6.1f}" + "".join(f"{nr*qr**p:>10.2f}" for p in (1, 2, 3, 4)))
    print("\n  1보다 크면 물러나는 것이 이득. p를 올려 1 아래로 내리는 것이 목표.")

    # ── 시점 다양성 검증 ────────────────────────────────────────────────
    # 같은 거리에서 방위만 바꿔가며 **누적**했을 때 coverage가 계속 오르는가.
    # cos 항이 있으면 한 시점에서 만점을 받는 voxel이 작은 패치뿐이라
    # 방위를 옮길수록 계속 쌓여야 한다. 거리 항만 있으면 일찍 포화한다.
    print(f"\n[다양성] psi 고정, 방위만 누적 (같은 에피소드 내 max 누적)")
    print(f"{'psi':>6}" + "".join(f"{'방위'+str(k+1):>9}" for k in range(6)))
    for psi in (1.0, 1.6, 2.5):
        env.reset()
        env._sph_phi[:] = (cfg.phi_min + cfg.phi_max) / 2
        env._sph_psi[:] = psi
        act = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        line = f"{psi:>6.1f}"
        for k in range(6):
            env._sph_theta[:] = 2 * math.pi * k / 6
            env.step(act)
            line += f"{float(env._coverage_for_reward().mean()):>9.3f}"
        print(line, flush=True)
    print("  방위를 옮길수록 계속 오르면 시점 다양성 압력이 있는 것이다.")

    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)

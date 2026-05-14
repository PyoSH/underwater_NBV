"""
evaluate_base.py
================
궤도 정책(orbital policy) 베이스라인으로 환경을 실행하고
evaluate_utils.py 의 저장 함수로 3D 복원 결과를 기록.

궤도 정책 설계
--------------
orbital_basic.py 의 AZ_NEG 단일 회전에서 확장:
- STEPS_PER_RING 스텝(= 360° / delta_theta) 동안 theta 순회
- 한 바퀴 완료 시 phi 한 단계 하강 (EL_DOWN)
- phi 가 phi_max 에 도달하면 theta 순회만 반복

액션 슬롯 (algorithm2.make_env_action 기준, action_space=6)
    0: +Δθ  1: -Δθ  2: +Δφ  3: -Δφ  4: +Δψ  5: -Δψ

사용법
------
python evaluate_base.py \
    --num_episodes 3 \
    --out_dir ./recon_output_base
"""

from __future__ import annotations
import argparse, math, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--max_steps",    type=int, default=0)
parser.add_argument("--render",       action="store_true")
parser.add_argument("--out_dir",      type=str, default="./recon_output_base")

AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from envCfg         import OceanEnvCfg
from env            import OceanEnv
from algorithm2     import make_env_action
from evaluate_utils import save_episode_results

# ─────────────────────────────────────────────────────────────────────────────
# 궤도 정책
# ─────────────────────────────────────────────────────────────────────────────
AZ_NEG  = 1   # -Δθ : theta 감소 (azimuth 순회)
EL_DOWN = 2   # +Δφ : phi 증가   (고도 하강, 수평 방향으로)

# 360° / delta_theta(15°) = 24 스텝으로 theta 한 바퀴
STEPS_PER_RING = round(2 * math.pi / math.radians(15))   # = 24


def orbital_action(ep_step: int, E: int, device: torch.device,
                    phi_rings_done: list[int]) -> torch.Tensor:
    """
    에피소드 내 스텝 수(ep_step) 기반 결정론적 궤도 행동.

    ep_step 이 STEPS_PER_RING 의 배수가 되는 순간 EL_DOWN 1회 발행.
    phi 가 phi_max 에 도달한 env(phi_rings_done 표시)는 AZ_NEG 만 계속.

    phi_rings_done : 각 env 의 phi 하강 횟수 (phi_max 도달 여부 판단용)
    """
    pos  = ep_step % STEPS_PER_RING
    ring = ep_step // STEPS_PER_RING

    idxs = []
    for eid in range(E):
        if pos == 0 and ring > 0 and not phi_rings_done[eid]:
            idxs.append(EL_DOWN)
        else:
            idxs.append(AZ_NEG)

    idx = torch.tensor(idxs, dtype=torch.long, device=device)
    return make_env_action(idx, E, device)


# ─────────────────────────────────────────────────────────────────────────────
# 평가 루프
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_base(env: OceanEnv, device: torch.device,
                n_episodes: int, max_steps: int, out_dir: Path):
    E = env.num_envs
    if max_steps == 0:
        max_steps = env.max_episode_length * n_episodes * 2

    # phi_max 에 닿기까지 하강 가능한 링 수
    cfg = env.cfg
    max_rings = int((cfg.phi_max - cfg.phi_min) / cfg.delta_phi)

    env.reset()
    _K = env._camera.data.intrinsic_matrices[0].cpu().numpy()
    K_cache = (float(_K[0,0]), float(_K[1,1]), float(_K[0,2]), float(_K[1,2]))

    ep_return      = torch.zeros(E, device=device)
    ep_len         = torch.zeros(E, device=device, dtype=torch.long)
    ep_step        = [0] * E   # env 별 에피소드 내 스텝 (궤도 정책용)
    phi_rings_done = [False] * E  # phi_max 도달 여부
    phi_ring_cnt   = [0] * E      # 현재까지 EL_DOWN 발행 횟수
    completed      = [0] * E
    ep_counter     = [0] * E

    cam_trajs = [[] for _ in range(E)]
    cov_hists = [[] for _ in range(E)]
    cam_poses = [[] for _ in range(E)]
    rgb_imgs  = [[] for _ in range(E)]

    Nx, Ny, Nz = cfg.tsdf.vol_dim
    tsdf_snap   = [np.zeros((Nx, Ny, Nz), np.float32) for _ in range(E)]
    weight_snap = [np.zeros((Nx, Ny, Nz), np.float32) for _ in range(E)]
    surf_snap   = [np.zeros((Nx, Ny, Nz), bool)        for _ in range(E)]
    origin_snap = [np.zeros(3,            np.float32)  for _ in range(E)]

    with torch.no_grad():
        for _step in range(max_steps):
            if min(completed) >= n_episodes:
                break

            # ── 궤도 행동 생성 ────────────────────────────────────────────
            action = orbital_action(ep_step[0], E, device, phi_rings_done)

            # EL_DOWN 발행 여부 추적 (phi ring 카운터 갱신)
            pos  = ep_step[0] % STEPS_PER_RING
            ring = ep_step[0] // STEPS_PER_RING
            if pos == 0 and ring > 0:
                for eid in range(E):
                    if not phi_rings_done[eid]:
                        phi_ring_cnt[eid] += 1
                        if phi_ring_cnt[eid] >= max_rings:
                            phi_rings_done[eid] = True

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done_any = terminated | truncated

            ep_return += reward
            ep_len    += 1
            for eid in range(E):
                ep_step[eid] += 1

            # ── non-done: 데이터 수집 + 스냅샷 갱신 ─────────────────────
            for eid in range(E):
                if completed[eid] >= n_episodes or done_any[eid]:
                    continue

                cam_trajs[eid].append(env.cam_pos[eid].cpu().numpy().copy())
                cov_hists[eid].append(env.curr_coverage[eid].item())

                cp = env._build_cam_pose()
                cam_poses[eid].append((
                    cp[eid, :3, :3].cpu().numpy().copy(),
                    cp[eid, :3,  3].cpu().numpy().copy(),
                ))
                rgb_imgs[eid].append(
                    env._camera.data.output["uw_rgb"][eid, :, :, :3]
                    .cpu().numpy().copy().astype(np.uint8)
                )

                tsdf_snap  [eid] = env._tsdf_vol  [eid].cpu().numpy().copy()
                weight_snap[eid] = env._weight_vol[eid].cpu().numpy().copy()
                surf_snap  [eid] = env._surf_vol  [eid].cpu().numpy().copy()
                origin_snap[eid] = env._vol_origin[eid].cpu().numpy().copy()

            # ── done: 스냅샷으로 저장 ────────────────────────────────────
            for eid in done_any.nonzero(as_tuple=True)[0].tolist():
                if completed[eid] < n_episodes:
                    status    = "SUCCESS" if terminated[eid].item() else "timeout"
                    final_cov = cov_hists[eid][-1] if cov_hists[eid] else 0.0
                    print(f"  [ep done] env={eid}  ep={ep_counter[eid]:3d}"
                        f"  {status}  len={ep_len[eid].item():.0f}"
                        f"  cov={final_cov:.4f}")

                    save_episode_results(
                        out_dir, ep_counter[eid], eid,
                        tsdf_snap[eid], weight_snap[eid],
                        surf_snap[eid], origin_snap[eid],
                        cfg.tsdf.voxel_size,
                        cam_trajs[eid], cov_hists[eid],
                        cam_poses[eid], rgb_imgs[eid], K_cache,
                        env.rock_pos[eid].cpu().numpy(),
                    )
                    completed[eid]  += 1
                    ep_counter[eid] += 1

                # 다음 에피소드 초기화
                cam_trajs     [eid] = []
                cov_hists     [eid] = []
                cam_poses     [eid] = []
                rgb_imgs      [eid] = []
                ep_step       [eid] = 0
                phi_ring_cnt  [eid] = 0
                phi_rings_done[eid] = False
                ep_return     [eid] = 0.0
                ep_len        [eid] = 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = OceanEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.debug_vis = True
    env_cfg.eval_mode = True
    env_cfg.tsdf.voxel_size = 0.025
    env_cfg.tsdf.vol_dim    = (80, 80, 80)


    env    = OceanEnv(cfg=env_cfg, render_mode="rgb_array" if args.render else None)
    device = env.device

    print(f"\n[base] start  num_envs={args.num_envs}  "
        f"num_episodes={args.num_episodes}")
    max_rings = round((env_cfg.phi_max - env_cfg.phi_min) / env_cfg.delta_phi)
    print(f"[base] policy : orbital  (STEPS_PER_RING={STEPS_PER_RING}, max_rings={max_rings})")
    print(f"[base] output → {out_dir.resolve()}\n")

    evaluate_base(env, device,
                n_episodes=args.num_episodes,
                max_steps=args.max_steps,
                out_dir=out_dir)

    try:
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
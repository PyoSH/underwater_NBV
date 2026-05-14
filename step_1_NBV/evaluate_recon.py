"""
evaluate_recon.py
=================
학습된 체크포인트를 로드해 에피소드를 실행하고,
각 스텝의 깊이+RGB 데이터로 버텍스 컬러 메시를 복원 및 저장.

사용법
------
python evaluate_recon.py \
    --checkpoint ./checkpoints/step_0000993000.pt \
    --num_episodes 3 \
    --out_dir ./recon_output
"""

from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--max_steps",    type=int, default=0)
parser.add_argument("--render",       action="store_true")
parser.add_argument("--out_dir",      type=str, default="./recon_output")

AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from envCfg          import OceanEnvCfg
from env             import OceanEnv
from algorithm2      import Actor, make_env_action
from evaluate_utils  import save_episode_results


# ─────────────────────────────────────────────────────────────────────────────
# 체크포인트 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_actor(checkpoint_path: str, env_cfg: OceanEnvCfg,
                device: torch.device) -> Actor:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    actor = Actor(img_ch=env_cfg.visual.num_seq_actor, scalar_dim=3).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    print(f"[ckpt] loaded → {checkpoint_path}(step={ckpt.get('global_step','?')})")

    return actor


# ─────────────────────────────────────────────────────────────────────────────
# 메인 평가 루프
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_recon(env: OceanEnv, actor: Actor, device: torch.device,
                    n_episodes: int, max_steps: int, out_dir: Path):
    E = env.num_envs
    if max_steps == 0:
        max_steps = env.max_episode_length * n_episodes * 2

    obs, _     = env.reset()
    obs_img    = obs["policy"]
    obs_scalar = obs["extra_info"]

    _K = env._camera.data.intrinsic_matrices[0].cpu().numpy()
    K_cache = (float(_K[0,0]), float(_K[1,1]), float(_K[0,2]), float(_K[1,2]))


    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)
    completed  = [0] * E
    ep_counter = [0] * E

    cam_trajs = [[] for _ in range(E)]
    cov_hists = [[] for _ in range(E)]
    cam_poses = [[] for _ in range(E)]   # list of (R (3,3), t (3,))
    rgb_imgs  = [[] for _ in range(E)]   # list of (H,W,3) uint8

    Nx, Ny, Nz = env.cfg.tsdf.vol_dim
    tsdf_snap   = [np.zeros((Nx, Ny, Nz), np.float32) for _ in range(E)]
    weight_snap = [np.zeros((Nx, Ny, Nz), np.float32) for _ in range(E)]
    surf_snap   = [np.zeros((Nx, Ny, Nz), bool)        for _ in range(E)]
    origin_snap = [np.zeros(3,            np.float32)  for _ in range(E)]

    with torch.no_grad():
        for _step in range(max_steps):
            if min(completed) >= n_episodes:
                break

            pose_act   = actor.greedy(obs_img, obs_scalar)
            env_action = make_env_action(pose_act, E, device)

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done_any = terminated | truncated

            ep_return += reward
            ep_len    += 1

            # ── non-done: 데이터 수집 + 스냅샷 갱신 ─────────────────────────
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

            # ── done: 스냅샷으로 저장 ────────────────────────────────────────
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
                        env.cfg.tsdf.voxel_size,
                        cam_trajs[eid], cov_hists[eid],
                        cam_poses[eid], rgb_imgs[eid], K_cache,
                        env.rock_pos[eid].cpu().numpy(),
                    )
                    completed[eid]  += 1
                    ep_counter[eid] += 1

                cam_trajs[eid] = []
                cov_hists[eid] = []
                cam_poses[eid] = []
                rgb_imgs [eid] = []
                ep_return[eid] = 0.0
                ep_len   [eid] = 0

            obs_img    = next_obs["policy"]
            obs_scalar = next_obs["extra_info"]


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
    env_cfg.tsdf.trunc_margin = 0.025


    env    = OceanEnv(cfg=env_cfg, render_mode="rgb_array" if args.render else None)
    device = env.device
    actor  = load_actor(args.checkpoint, env_cfg, device)

    print(f"\n[recon] start  num_envs={args.num_envs}  "
        f"num_episodes={args.num_episodes}")
    print(f"[recon] output → {out_dir.resolve()}\n")

    evaluate_recon(env, actor, device,
                    n_episodes=args.num_episodes,
                    max_steps=args.max_steps,
                    out_dir=out_dir)

    try:
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

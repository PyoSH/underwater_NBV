"""
evaluate_recon.py
=================
학습된 체크포인트를 로드해 에피소드를 실행하고,
각 스텝의 깊이+RGB 데이터로 버텍스 컬러 메시를 복원 및 저장.

공통 평가 조건 (모든 알고리즘):
  - 환경: OceanEnvGenNBVQuality (quality metric 추적)
  - 종료: stall 5스텝(delta_q < 1e-2) 또는 timeout 50스텝 중 먼저 발생하는 쪽

사용법
------
python evaluate_recon.py \
    --checkpoint ./checkpoints/step_0000993000.pt \
    --num_episodes 30 \
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
parser.add_argument("--num_episodes", type=int, default=30)
parser.add_argument("--max_steps",    type=int, default=0)
parser.add_argument("--render",       action="store_true")
parser.add_argument("--out_dir",      type=str, default="./recon_output")

AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv
import numpy as np
import torch
import isaaclab.sim as sim_utils

sys.path.insert(0, os.path.dirname(__file__))
from envCfg             import OceanEnvCfg
from env_GenNBV_quality import OceanEnvGenNBVQuality
from algorithm3         import Actor, make_env_action
from algo_scanRL        import QNetwork
from algo_GenNBV        import Actor as GenNBVActor
from evaluate_utils     import save_episode_results, save_episode_video

ACTION_NAMES = ["+θ", "-θ", "-φ", "+φ", "-ψ", "+ψ"]

STALL_STEPS   = 5      # 연속 정체 판정 스텝 수
STALL_DELTA_Q = 1e-2   # coverage_q 증가 최소 임계값
TIMEOUT_STEPS = 50     # 에피소드 최대 스텝


def _is_scanrl(ckpt: dict) -> bool:
    return "q_net" in ckpt


def _is_gennbv(ckpt: dict) -> bool:
    return "actor" in ckpt and "embed.geo.conv.0.weight" in ckpt["actor"]


def _is_gennbv_quality(ckpt: dict) -> bool:
    """train_GenNBV_quality.py로 저장된 체크포인트 (ch2=quality float)."""
    return _is_gennbv(ckpt) and ckpt.get("env_type") == "gennbv_quality"


# ─────────────────────────────────────────────────────────────────────────────
# 체크포인트 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
    """
    algo_scanRL / algorithm2 / algorithm3 / algo_GenNBV 체크포인트 모두 지원.

    Returns:
        model       : 네트워크 (eval 모드)
        greedy_fn   : callable → pose_act
        is_gennbv   : bool  — GenNBV 계열 (vox+img 입력) 여부
        quality_vox : bool  — True = quality ch2 (env._get_vox_actor()),
                              False = binary ch2 (occupied surface)
        K_img       : int   — 이미지 채널 수 (ScanRL num_seq_actor 설정용)
    """
    ckpt       = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt.get("args", {})
    use_visit_map = saved_args.get("use_visit_map", False)

    if _is_scanrl(ckpt):
        K_img  = ckpt["q_net"]["cnn.0.weight"].shape[1]
        model  = QNetwork(in_ch=K_img).to(device)
        model.load_state_dict(ckpt["q_net"])
        greedy_fn   = lambda img, scalar: model(img).argmax(dim=-1)
        algo_name   = "scanrl"
        is_gennbv   = False
        quality_vox = False

    elif _is_gennbv(ckpt):
        from algo_GenNBV import SemanticEncoder
        M      = ckpt["actor"]["embed.sem.conv.0.weight"].shape[1]
        n_flat = ckpt["actor"]["embed.sem.proj.weight"].shape[1]
        H = W  = next(
            s for s in range(32, 256)
            if SemanticEncoder(in_ch=M, H=s, W=s).proj.weight.shape[1] == n_flat
        )
        model  = GenNBVActor(img_ch=M, scalar_dim=3, H=H, W=W).to(device)
        model.load_state_dict(ckpt["actor"])
        greedy_fn   = lambda vox, img, scalar: model.greedy(vox, img, scalar)
        quality_vox = _is_gennbv_quality(ckpt)   # quality 여부 자동 판별
        algo_name   = "gennbv_quality" if quality_vox else "gennbv_binary"
        K_img       = M
        is_gennbv   = True

    else:
        K_img  = ckpt["actor"]["cnn.0.weight"].shape[1]
        model  = Actor(img_ch=K_img, scalar_dim=3).to(device)
        model.load_state_dict(ckpt["actor"])
        greedy_fn   = lambda img, scalar: model.greedy(img, scalar)
        algo_name   = "ppo3" if "optimizer_actor" in ckpt else "ppo2"
        is_gennbv   = False
        quality_vox = False

    model.eval()
    hw_str = f"  H=W={H}" if is_gennbv else ""
    print(f"[ckpt] loaded → {checkpoint_path}  "
          f"algo={algo_name}  K_img={K_img}{hw_str}  "
          f"use_visit_map={use_visit_map}  "
          f"step={ckpt.get('global_step','?')}")
    return model, greedy_fn, is_gennbv, quality_vox, K_img


# ─────────────────────────────────────────────────────────────────────────────
# 메인 평가 루프
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_recon(env: OceanEnvGenNBVQuality, greedy_fn, device: torch.device,
                   n_episodes: int, max_steps: int, out_dir: Path,
                   is_gennbv: bool = False, quality_vox: bool = True):
    E = env.num_envs
    if max_steps == 0:
        max_steps = TIMEOUT_STEPS * n_episodes * 2 + 100

    ext_cam    = env.scene["ext_camera"]
    ext_frames = [[] for _ in range(E)]

    obs, _     = env.reset()
    obs_img    = obs["policy"]
    obs_scalar = obs["extra_info"]

    _K = env._camera.data.intrinsic_matrices[0].cpu().numpy()
    K_cache = (float(_K[0,0]), float(_K[1,1]), float(_K[0,2]), float(_K[1,2]))

    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)
    completed  = [0] * E
    ep_counter = [0] * E

    # step_logs: (ep_step, act_idx, cov_bin, cov_q, reward)
    step_logs = [[] for _ in range(E)]

    cam_trajs    = [[] for _ in range(E)]
    cov_hists    = [[] for _ in range(E)]   # binary coverage per step
    cov_q_hists  = [[] for _ in range(E)]   # quality coverage per step
    cam_poses    = [[] for _ in range(E)]
    rgb_imgs     = [[] for _ in range(E)]

    Nx, Ny, Nz  = env.cfg.tsdf.vol_dim
    tsdf_snap   = [np.zeros((Nx, Ny, Nz), np.float32) for _ in range(E)]
    weight_snap = [np.zeros((Nx, Ny, Nz), np.float32) for _ in range(E)]
    surf_snap   = [np.zeros((Nx, Ny, Nz), bool)        for _ in range(E)]
    origin_snap = [np.zeros(3,            np.float32)  for _ in range(E)]

    # 공통 종료 조건: stall 추적
    stall_counter   = [0]   * E
    prev_coverage_q = [0.0] * E

    with torch.no_grad():
        for _step in range(max_steps):
            if min(completed) >= n_episodes:
                break

            # ── 행동 선택 ─────────────────────────────────────────────────────
            if is_gennbv:
                if quality_vox:
                    vox = env._get_vox_actor()      # (E, 3, Nx, Ny, Nz) quality ch2
                else:
                    observed = env._weight_vol > 0
                    vox = torch.stack([
                        (~observed).float(),
                        (observed & (env._tsdf_vol > 0)).float(),
                        (observed & (env._tsdf_vol <= 0)).float(),
                    ], dim=1)
                img = env._image_buffer[:, -2:, :, :]
                pose_act = greedy_fn(vox, img, obs_scalar)
            else:
                pose_act = greedy_fn(obs_img, obs_scalar)
            env_action = make_env_action(pose_act, E, device)

            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            ep_return += reward
            ep_len    += 1

            # ── 공통 종료 조건 계산 ───────────────────────────────────────────
            stall_fired  = torch.zeros(E, dtype=torch.bool, device=device)
            timeout_fired = ep_len >= TIMEOUT_STEPS

            for eid in range(E):
                if completed[eid] >= n_episodes:
                    continue
                cov_q_now = env.curr_coverage_q[eid].item()
                delta_q   = abs(cov_q_now - prev_coverage_q[eid])
                if delta_q < STALL_DELTA_Q:
                    stall_counter[eid] += 1
                else:
                    stall_counter[eid] = 0
                prev_coverage_q[eid] = cov_q_now
                if stall_counter[eid] >= STALL_STEPS:
                    stall_fired[eid] = True

            # env의 done (success or env-internal timeout) + 우리 custom done
            done_any = terminated | truncated | stall_fired | timeout_fired

            # stall/timeout으로 인한 강제 리셋 (env가 자체 done하지 않은 경우)
            force_reset_mask = (stall_fired | timeout_fired) & ~(terminated | truncated)
            force_ids = force_reset_mask.nonzero(as_tuple=True)[0].tolist()
            if force_ids:
                env._reset_idx(force_ids)
                # force-reset envs의 obs_scalar를 새 에피소드 초기값으로 갱신
                fresh_obs = env._get_observations()
                for eid in force_ids:
                    next_obs["extra_info"][eid] = fresh_obs["extra_info"][eid]

            # ── non-done: 데이터 수집 + 스냅샷 갱신 ─────────────────────────
            for eid in range(E):
                if completed[eid] >= n_episodes or done_any[eid]:
                    continue

                cov_bin = env.curr_coverage[eid].item()
                cov_q   = env.curr_coverage_q[eid].item()
                act_idx = pose_act[eid].item()
                rew     = reward[eid].item()

                cam_trajs[eid].append(env.cam_pos[eid].cpu().numpy().copy())
                cov_hists[eid].append(cov_bin)
                cov_q_hists[eid].append(cov_q)

                cp = env._build_cam_pose()
                cam_poses[eid].append((
                    cp[eid, :3, :3].cpu().numpy().copy(),
                    cp[eid, :3,  3].cpu().numpy().copy(),
                ))
                rgb_imgs[eid].append(
                    env._camera.data.output["uw_rgb"][eid, :, :, :3]
                    .cpu().numpy().copy().astype(np.uint8)
                )
                ext_frames[eid].append(
                    ext_cam.data.output["rgb"][eid, :, :, :3]
                    .cpu().numpy().copy().astype(np.uint8)
                )

                tsdf_snap  [eid] = env._tsdf_vol  [eid].cpu().numpy().copy()
                weight_snap[eid] = env._weight_vol[eid].cpu().numpy().copy()
                surf_snap  [eid] = env._surf_vol  [eid].cpu().numpy().copy()
                origin_snap[eid] = env._vol_origin[eid].cpu().numpy().copy()

                step_logs[eid].append(
                    (int(ep_len[eid].item()), act_idx, cov_bin, cov_q, rew)
                )
                print(f"    step={int(ep_len[eid].item()):3d}  "
                      f"act={ACTION_NAMES[act_idx]}({act_idx})  "
                      f"cov_bin={cov_bin:.4f}  cov_q={cov_q:.4f}  rew={rew:+.5f}")

            # ── done: 저장 ───────────────────────────────────────────────────
            for eid in done_any.nonzero(as_tuple=True)[0].tolist():
                if completed[eid] < n_episodes:
                    # curr_coverage_q는 _reset_idx() 이후 이미 0으로 클리어됨.
                    # _terminal_coverage_q는 _reset_idx() 내에서 reset 직전에 저장됨.
                    coverage_q   = env._terminal_coverage_q[eid].item()
                    coverage_bin = cov_hists[eid][-1] if cov_hists[eid] else 0.0
                    steps        = int(ep_len[eid].item())

                    if terminated[eid]:
                        status = "SUCCESS"
                    elif stall_fired[eid]:
                        status = "stall"
                    else:
                        status = "timeout"

                    print(f"  [ep done] env={eid}  ep={ep_counter[eid]:3d}  {status}  "
                          f"steps={steps}  "
                          f"coverage_bin={coverage_bin:.4f}  coverage_q={coverage_q:.4f}")

                    log_dir  = out_dir / f"ep_{ep_counter[eid]:03d}_env{eid}"
                    log_dir.mkdir(parents=True, exist_ok=True)

                    log_path = log_dir / "step_log.csv"
                    with open(log_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["step", "action_idx", "action_name",
                                    "coverage_bin", "coverage_q", "reward"])
                        for s, a, cb, cq, r in step_logs[eid]:
                            w.writerow([s, a, ACTION_NAMES[a],
                                        f"{cb:.6f}", f"{cq:.6f}", f"{r:.6f}"])
                    print(f"  [log] step_log → {log_path}")

                    np.save(str(log_dir / "coverage_bin_hist.npy"),
                            np.array(cov_hists[eid],   dtype=np.float32))
                    np.save(str(log_dir / "coverage_q_hist.npy"),
                            np.array(cov_q_hists[eid], dtype=np.float32))

                    save_episode_results(
                        out_dir, ep_counter[eid], eid,
                        tsdf_snap[eid], weight_snap[eid],
                        surf_snap[eid], origin_snap[eid],
                        env.cfg.tsdf.voxel_size,
                        cam_trajs[eid], cov_hists[eid],
                        cam_poses[eid], rgb_imgs[eid], K_cache,
                        env.rock_pos[eid].cpu().numpy(),
                    )
                    save_episode_video(
                        str(log_dir / "ext_view.mp4"),
                        ext_frames[eid],
                        fps=10,
                    )
                    completed[eid]  += 1
                    ep_counter[eid] += 1

                # per-episode 상태 초기화
                cam_trajs[eid]    = []
                cov_hists[eid]    = []
                cov_q_hists[eid]  = []
                cam_poses[eid]    = []
                rgb_imgs[eid]     = []
                ext_frames[eid]   = []
                ep_return[eid]    = 0.0
                ep_len[eid]       = 0
                step_logs[eid]    = []
                stall_counter[eid]   = 0
                prev_coverage_q[eid] = 0.0

            obs_img    = next_obs["policy"]
            obs_scalar = next_obs["extra_info"]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 체크포인트를 먼저 peek해 env_cfg 구성 결정
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu")
    saved_args    = ckpt_peek.get("args", {})
    use_visit_map = saved_args.get("use_visit_map", False)
    is_scanrl_peek = _is_scanrl(ckpt_peek)
    if is_scanrl_peek:
        K_img = ckpt_peek["q_net"]["cnn.0.weight"].shape[1]
    elif _is_gennbv(ckpt_peek):
        K_img = ckpt_peek["actor"]["embed.sem.conv.0.weight"].shape[1]
    else:
        K_img = ckpt_peek["actor"]["cnn.0.weight"].shape[1]
    num_seq = K_img - (1 if use_visit_map else 0)
    del ckpt_peek

    # ── 환경 설정 ─────────────────────────────────────────────────────────────
    env_cfg = OceanEnvCfg()
    env_cfg.scene.num_envs    = args.num_envs
    env_cfg.debug_vis         = True
    env_cfg.eval_mode         = True
    # 학습과 동일한 voxel / image 해상도 (공정 비교)
    env_cfg.tsdf.vol_dim      = (40, 40, 40)
    env_cfg.tsdf.voxel_size   = 0.05
    env_cfg.tsdf.trunc_margin = 0.05
    env_cfg.visual.h          = 64
    env_cfg.visual.w          = 64
    # ScanRL: 이미지 채널 수 = K_img 프레임; GenNBV quality: 학습 고정값 2
    if is_scanrl_peek:
        env_cfg.visual.num_seq_actor  = K_img
        env_cfg.visual.num_seq_critic = K_img
    else:
        env_cfg.visual.num_seq_actor  = 2
        env_cfg.visual.num_seq_critic = 2

    if use_visit_map:
        env_cfg.visual.num_seq_actor  = num_seq
        env_cfg.visual.num_seq_critic = num_seq
        env_cfg.use_visit_map         = True
        env_cfg.observation_space     = (K_img, env_cfg.visual.h, env_cfg.visual.w)
        env_cfg.state_space           = (K_img, env_cfg.visual.h, env_cfg.visual.w)

    # 공통 quality 파라미터
    env_cfg.coverage_terminal = 0.82
    env_cfg.coverage_bonus    = 30.0
    env_cfg.k_c_q             = 5.0
    env_cfg.k_c               = 0.0
    env_cfg.k_x               = 0.0
    env_cfg.c_step            = 0.02
    env_cfg.k_still           = 0.05
    env_cfg.water_dr_enabled  = False

    from isaaclab.sensors import CameraCfg
    env_cfg.scene.ext_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExtCamera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            clipping_range=(0.1, 30.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-5.5, 0.0, 4.0),
            rot=(0.9239, 0.0, 0.3827, 0.0),
            convention="world",
        ),
    )

    env    = OceanEnvGenNBVQuality(cfg=env_cfg,
                                   render_mode="rgb_array" if args.render else None)
    device = env.device
    model, greedy_fn, is_gennbv, quality_vox, _ = load_model(args.checkpoint, device)

    print(f"\n[recon] start  num_envs={args.num_envs}  "
          f"num_episodes={args.num_episodes}  "
          f"stall_steps={STALL_STEPS}  timeout={TIMEOUT_STEPS}")
    print(f"[recon] output → {out_dir.resolve()}\n")

    evaluate_recon(env, greedy_fn, device,
                   n_episodes=args.num_episodes,
                   max_steps=args.max_steps,
                   out_dir=out_dir,
                   is_gennbv=is_gennbv,
                   quality_vox=quality_vox)

    try:
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

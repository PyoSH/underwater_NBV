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
import matplotlib.image as mpimg

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=6)
parser.add_argument("--max_steps",    type=int, default=10)
parser.add_argument("--render",       action="store_true")
parser.add_argument("--out_dir",      type=str, default="./recon_output_base")
parser.add_argument("--eval_phi",     type=float, default=None,
                    help="초기 고도각 (도 단위). 미지정시 envCfg 기본값(20°) 사용")
parser.add_argument("--eval_psi",     type=float, default=None,
                    help="초기 거리 (m). 미지정시 envCfg 기본값(4.5m) 사용")
parser.add_argument("--psi_max",      type=float, default=None,
                    help="psi 최대 허용 거리 (m). eval_psi보다 크거나 같아야 함")
parser.add_argument("--static_sonar", action="store_true",
                    help="소나 검증 모드: 대상 물체와 동일 고도(phi=90°)에서 정지, 이동 없음")
parser.add_argument("--azimuth_only", action="store_true",
                    help="소나 검증 모드: phi/psi 고정, +Δθ(azimuth) 만 증가")

AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.envCfg              import OceanEnvCfg
from env.env_GenNBV_quality  import OceanEnvGenNBVQuality
from algorithm.algorithm2    import make_env_action
from evaluate.evaluate_utils import save_episode_results, save_episode_video, fuse_highres_tsdf, fuse_highres_quality

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


def static_action(E: int, device: torch.device) -> torch.Tensor:
    """정지 모드: delta_theta/phi/psi=0 이므로 어떤 액션도 이동 없음. action 0 반환."""
    idx = torch.zeros(E, dtype=torch.long, device=device)
    return make_env_action(idx, E, device)


def azimuth_action(E: int, device: torch.device) -> torch.Tensor:
    """azimuth 전용 모드: 항상 action 0 (+Δθ). phi/psi는 delta=0으로 고정."""
    idx = torch.zeros(E, dtype=torch.long, device=device)
    return make_env_action(idx, E, device)


# ─────────────────────────────────────────────────────────────────────────────
# 평가 루프
# ─────────────────────────────────────────────────────────────────────────────
TIMEOUT_STEPS = 50

ACTION_NAMES = ["+θ", "-θ", "+φ", "-φ", "-ψ", "+ψ"]

def evaluate_base(env: OceanEnvGenNBVQuality, device: torch.device,
                n_episodes: int, max_steps: int, out_dir: Path,
                static_mode: bool = False, azimuth_mode: bool = False):
    E = env.num_envs
    timeout = max_steps if max_steps > 0 else TIMEOUT_STEPS

    cfg = env.cfg
    max_rings = int((cfg.phi_max - cfg.phi_min) / cfg.delta_phi) if cfg.delta_phi > 0 else 0

    _K = env._camera.data.intrinsic_matrices[0].cpu().numpy()
    K_cache = (float(_K[0,0]), float(_K[1,1]), float(_K[0,2]), float(_K[1,2]))

    Nx, Ny, Nz = cfg.tsdf.vol_dim

    ext_cam = env.scene["ext_camera"]

    if static_mode:
        print(f"\n[base] start  num_envs={E}  num_episodes={n_episodes}  timeout={timeout}")
        print(f"[base] policy : STATIC (phi=90°, 이동 없음 — 소나 검증 모드)")
        print(f"[base] output → {out_dir.resolve()}\n")
    elif azimuth_mode:
        print(f"\n[base] start  num_envs={E}  num_episodes={n_episodes}  timeout={timeout}")
        print(f"[base] policy : AZIMUTH ONLY (+Δθ per step, phi/psi 고정 — 소나 검증 모드)")
        print(f"[base] output → {out_dir.resolve()}\n")
    else:
        print(f"\n[base] start  num_envs={E}  num_episodes={n_episodes}  timeout={timeout}")
        print(f"[base] policy : orbital  (STEPS_PER_RING={STEPS_PER_RING}  max_rings={max_rings})")
        print(f"[base] output → {out_dir.resolve()}\n")

    TYPES = ["IB", "II", "III", "1C", "3C", "5C"]

    with torch.no_grad():
        for ep_idx in range(n_episodes):
            env.cfg.jerlov_types = (TYPES[ep_idx% len(TYPES)],)
            env.reset()

            if env.cfg.jerlov_dr_enabled:
                jerlov = env._current_jerlov_type[0]
                print(f"  [DR] ep={ep_idx}  type={jerlov}  mu={env._quality_mu[0].item():.4f}  Q_sat={env._quality_Q_sat[0].item():.4f}")
            img_dir = out_dir / f"ep_{ep_idx:03d}_dr_images"
            if env.cfg.jerlov_dr_enabled:
                img_dir.mkdir(parents=True, exist_ok=True)

            sonar_dir = out_dir / f"ep_{ep_idx:03d}_sonar_images"
            sonar_dir.mkdir(parents=True, exist_ok=True)

            phi_rings_done = False
            phi_ring_cnt   = 0

            step_log   = []
            cov_hist   = []
            cov_q_hist = []
            cam_traj   = []
            cam_poses  = []
            rgb_imgs   = []
            depth_imgs = []
            ext_frames = []

            tsdf_snap     = np.zeros((Nx, Ny, Nz), np.float32)
            weight_snap   = np.zeros((Nx, Ny, Nz), np.float32)
            quality_snap  = np.zeros((Nx, Ny, Nz), np.float32)
            surf_snap     = np.zeros((Nx, Ny, Nz), bool)
            origin_snap   = np.zeros(3,             np.float32)
            rock_pos_snap = env.rock_pos[0].cpu().numpy().copy()

            done_reason = "timeout"

            for ep_step in range(1, timeout + 1):
                if static_mode:
                    action  = static_action(E, device)
                    act_idx = 0
                elif azimuth_mode:
                    action  = azimuth_action(E, device)
                    act_idx = 0
                else:
                    # EL_DOWN 카운터 갱신
                    pos  = (ep_step - 1) % STEPS_PER_RING
                    ring = (ep_step - 1) // STEPS_PER_RING
                    if pos == 0 and ring > 0 and not phi_rings_done:
                        phi_ring_cnt += 1
                        if phi_ring_cnt >= max_rings:
                            phi_rings_done = True
                    action  = orbital_action(ep_step - 1, E, device, [phi_rings_done])
                    act_idx = int(action[0].argmax().item()) if action.dim() > 1 else int(action[0].item())

                _, reward, terminated, truncated, _ = env.step(action)

                if env.cfg.water_dr_enabled:
                    for eid in range(E):
                        try:
                            raw = env._camera.data.output["uw_rgb"]
                            img = raw[eid, :, :, :3].cpu().numpy().copy().astype(np.uint8)
                            save_path = str(img_dir / f"step{ep_step:03d}_env{eid}.png")
                            mpimg.imsave(save_path, img)
                            print(f"  [DR img] saved → {save_path}  shape={img.shape}")
                        except Exception as e:
                            print(f"  [DR img] ERROR env{eid} step{ep_step}: {e}")

                # 소나 이미지 저장 (env별, step별)
                sonar_out = env._sonar.data.output.get("sonar_image")
                if sonar_out is not None:
                    for eid in range(E):
                        try:
                            # sonar_out: (N, R, A+1, 4) uint8 — R channel as grayscale
                            simg = sonar_out[eid, :, :-1, 0].cpu().numpy().copy()  # (R, A)
                            save_path = str(sonar_dir / f"step{ep_step:03d}_env{eid}.png")
                            mpimg.imsave(save_path, simg, cmap="gray", vmin=0, vmax=255)
                            print(f"  [sonar] saved → {save_path}  shape={simg.shape}")
                        except Exception as e:
                            print(f"  [sonar] ERROR env{eid} step{ep_step}: {e}")

                is_env_done = terminated[0].item() or truncated[0].item()
                if is_env_done:
                    done_reason = "SUCCESS" if terminated[0] else "timeout_env"
                    break

                cov_bin = env.curr_coverage[0].item()
                cov_q   = env.curr_coverage_q[0].item()
                rew     = reward[0].item()

                cam_traj.append(env.cam_pos[0].cpu().numpy().copy())
                cov_hist.append(cov_bin)
                cov_q_hist.append(cov_q)

                cp = env._build_cam_pose()
                cam_poses.append((
                    cp[0, :3, :3].cpu().numpy().copy(),
                    cp[0, :3,  3].cpu().numpy().copy(),
                ))
                rgb_imgs.append(
                    env._camera.data.output["uw_rgb"][0, :, :, :3]
                    .cpu().numpy().copy().astype(np.uint8)
                )
                _d = env._camera.data.output["distance_to_camera"][0]
                if _d.dim() == 3:
                    _d = _d.squeeze(-1)
                depth_imgs.append(_d.cpu().numpy().copy())
                ext_frames.append(
                    ext_cam.data.output["rgb"][0, :, :, :3]
                    .cpu().numpy().copy().astype(np.uint8)
                )

                tsdf_snap     = env._tsdf_vol   [0].cpu().numpy().copy()
                weight_snap   = env._weight_vol [0].cpu().numpy().copy()
                quality_snap  = env._quality_vol[0].cpu().numpy().copy()
                surf_snap     = env._surf_vol   [0].cpu().numpy().copy()
                origin_snap   = env._vol_origin [0].cpu().numpy().copy()
                rock_pos_snap = env.rock_pos    [0].cpu().numpy().copy()

                step_log.append((ep_step, act_idx, cov_bin, cov_q, rew))
                print(f"    step={ep_step:3d}  act={ACTION_NAMES[act_idx]}({act_idx})  "
                      f"cov_bin={cov_bin:.4f}  cov_q={cov_q:.4f}  rew={rew:+.5f}")

            # ── 에피소드 저장 ─────────────────────────────────────────────────
            import csv
            coverage_bin = cov_hist[-1]   if cov_hist   else 0.0
            coverage_q   = cov_q_hist[-1] if cov_q_hist else 0.0
            n_steps      = len(cov_hist)

            print(f"  [ep done] ep={ep_idx:3d}  {done_reason}  steps={n_steps}  "
                  f"coverage_bin={coverage_bin:.4f}  coverage_q={coverage_q:.4f}")

            log_dir = out_dir / f"ep_{ep_idx:03d}_env0"
            log_dir.mkdir(parents=True, exist_ok=True)

            log_path = log_dir / "step_log.csv"
            with open(log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "action_idx", "action_name",
                            "coverage_bin", "coverage_q", "reward"])
                for s, a, cb, cq, r in step_log:
                    w.writerow([s, a, ACTION_NAMES[a],
                                f"{cb:.6f}", f"{cq:.6f}", f"{r:.6f}"])
            print(f"  [log] step_log → {log_path}")

            np.save(str(log_dir / "coverage_bin_hist.npy"),
                    np.array(cov_hist,   dtype=np.float32))
            np.save(str(log_dir / "coverage_q_hist.npy"),
                    np.array(cov_q_hist, dtype=np.float32))
            np.save(str(log_dir / "quality_vol.npy"),  quality_snap)
            np.save(str(log_dir / "weight_vol.npy"),   weight_snap)
            np.save(str(log_dir / "tsdf_vol.npy"),     tsdf_snap)

            # 고해상도 TSDF + quality 재융합
            tsdf_hi = weight_hi = quality_hi = None
            if depth_imgs and cam_poses:
                tsdf_hi, weight_hi = fuse_highres_tsdf(
                    depth_imgs, cam_poses, K_cache, origin_snap,
                    vol_dim=(80, 80, 80), voxel_size=0.025, trunc_margin=0.025,
                )
            if weight_hi is not None and cam_traj:
                quality_hi = fuse_highres_quality(
                    cam_traj, origin_snap, weight_hi,
                    mu=0.217, voxel_size=0.025,
                )

            if tsdf_hi is not None:
                np.save(str(log_dir / "tsdf_vol_hires.npy"),    tsdf_hi)
                np.save(str(log_dir / "weight_vol_hires.npy"),  weight_hi)
            if quality_hi is not None:
                np.save(str(log_dir / "quality_vol_hires.npy"), quality_hi)

            save_episode_results(
                out_dir, ep_idx, 0,
                tsdf_snap, weight_snap,
                surf_snap, origin_snap,
                cfg.tsdf.voxel_size,
                cam_traj, cov_hist,
                cam_poses, rgb_imgs, K_cache,
                rock_pos_snap,
                coverage_q_hist=cov_q_hist,
                tsdf_hires=tsdf_hi,
                weight_hires=weight_hi,
                voxel_hires=0.025,
            )
            save_episode_video(
                str(log_dir / "ext_view.mp4"),
                ext_frames,
                fps=10,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = OceanEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.debug_vis      = True
    env_cfg.eval_mode      = True

    # evaluate_recon.py와 동일한 TSDF 해상도/품질 파라미터 (학습 설정과 일치)
    env_cfg.tsdf.vol_dim      = (40, 40, 40)
    env_cfg.tsdf.voxel_size   = 0.05
    env_cfg.tsdf.trunc_margin = 0.05
    env_cfg.visual.h          = 64
    env_cfg.visual.w          = 64

    env_cfg.coverage_terminal = 0.65   # max 메트릭 상한 ~0.805의 81%
    env_cfg.coverage_bonus    = 30.0
    env_cfg.k_c_q             = 5.0
    env_cfg.k_c               = 0.0
    env_cfg.k_x               = 0.0
    env_cfg.c_step            = 0.02
    env_cfg.k_still           = 0.05
    env_cfg.water_dr_enabled  = True
    env_cfg.jerlov_dr_enabled = True

    import math as _math

    if args.static_sonar:
        # 소나 검증 모드: 암석과 동일 고도(phi=90°)에서 정지
        # delta를 0으로 설정하면 어떤 액션도 실제 이동을 일으키지 않음
        env_cfg.delta_theta = 0.0
        env_cfg.delta_phi   = 0.0
        env_cfg.delta_psi   = 0.0
        env_cfg.phi_max     = _math.radians(90.0)   # phi=90° 허용
        env_cfg.eval_phi    = _math.radians(90.0)   # 수평 → 암석과 동일 고도
        env_cfg.eval_psi    = args.eval_psi if args.eval_psi is not None else 2.0
        env_cfg.eval_mode   = True
    elif args.azimuth_only:
        # azimuth 전용 모드: phi/psi 고정, +Δθ만 적용
        env_cfg.delta_phi   = 0.0
        env_cfg.delta_psi   = 0.0
        # delta_theta는 기본값(15°) 유지
        env_cfg.phi_max     = _math.radians(90.0)
        env_cfg.eval_phi    = args.eval_phi if args.eval_phi is not None else 90.0
        if isinstance(env_cfg.eval_phi, float) and env_cfg.eval_phi > _math.pi:
            env_cfg.eval_phi = _math.radians(env_cfg.eval_phi)
        else:
            env_cfg.eval_phi = _math.radians(
                args.eval_phi if args.eval_phi is not None else 90.0
            )
        env_cfg.eval_psi    = args.eval_psi if args.eval_psi is not None else 2.0
        env_cfg.eval_mode   = True
    else:
        if args.eval_phi is not None:
            env_cfg.eval_phi = _math.radians(args.eval_phi)
        if args.eval_psi is not None:
            env_cfg.eval_psi = args.eval_psi
        if args.psi_max is not None:
            env_cfg.psi_max = args.psi_max

    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils
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

    env    = OceanEnvGenNBVQuality(cfg=env_cfg, render_mode="rgb_array" if args.render else None)
    device = env.device

    evaluate_base(env, device,
                n_episodes=args.num_episodes,
                max_steps=args.max_steps,
                out_dir=out_dir,
                static_mode=args.static_sonar,
                azimuth_mode=args.azimuth_only)

    try:
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
"""Coverage 축적 검증 — 물체와 평행한 높이에서 azimuth를 돌리며 coverage 상승 확인.

지금까지 스모크테스트는 전부 제어 성능(위치/자세 오차)만 봤고 `curr_coverage`는
계속 0이었다. 그게 "아직 아무것도 못 봤다"인지 "인지 파이프라인이 안 돈다"인지
구분이 안 되므로, 여기서 의도적으로 **coverage가 올라야만 하는 궤적**(대상 물체
주위를 수평으로 공전)을 만들어 실제로 오르는지 확인한다.

phi는 +Z에서 잰 각이므로(offset_z = psi·cos φ) φ=90°가 물체와 같은 높이(수평).
단 cfg.phi_max=80°라 범위 안에서 가장 수평에 가까운 80°를 쓴다.

azimuth(θ)를 매 결정마다 일정량 증가시키면 카메라가 물체 주위를 돌며 새로운
면을 보게 되므로 coverage는 **단조 증가**해야 한다. 오르지 않으면 TSDF 융합
경로(카메라 pose, 투영, depth 샘플링, GT surface voxel) 어딘가가 깨진 것이다.

진단을 위해 중간 지표도 같이 찍는다:
  - `total_surf`   : GT rock 표면 voxel 수 (0이면 메쉬 복셀화 실패)
  - `weight>0`     : 한 번이라도 관측된 voxel 수 (0이면 투영/depth 경로 실패)
  - `depth` 통계   : 카메라가 실제로 물체를 보고 있는지(전부 max range면 헛봄)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--num_decisions", type=int, default=12)
parser.add_argument("--azimuth_action", type=float, default=0.5,
                    help="정책스텝당 theta 증가량 (×max_rate_theta, 기본 0.5=15도)")
parser.add_argument("--psi", type=float, default=2.5, help="공전 반경 [m]")
parser.add_argument("--viewport", action="store_true",
                    help="GUI에 카메라 획득 이미지(uw_rgb) 창 표시")
parser.add_argument("--save_images", action="store_true",
                    help="지정한 결정 시점의 env별 획득 이미지를 PNG로 저장")
parser.add_argument("--save_at", type=int, default=5,
                    help="몇 번째 결정에서 저장할지 (기본 5)")
parser.add_argument("--image_dir", type=str, default="debug_images",
                    help="저장 경로 (컨테이너 기준 상대경로 = 호스트에서도 동일 위치)")
parser.add_argument("--randomize", action="store_true",
                    help="eval_mode를 끄고 rock 자세/스케일 랜덤화 + 랜덤 시작점 사용. "
                         "env마다 다른 장면이 되어 이미지 비교가 의미 있어진다 "
                         "(기본 eval_mode에서는 모든 env가 동일 조건이라 이미지도 같음)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import os

import numpy as np
import torch

from envs.env_cfg import NBVBROVEnvCfg
from envs.env import NBVBROVEnv


def save_env_images(env, out_dir: str, tag: str) -> None:
    """env별 카메라 획득 이미지를 PNG로 저장 (사용자 육안 비교용).

    step_1_NBV/env/env_utils.py::_save_debug_obs()의 저장 방식을 따름
    (cv2, RGB→BGR 변환, depth는 min-max 정규화 후 grayscale).

    저장 항목
    ---------
    uw_rgb   : 수중 감쇠까지 적용된 실제 정책 관측 소스 — 가장 중요
    rgba     : 감쇠 전 원본 렌더 (조명 효과 확인용)
    depth    : TSDF 융합에 쓰이는 depth (정규화된 시각화)
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)

    out = env._camera.data.output
    uw = out["uw_rgb"][..., :3].detach().cpu().numpy()
    raw = out["rgba"][..., :3].detach().cpu().numpy() if "rgba" in out else None
    depth = out["distance_to_camera"].detach().cpu().numpy()
    if depth.ndim == 4:
        depth = depth[..., 0]

    for i in range(env.num_envs):
        cv2.imwrite(
            os.path.join(out_dir, f"{tag}_env{i}_uw_rgb.png"),
            cv2.cvtColor(np.clip(uw[i], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        if raw is not None:
            cv2.imwrite(
                os.path.join(out_dir, f"{tag}_env{i}_rgba.png"),
                cv2.cvtColor(np.clip(raw[i], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
        d = np.nan_to_num(depth[i], nan=0.0, posinf=0.0, neginf=0.0)
        dmin, dmax = float(d.min()), float(d.max())
        vis = (
            ((d - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
            if dmax > dmin else np.zeros_like(d, dtype=np.uint8)
        )
        cv2.imwrite(os.path.join(out_dir, f"{tag}_env{i}_depth.png"), vis)

    print(f"[img] saved {env.num_envs} env × 3 images -> {os.path.abspath(out_dir)}/{tag}_env*_*.png")

cfg = NBVBROVEnvCfg()
cfg.scene.num_envs = args.num_envs
# 고정 시작점: 물체와 (거의) 평행한 높이, 지정 반경
# (--randomize를 주면 eval_mode를 꺼서 rock 자세/스케일과 시작점이 env마다 달라짐)
cfg.eval_mode = not args.randomize
cfg.eval_theta = 0.0
cfg.eval_phi = cfg.phi_max          # 80도 — 범위 내 가장 수평에 가까움
cfg.eval_psi = args.psi
# 에피소드가 도중에 truncate되지 않도록 충분히 길게
cfg.episode_length_s = (args.num_decisions + 2) * (cfg.sim.dt * cfg.decimation)
# 커리큘럼 임계값 변동이 종료조건을 흔들지 않도록 진단 중에는 고정
cfg.curriculum_enabled = False
cfg.enable_camera_viewport = args.viewport

env = None
try:
    env = NBVBROVEnv(cfg)
    obs, _ = env.reset()

    total_surf = env._total_surf_voxels
    print(f"[cov] GT surface voxels per env: {total_surf.tolist()}")
    if float(total_surf.min()) <= 1.0:
        print("[cov] !! GT surface voxel 수가 1 이하 — 메쉬 복셀화 실패 의심")

    action = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    action[:, 0] = args.azimuth_action   # +theta only

    print(f"\n{'결정':>4} {'theta(deg)':>11} {'coverage':>10} {'weight>0':>10} "
          f"{'unk%':>7} {'free%':>7} {'occ%':>6} {'occ/GT':>8} {'ch합오차':>9}")
    print("-" * 84)

    coverages = []
    unknowns = []
    occupieds = []
    for step in range(args.num_decisions):
        env.step(action)

        cov = env.curr_coverage
        observed = (env._weight_vol > 0).sum(dim=(1, 2, 3))
        depth = env._camera.data.output["distance_to_camera"]
        if depth.dim() == 4:
            depth = depth.squeeze(-1)
        finite = depth[torch.isfinite(depth)]
        dmin = float(finite.min()) if finite.numel() else float("nan")
        dmax = float(finite.max()) if finite.numel() else float("nan")
        dmean = float(finite.mean()) if finite.numel() else float("nan")

        # 실제 도달한 azimuth (명령값이 아니라 실측)
        rel = env._robot.data.root_pos_w - env.rock_pos
        theta_actual = torch.atan2(rel[:, 1], rel[:, 0]) % (2 * math.pi)

        # ── voxel 관측 검증 (step_1 방식 궤적으로 vox_actor 거동 확인) ──
        vox = env._get_vox_actor()                      # (E,3,Nx,Ny,Nz)
        n_vox = vox.shape[2] * vox.shape[3] * vox.shape[4]
        unk = vox[:, 0].sum(dim=(1, 2, 3)).mean().item() / n_vox
        fre = vox[:, 1].sum(dim=(1, 2, 3)).mean().item() / n_vox
        occ_cnt = vox[:, 2].sum(dim=(1, 2, 3)).mean().item()
        occ = occ_cnt / n_vox
        # 각 voxel은 정확히 한 상태여야 하므로 채널합은 1
        ch_err = float((vox.sum(dim=1) - 1.0).abs().max())
        occ_ratio = occ_cnt / float(total_surf.mean())

        coverages.append(cov.mean().item())
        unknowns.append(unk)
        occupieds.append(occ_cnt)
        print(f"{step:>4} {math.degrees(theta_actual[0].item()):>11.1f} "
              f"{cov.mean().item():>10.4f} {observed.float().mean().item():>10.0f} "
              f"{unk*100:>7.2f} {fre*100:>7.2f} {occ*100:>6.2f} {occ_ratio:>8.2f} {ch_err:>9.1e}")

        if args.save_images and step == args.save_at:
            save_env_images(env, args.image_dir, tag=f"decision{step:02d}")

    print("-" * 84)
    first, last = coverages[0], coverages[-1]
    increases = sum(
        1 for a, b in zip(coverages, coverages[1:]) if b > a + 1e-6
    )
    print(f"[cov] coverage {first:.4f} -> {last:.4f} "
          f"(증가한 구간 {increases}/{len(coverages)-1})")
    if last > first + 1e-3:
        print("[cov] RESULT: PASS — azimuth 증가에 따라 coverage 상승 확인")
    else:
        print("[cov] RESULT: FAIL — coverage가 오르지 않음, TSDF/인지 경로 점검 필요")

    # ── voxel 관측 판정 (step_1 방식 궤적 기준) ──
    unk_mono = sum(1 for a, b in zip(unknowns, unknowns[1:]) if b < a + 1e-9)
    occ_mono = sum(1 for a, b in zip(occupieds, occupieds[1:]) if b > a - 1e-9)
    print(f"[vox] unknown {unknowns[0]*100:.2f}% -> {unknowns[-1]*100:.2f}% "
          f"(감소 구간 {unk_mono}/{len(unknowns)-1})")
    print(f"[vox] occupied {occupieds[0]:.0f} -> {occupieds[-1]:.0f} voxels "
          f"(증가 구간 {occ_mono}/{len(occupieds)-1}), GT surface={float(total_surf.mean()):.0f}")
    vox_ok = (
        unknowns[-1] < unknowns[0]
        and occupieds[-1] > occupieds[0]
        and unk_mono == len(unknowns) - 1
        and occ_mono == len(occupieds) - 1
    )
    print(f"[vox] RESULT: {'PASS' if vox_ok else 'FAIL'} — "
          "azimuth 증가에 따라 unknown 단조감소 & occupied 단조증가"
          + ("" if vox_ok else " (조건 위반)"))
finally:
    if env is not None:
        env.close()

simulation_app.close()

"""
train_diag.py — 실제 학습 전 환경 진단 스크립트
=================================================
검증 항목:
    STAGE 1. 환경 초기화 & 센서 데이터 키 확인
    STAGE 2. 카메라 내재 파라미터 (intrinsics) 확인
    STAGE 3. 센서리그 위치/방향 검증 (구면좌표계 <-> 월드좌표계 일치 여부)
    STAGE 4. 카메라 forward 방향 검증 (바위를 실제로 바라보는가)
    STAGE 5. 행동(Action) 적용 검증 (θ/φ/ψ 각 축이 의도대로 움직이는가)
    STAGE 6. TSDF 통합 & 커버리지 연산 sanity check
    STAGE 7. 보상 함수 부호 및 스케일 확인
    STAGE 8. 이미지 버퍼 시각화 (UW 렌더링 적용 여부)

실행 방법:
    python train_diag.py --num_envs 1
    python train_diag.py --num_envs 1 --headless   # 헤드리스 모드
"""

import argparse
import sys
import os
import math
import torch
import numpy as np

# ── AppLauncher ───────────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OceanNBV 환경 진단")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--stage", type=int, default=0,
                    help="실행할 스테이지 번호 (0=전체, 1~8=개별)")
AppLauncher.add_app_launcher_args(parser)

if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 이후 import ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from envCfg import OceanEnvCfg
from env    import OceanEnv

from isaaclab.utils.math import quat_apply

# ══════════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "=" * 60
PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [INFO]"
WARN = "  [WARN]"


def _sph_to_world(theta, phi, psi, rock_pos):
    """구면좌표 → 월드 위치 (numpy scalar 입력)."""
    ox = psi * math.sin(phi) * math.cos(theta)
    oy = psi * math.sin(phi) * math.sin(theta)
    oz = psi * math.cos(phi)
    return rock_pos + np.array([ox, oy, oz], dtype=np.float32)


def _quat_to_forward(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    리그의 쿼터니언 [w,x,y,z] (N,4) → 월드 기준 forward 벡터 (N,3).

    Isaac body frame 에서 "앞 방향"이 어느 축인지에 따라
    아래 local_fwd 를 바꿔가며 테스트합니다.
    기본값: +X  (Isaac 관행)
    """
    # ── 테스트할 로컬 forward 후보 ──────────────────────────────────────
    # local_fwd_label = "+X"
    # local_fwd = torch.tensor([[1., 0., 0.]], device=quat_wxyz.device)

    # local_fwd_label = "-X"
    # local_fwd = torch.tensor([[-1., 0., 0.]], device=quat_wxyz.device)

    # local_fwd_label = "+Y"
    # local_fwd = torch.tensor([[0., 1., 0.]], device=quat_wxyz.device)

    # local_fwd_label = "-Y"
    # local_fwd = torch.tensor([[0., -1., 0.]], device=quat_wxyz.device)

    local_fwd_label = "+Z"
    local_fwd = torch.tensor([[0., 0., 1.]], device=quat_wxyz.device)

    # local_fwd_label = "-Z"
    # local_fwd = torch.tensor([[0., 0., -1.]], device=quat_wxyz.device)
    # ─────────────────────────────────────────────────────────────────────

    N = quat_wxyz.shape[0]
    local = local_fwd.expand(N, -1)
    world_fwd = quat_apply(quat_wxyz, local)   # isaaclab 제공 유틸
    return world_fwd, local_fwd_label


def _check_camera_direction(env: OceanEnv, label: str = "") -> float:
    """
    현재 리그 자세로부터 forward → rock 내적을 계산하여 출력.
    내적 값(float) 을 반환.  1.0 에 가까울수록 정렬됨.
    """
    device = env.device

    cam_pos    = env.cam_pos.clone()         # (E, 3)
    cam_orient = env.cam_orient.clone()      # (E, 4)  [w,x,y,z]
    rock_pos   = env.rock_pos.clone()        # (E, 3)

    world_fwd, fwd_label = _quat_to_forward(cam_orient)   # (E, 3)

    to_rock = rock_pos - cam_pos
    to_rock = to_rock / (to_rock.norm(dim=-1, keepdim=True) + 1e-8)

    dot = (world_fwd * to_rock).sum(dim=-1)  # (E,)

    for i in range(env.num_envs):
        prefix = f"  env[{i}]"
        dp     = dot[i].item()
        tag    = PASS if dp > 0.95 else (WARN if dp > 0.0 else FAIL)
        print(f"{prefix} forward({fwd_label})={world_fwd[i].cpu().numpy()}")
        print(f"{prefix} to_rock={to_rock[i].cpu().numpy()}")
        print(f"{prefix} dot={dp:.4f}  {tag}  {label}")

    return dot[0].item()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 : 리셋 후 구면좌표 ↔ 월드 위치 일치 확인
# ══════════════════════════════════════════════════════════════════════════════

def stage1_spherical_consistency(env: OceanEnv):
    print(f"\n{SEP}")
    print(" STAGE 1: 구면좌표 ↔ 월드 위치 일치 확인")
    print(SEP)

    env.reset()
    simulation_app.update()

    for i in range(env.num_envs):
        theta = env._sph_theta[i].item()
        phi   = env._sph_phi[i].item()
        psi   = env._sph_psi[i].item()

        rock_np   = env.rock_pos[i].cpu().numpy()
        expected  = _sph_to_world(theta, phi, psi, rock_np)
        actual    = env.cam_pos[i].cpu().numpy()
        err       = np.linalg.norm(expected - actual)

        print(f"{INFO} env[{i}] θ={math.degrees(theta):.1f}° φ={math.degrees(phi):.1f}° ψ={psi:.2f}m")
        print(f"{INFO} env[{i}] expected={expected}  actual={actual}")
        if err < 1e-3:
            print(f"{PASS} env[{i}] 위치 일치 (오차={err:.6f}m)")
        else:
            print(f"{FAIL} env[{i}] 위치 불일치 (오차={err:.6f}m)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 : 카메라 forward → rock 내적 확인
# ══════════════════════════════════════════════════════════════════════════════

def stage2_camera_forward(env: OceanEnv):
    print(f"\n{SEP}")
    print(" STAGE 2: 카메라 forward 벡터 → rock 내적 확인")
    print(SEP)

    # 리셋 없이 현재 상태 그대로 검사
    _check_camera_direction(env, label="(리셋 직후 상태)")

    # ── sceneCfg offset rot 분석 ──────────────────────────────────────────
    print(f"\n{INFO} sceneCfg camera offset rot = (0.5, -0.5, 0.5, -0.5)  [w,x,y,z]")
    q = torch.tensor([[0.5, -0.5, 0.5, -0.5]], device=env.device)
    for label, v in [("+X", [1,0,0]), ("-X", [-1,0,0]),
                     ("+Y", [0,1,0]), ("-Y", [0,-1,0]),
                     ("+Z", [0,0,1]), ("-Z", [0,0,-1])]:
        lv  = torch.tensor([v], dtype=torch.float, device=env.device)
        wv  = quat_apply(q, lv)[0].cpu().numpy()
        print(f"{INFO}   offset rot 적용 시 로컬 {label} → 월드 {wv}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 : (θ, φ) 그리드 텔레포트 → 매 위치 forward 검증
# ══════════════════════════════════════════════════════════════════════════════

def stage3_teleport_grid(env: OceanEnv):
    print(f"\n{SEP}")
    print(" STAGE 3: (θ, φ) 그리드 텔레포트 후 forward 검증")
    print(SEP)

    device = env.device
    psi    = 3.0   # 고정 거리 [m]

    theta_vals = [0, 90, 180, 270]          # 방위각 [deg]
    phi_vals   = [20, 45, 70]               # 고도각 [deg]

    results = []

    for theta_deg in theta_vals:
        for phi_deg in phi_vals:
            theta = math.radians(theta_deg)
            phi   = math.radians(phi_deg)

            # ── 구면 → 오프셋 ──────────────────────────────────────────
            ox = psi * math.sin(phi) * math.cos(theta)
            oy = psi * math.sin(phi) * math.sin(theta)
            oz = psi * math.cos(phi)
            offset = torch.tensor([[ox, oy, oz]], dtype=torch.float, device=device)

            cam_pos_new = env.rock_pos[0:1] + offset              # env 0 만 사용
            forward     = -offset / (offset.norm(dim=-1, keepdim=True) + 1e-8)
            cam_quat_new = env._forward_to_quat(forward)

            # ── 텔레포트 ──────────────────────────────────────────────
            env._sph_theta[0] = theta
            env._sph_phi[0]   = phi
            env._sph_psi[0]   = psi

            state = torch.zeros(1, 13, device=device)
            state[0, 0:3] = cam_pos_new[0]
            state[0, 3:7] = cam_quat_new[0]
            env_ids_t = torch.tensor([0], device=device)
            env._sensor_rig.write_root_state_to_sim(state, env_ids=env_ids_t)
            simulation_app.update()   # 시뮬 1 스텝 진행하여 PhysX 반영

            # ── 검증 ──────────────────────────────────────────────────
            print(f"\n  ── θ={theta_deg:3d}°  φ={phi_deg:2d}°  ψ={psi}m ──")
            dot = _check_camera_direction(env, label=f"θ={theta_deg}° φ={phi_deg}°")
            results.append((theta_deg, phi_deg, dot))

    # ── 요약 ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STAGE 3 요약 (dot > 0.95 = PASS)")
    print(f"  {'θ':>5} {'φ':>5} {'dot':>8}  결과")
    for td, pd, d in results:
        tag = "PASS" if d > 0.95 else ("WARN" if d > 0.0 else "FAIL")
        print(f"  {td:5}°{pd:5}°{d:8.4f}  {tag}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 : depth map 유효값 분포 확인
# ══════════════════════════════════════════════════════════════════════════════

def stage4_depth_map(env: OceanEnv):
    print(f"\n{SEP}")
    print(" STAGE 4: distance_to_camera depth map 유효값 분포")
    print(SEP)

    cam_output = env._camera.data.output

    if "distance_to_camera" not in cam_output:
        print(f"{FAIL} 'distance_to_camera' 키 없음 — sceneCfg data_types 확인 필요")
        return

    depth = cam_output["distance_to_camera"]           # (E, H, W) or (E, H, W, 1)
    if depth.dim() == 4:
        depth = depth.squeeze(-1)

    E, H, W = depth.shape
    print(f"{INFO} depth shape: {list(depth.shape)}")

    for i in range(E):
        d = depth[i]
        valid = d[d > 0]
        if valid.numel() == 0:
            print(f"{FAIL} env[{i}] 유효 depth 픽셀 없음 (모두 0)")
            continue

        nonzero_ratio = valid.numel() / (H * W)
        print(f"{INFO} env[{i}] 유효 픽셀 비율: {nonzero_ratio*100:.1f}%  "
              f"min={valid.min().item():.3f}m  "
              f"max={valid.max().item():.3f}m  "
              f"mean={valid.mean().item():.3f}m")

        # 바위 예상 거리 (psi) 와 depth 평균 비교
        psi = env._sph_psi[i].item()
        mean_d = valid.mean().item()
        rel_err = abs(mean_d - psi) / (psi + 1e-8)
        tag = PASS if rel_err < 0.5 else WARN
        print(f"{tag} env[{i}] depth 평균({mean_d:.2f}m) vs ψ({psi:.2f}m)  상대오차={rel_err*100:.1f}%")

        # 카메라 뒤쪽 픽셀 존재 여부 (음수 depth)
        behind = (d < 0).sum().item()
        if behind > 0:
            print(f"{WARN} env[{i}] 카메라 뒤쪽 픽셀 {behind}개 존재 — 방향 반전 의심")
        else:
            print(f"{PASS} env[{i}] 카메라 뒤쪽 픽셀 없음")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

STAGES = {
    1: stage1_spherical_consistency,
    2: stage2_camera_forward,
    3: stage3_teleport_grid,
    4: stage4_depth_map,
}

if __name__ == "__main__":
    cfg = OceanEnvCfg()
    cfg.scene.num_envs = 1      # 진단은 단일 env 로 충분
    cfg.debug_vis      = True
 
    env = OceanEnv(cfg=cfg, render_mode="rgb_array")
    env.reset()
    simulation_app.update()
 
    # failed = []
    # for s in sorted(STAGES):
    #     try:
    #         STAGES[s](env)
    #     except Exception as e:
    #         import traceback
    #         print(f"{FAIL} STAGE {s} 예외 발생: {e}")
    #         traceback.print_exc()
    #         failed.append(s)
 
    # print(f"\n{SEP}")
    # if failed:
    #     print(f"  실패 스테이지: {failed}")
    # else:
    #     print("  모든 스테이지 완료")
    # print(f"  뷰포트 유지 중 — 종료하려면 Ctrl+C")
    # print(SEP)
 
    while simulation_app.is_running():
        simulation_app.update()
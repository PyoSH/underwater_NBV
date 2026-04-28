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

def run_sequential_policy(env: OceanEnv, steps: int = 1000) -> None:
    print("시작", flush=True)
    # Action index constants (matches your cfg order)
    AZ_POS, AZ_NEG = 0, 1
    EL_DOWN, EL_UP = 2, 3
    DI_FAR, DI_NEAR = 4, 5
    LI_DEC, LI_HLD, LI_INC = 6, 7, 8

    light_direction = +1 # LI_POS

    # 1. reset
    obs, _ = env.reset()  # DirectRLEnv는 obs, info 반환 [web:49]
    # prev_uw = None

    for t in range(steps):
        curr_light_level: int = env._light_level[0].item()

        a = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)

        a[:, AZ_NEG] = 1.0
        # a[:, ]

        if curr_light_level >= 8:
            light_direction = -1
        elif curr_light_level <= 1:
            light_direction = +1
            
        a[:, LI_INC if light_direction == +1 else LI_DEC] = 1.0

        obs, reward, terminated, truncated, info = env.step(a)  # [web:49]

        cam_pos = env.cam_pos[0].cpu().numpy()
        print(
            f"[step {t+1:4d}] "
            f"terminated={terminated[0].item()} "
            f"truncated={truncated[0].item()} "
            f"reward={reward[0].item():.4f} "
            f"cam=({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f})",
            flush=True,
        )

        # episode 종료 시 reset
        if terminated.any() or truncated.any():
            obs, _ = env.reset()
            prev_uw = None

    print("[train_random] run_random_policy 종료", flush=True)


if __name__ == "__main__":
    cfg = OceanEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.dt = 1.0 / 30.0

    env = OceanEnv(cfg=cfg, render_mode="rgb_array")

    try:
        run_sequential_policy(env, steps=10)
    finally:
        env.close()
        simulation_app.close()
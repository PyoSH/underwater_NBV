"""
orbital_policy.py — env.py _apply_action() 기반 궤도 순회
===========================================================

액션 구조 (action_space=9, one-hot argmax 방식):
    [0] θ+  [1] θ-  [2] φ+  [3] φ-  [4] ψ+  [5] ψ-
    [6] light-  [7] light 유지  [8] light+

좌표계 (env.py 실제 구현 기준):
    x = ψ * sin(φ) * cos(θ)
    y = ψ * sin(φ) * sin(θ)
    z = ψ * cos(φ)

    φ=0   → 정수직 위 (z축 최대)
    φ=90° → xy평면 (수평)
    
궤도 설계:
    4개 φ 레벨에서 θ를 0→2π 균등 순회
    각 레벨 내 ψ와 φ는 고정, θ만 delta_theta씩 증가
"""

import torch

def run_sequential_policy(env, steps: int = 1000) -> None:
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
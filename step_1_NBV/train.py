import argparse
import sys
import os
import torch

# ── AppLauncher (Isaac Sim 시작) ───────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OceanNBV RL 환경 학습")
parser.add_argument("--num_envs", type=int, default=1, help="병렬 환경 수 (현재 1만 지원)")
AppLauncher.add_app_launcher_args(parser)  # --headless 등 Isaac Lab 공통 인수 추가

if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

if "--headless" not in sys.argv:
    sys.argv.append("--headless")
    sys.argv.append("--livestream 2")
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 이후 import ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from envCfg import OceanEnvCfg
from env    import OceanEnv


def run_random_policy(env: OceanEnv, steps: int = 100) -> None:
    """랜덤 정책으로 환경 동작을 검증한다."""
    import traceback

    print("[train] 랜덤 정책 실행 중...", flush=True)

    try:
        obs, _ = env.reset()
    except Exception:
        print("[ERROR] env.reset() 실패:", flush=True)
        traceback.print_exc()
        return

    total_reward = 0.0

    for step in range(steps):
        try:
            # 랜덤 행동 [-1, 1]^6
            action = torch.rand(env.num_envs, env.cfg.action_space, device=env.device) * 2 - 1
            # action = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception:
            print(f"[ERROR] step={step} 에서 오류:", flush=True)
            traceback.print_exc()
            return

        total_reward += reward.sum().item()

        if (step + 1) % 100 == 0:
            cam  = env.cam_pos[0].cpu().numpy()
            dist = (env.cam_pos[0] - env.rock_pos[0]).norm().item()
            print(
                f"  step={step+1:4d} | "
                f"cam=({cam[0]:.2f},{cam[1]:.2f},{cam[2]:.2f}) | "
                f"dist_to_rock={dist:.2f}m | "
                f"reward={reward[0]:.3f}",
                flush=True,
            )

        if terminated.any() or truncated.any():
            obs, _ = env.reset()

    print(f"[train] 완료. 총 누적 보상: {total_reward:.2f}")

if __name__ == "__main__":
    cfg = OceanEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 10.0

    env = OceanEnv(cfg)

    try:
        run_random_policy(env, steps=10000)
    finally:
        env.close()
        simulation_app.close()

"""
train.py
--------
강화학습 진입점.

실행 (컨테이너 내부):
    /isaac-sim/python.sh train.py              # GUI 모드
    /isaac-sim/python.sh train.py --headless   # 헤드리스

RL 라이브러리 연동:
    현재는 환경 동작 확인용 랜덤 정책으로 실행됨.
    실제 학습은 아래 "PPO 연동 예시" 주석 블록을 활성화할 것.
"""

import argparse
import sys
import os
import torch

# ── AppLauncher (Isaac Sim 시작) ───────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OceanNBV RL 환경 학습")
parser.add_argument("--num_envs", type=int, default=1, help="병렬 환경 수 (현재 1만 지원)")
AppLauncher.add_app_launcher_args(parser)  # --headless 등 Isaac Lab 공통 인수 추가

# Camera 센서 사용을 위해 RTX 렌더링 활성화
# parse_args() 전에 sys.argv 에 추가해야 AppLauncher 가 정상 인식함
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 이후 import ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from envCfg import OceanNBVEnvCfg
from env    import OceanNBVEnv


def run_random_policy(env: OceanNBVEnv, steps: int = 100) -> None:
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


# ── PPO 연동 예시 (skrl) ──────────────────────────────────────────────────────
# skrl 사용 시 아래 블록을 활성화하고 `pip install skrl` 필요.
#
# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
# from skrl.trainers.torch    import SequentialTrainer
# from skrl.models.torch      import DeterministicMixin, GaussianMixin, Model
# import torch.nn as nn
#
# class Policy(GaussianMixin, Model):
#     def __init__(self, obs_space, act_space, device):
#         Model.__init__(self, obs_space, act_space, device)
#         GaussianMixin.__init__(self, clip_actions=True)
#         self.net = nn.Sequential(
#             nn.Linear(9, 64), nn.Tanh(),
#             nn.Linear(64, 64), nn.Tanh(),
#             nn.Linear(64, act_space.shape[0]),
#         )
#     def compute(self, inputs, role):
#         return self.net(inputs["states"]), self.log_std_parameter, {}
#
# class Value(DeterministicMixin, Model):
#     def __init__(self, obs_space, act_space, device):
#         Model.__init__(self, obs_space, act_space, device)
#         DeterministicMixin.__init__(self)
#         self.net = nn.Sequential(
#             nn.Linear(9, 64), nn.Tanh(),
#             nn.Linear(64, 64), nn.Tanh(),
#             nn.Linear(64, 1),
#         )
#     def compute(self, inputs, role):
#         return self.net(inputs["states"]), {}
#
# cfg  = OceanNBVEnvCfg()
# env  = OceanNBVEnv(cfg)
# models = {"policy": Policy(...), "value": Value(...)}
# agent = PPO(models=models, cfg=PPO_DEFAULT_CONFIG, env=env)
# trainer = SequentialTrainer(cfg={"timesteps": 100_000}, env=env, agents=agent)
# trainer.train()
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    cfg = OceanNBVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 10.0

    env = OceanNBVEnv(cfg)

    try:
        run_random_policy(env, steps=10000)
    finally:
        env.close()
        simulation_app.close()

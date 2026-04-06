import argparse
import sys
import os

from isaaclab.app import AppLauncher

# 인수 파싱
parser = argparse.ArgumentParser(description="BROV2 Bottom-Up 검증")
parser.add_argument("--test", choices=["neutral_buoyancy", "straight_line", "thruster_model"], default="neutral_buoyancy")
parser.add_argument("--thrust", type=float, default=0.5)
parser.add_argument("--duration", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 환경 및 검증 모듈 import
sys.path.insert(0, os.path.dirname(__file__))
from envCfg import BROVTrajEnvCfg
from env import BROVTrajEnv
import bottom_up  # 분리한 파일 import

if __name__ == "__main__":
    cfg = BROVTrajEnvCfg()
    cfg.scene.num_envs = 1
    cfg.max_bound_x = cfg.max_bound_y = cfg.max_bound_z = 50.0

    env = BROVTrajEnv(cfg)

    try:
        if args.test == "neutral_buoyancy":
            bottom_up.test_neutral_buoyancy(env, duration_s=args.duration)
        elif args.test == "straight_line":
            bottom_up.test_straight_line(env, thrust=args.thrust, duration_s=args.duration)
        elif args.test == "thruster_model":
            bottom_up.test_thruster_model(env, duration_s=args.duration)
    finally:
        env.close()
        simulation_app.close()
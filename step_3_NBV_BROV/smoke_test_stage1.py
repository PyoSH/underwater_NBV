"""Stage 1 스켈레톤 구성/스텝 스모크테스트 — import/구성/reset/step만 검증한다.
`python.sh smoke_test_stage1.py --headless` (isaac-lab-base 컨테이너 안).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=5)
parser.add_argument(
    "--action", type=float, nargs=3, default=[0.0, 0.0, 0.0],
    help="(dtheta,dphi,dpsi) in [-1,1] applied for --nudge_steps, then zero "
         "for the rest -- mirrors real NBV usage (one viewpoint change, then "
         "hold/observe), unlike a sustained non-stop target rotation",
)
parser.add_argument("--nudge_steps", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# step_1_NBV/train/train_GenNBV_quality.py와 동일 관례: 진짜 headless kit은
# omni.ui가 빠져 있어 sensors/UWCamera가 import 실패한다 — Xvfb 가상 디스플레이
# 위에서 일반(비-headless) 커널 익스피리언스를 그대로 띄운다.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from envs.env_cfg import NBVBROVEnvCfg
from envs.env import NBVBROVEnv

cfg = NBVBROVEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.debug_vis = False

# 예외 발생 시에도 반드시 env.close()가 실행되도록 try/finally로 감싼다 —
# 감싸지 않으면 uncaught exception 이후 SimulationApp 정리가 안 돼 프로세스가
# GPU/렌더 스레드를 계속 붙든 채 아무 출력도 없이 몇 시간이고 멈춰있을 수
# 있다(이번 세션에서 실제로 겪음 — 9시간 hang, 원인은 트레이스백 자체가 아니라
# 이 finally 누락).
env = None
try:
    env = NBVBROVEnv(cfg)
    print(f"[smoke] env constructed, num_envs={env.num_envs}, device={env.device}")

    obs, _ = env.reset()
    print(f"[smoke] reset OK — policy obs shape={obs['policy'].shape}, extra_info shape={obs['extra_info'].shape}")
    print(f"[smoke] curr_coverage after reset: {env.curr_coverage.tolist()}")

    action_row = torch.tensor(args.action, device=env.device)
    zero_row = torch.zeros_like(action_row)
    for step in range(args.num_steps):
        row = action_row if step < args.nudge_steps else zero_row
        actions = row.unsqueeze(0).expand(env.num_envs, -1).clone()
        obs, rew, terminated, truncated, info = env.step(actions)
        pos_err = (env._guidance.p_target - env._robot.data.root_pos_w).norm(dim=-1)
        import isaaclab.utils.math as math_utils
        att_err_deg = math_utils.quat_error_magnitude(
            env._guidance.q_target, env._robot.data.root_quat_w
        ) * 180.0 / 3.14159265
        print(
            f"[smoke] step={step} reward={rew.tolist()} "
            f"terminated={terminated.tolist()} truncated={truncated.tolist()} "
            f"pos_err_to_target={pos_err.tolist()} att_err_deg={att_err_deg.tolist()} "
            f"root_pos={env._robot.data.root_pos_w.tolist()}"
        )
        if torch.isnan(rew).any() or torch.isinf(rew).any():
            raise RuntimeError("reward contains NaN/Inf")
        if torch.isnan(obs['policy']).any():
            raise RuntimeError("policy obs contains NaN")

    print("[smoke] PASSED")
finally:
    if env is not None:
        env.close()

simulation_app.close()

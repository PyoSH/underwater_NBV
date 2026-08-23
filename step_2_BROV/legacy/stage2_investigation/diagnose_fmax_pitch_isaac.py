"""
Isaac-native A/B: does raising simulated pitch F_max reduce deploy_v5's
pitch-axis saturation? Diagnostic only -- deploy_v5_pitch_fmax_diag's F_max
(28 N*m pitch) exceeds the measured real-hardware value and must never be
exported into a real/MK2-deployable artifact.

Rolls out a batch of parallel envs on the exact Case-A geometry (2.0 m
out-and-back, takeoff_then_align) so results are directly comparable to the
Gazebo diagnose_attitude_torque_budget.py numbers already measured for
deploy_v4/deploy_v5, without needing the Gazebo/MAVLink round-trip that
isn't relevant to this specific torque-budget question.

Usage (inside isaac-lab-base, same invocation convention as train.py):
python.sh diagnose_fmax_pitch_isaac.py \
  --checkpoint logs/<exp>/model_299.pt --profile deploy_v5 \
  --num_envs 256 --duration 20 --output out.json --headless
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--profile",
    choices=["deploy_v4", "deploy_v5", "deploy_v5_pitch_fmax_diag"],
    required=True,
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from importlib.metadata import version as _pkg_version

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from envs.vel_env_cfg import BROVVelEnvCfg, apply_training_profile
from envs.vel_env import BROVVelEnv
from guidance.los_guidance import LOSGuidance
from agents.rsl_rl_ppo_cfg import BROVVelPPORunnerCfg

_STARTING_DEPTH = 10.0
_AXIS_IDX = {"surge": 0, "sway": 1, "heave": 2, "roll": 3, "pitch": 4, "yaw": 5}
_F_MAX = {"roll": 26.0, "pitch": 14.0, "yaw": 22.0}


def main() -> None:
    print("[DEBUG] building env_cfg", flush=True)
    env_cfg = apply_training_profile(BROVVelEnvCfg(), args.profile)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.starting_depth = _STARTING_DEPTH
    env_cfg.episode_length_s = args.duration + 5.0
    # starting_depth=10.0 places envs exactly at the boundary if max_bound
    # were also 10.0 -- envs would terminate out_of_bounds almost
    # immediately and every logged step would be a just-post-reset state
    # (this actually happened on the first run of this script: near-zero
    # action_mean_abs on every axis was this bug, not a real result).
    # Match test_policy.py's own relaxed bound for the same reason.
    env_cfg.max_bound = 30.0

    print(f"[DEBUG] profile={args.profile} f_max={env_cfg.f_max}", flush=True)
    print("[DEBUG] constructing BROVVelEnv", flush=True)
    env = BROVVelEnv(cfg=env_cfg, render_mode=None)

    z = _STARTING_DEPTH
    # Exact Case-A geometry: 0,0,0 -> 0,0,0.20 (takeoff) -> 2.0,0,0.20
    # (relative to starting depth), looped 0->1->2->1 via takeoff_then_align.
    waypoints = torch.tensor(
        [[0.0, 0.0, z], [0.0, 0.0, z + 0.20], [2.0, 0.0, z + 0.20]],
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1, -1).clone()
    los = LOSGuidance(
        waypoints,
        env.device,
        cruise_speed=0.50,
        lookahead_dist=0.40,
        reach_threshold=0.15,
        heading_mode="takeoff_then_align",
    )
    env.attach_guidance(los)

    agent_cfg = BROVVelPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    print(f"[INFO] checkpoint loaded: {args.checkpoint}", flush=True)

    obs_dict, _ = env.reset()
    num_steps = int(args.duration / env._policy_dt)
    action_log: list[torch.Tensor] = []
    omega_log: list[torch.Tensor] = []
    reset_count = 0
    print(f"[DEBUG] rollout: {num_steps} steps x {env.num_envs} envs", flush=True)
    with torch.inference_mode():
        for i in range(num_steps):
            actions = policy(obs_dict)
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            reset_count += int((terminated | truncated).sum().item())
            action_log.append(env._actions.clone().cpu())
            omega_log.append(env._robot.data.root_ang_vel_b.clone().cpu())
            if i % 50 == 0:
                print(f"[DEBUG] step {i}/{num_steps}", flush=True)
    reset_fraction = reset_count / (num_steps * args.num_envs)
    print(f"[DEBUG] reset_count={reset_count} reset_fraction={reset_fraction:.4f}", flush=True)

    env.close()

    action_all = torch.cat(action_log, dim=0).numpy()  # (steps*envs, 6)
    omega_all = torch.cat(omega_log, dim=0).numpy()    # (steps*envs, 3) rad/s

    per_axis = {}
    for name in ("roll", "pitch", "yaw"):
        i = _AXIS_IDX[name]
        oi = {"roll": 0, "pitch": 1, "yaw": 2}[name]
        a = action_all[:, i]
        pinned = np.abs(a) >= 0.99
        om_deg_s = np.degrees(np.abs(omega_all[:, oi]))
        per_axis[name] = {
            "f_max_Nm": _F_MAX[name],
            "action_pinned_ge_0p99_fraction": float(np.mean(pinned)),
            "action_mean_abs": float(np.mean(np.abs(a))),
            "body_rate_abs_deg_s_while_pinned_mean": (
                float(np.mean(om_deg_s[pinned])) if np.any(pinned) else None
            ),
            "body_rate_abs_deg_s_while_not_pinned_mean": (
                float(np.mean(om_deg_s[~pinned])) if np.any(~pinned) else None
            ),
        }

    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "profile": args.profile,
        "f_max": list(env_cfg.f_max),
        "num_envs": args.num_envs,
        "num_steps": num_steps,
        "total_samples": int(action_all.shape[0]),
        "reset_count": reset_count,
        "reset_fraction": reset_fraction,
        "per_axis": per_axis,
        "action_any_axis_ge_0p99_fraction": float(
            np.mean(np.any(np.abs(action_all[:, 3:6]) >= 0.99, axis=1))
        ),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise

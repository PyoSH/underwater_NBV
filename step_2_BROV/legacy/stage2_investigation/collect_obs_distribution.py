"""
학습 시점 관측 분포 수집 — 16-dim(q_e/v_e_b/omega_b/z_v/z_q) 분포를
학습된 정책+학습용 커맨드 스케줄러(LOSGuidance 미부착 = attach_guidance() 호출 안 함)로
롤아웃해서 수집한다. `attach_guidance()`를 부착하지 않으면 `_current_v_d_b()`가
`_deploy_scheduler`(deploy_v2 학습 커맨드 분포, 0/0.1/0.5 m/s bin + 180도 반전 +
정지/재시동)를 그대로 쓰므로, 이 결과가 "정책이 학습 중 실제로 본 관측 분포"다.

목적: MK2_SIM2SIM_DEPLOY_RESULT.md가 진단한 "Gazebo 관측 재생 시 raw actor 출력이
99% 확률로 [-1,1] 밖"이라는 결과를 더 파고들어, 16차원 중 어느 성분이 실기/Gazebo
관측에서 학습 분포를 가장 크게 벗어나는지 특정하기 위한 기준선(baseline) 분포.

사용법 (isaac-lab-base 컨테이너 안, train.py/test_policy.py와 동일 실행 방식)
--------------------------------------------------------------
python.sh collect_obs_distribution.py \
  --checkpoint logs/stage3_deploy_v2_2048x128_seed42_i300_20260817/model_299.pt \
  --profile deploy_v2 --num_envs 2048 --num_steps 256 --headless
"""

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="deploy_v2 학습 관측 분포 수집")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL model_*.pt 체크포인트 경로")
parser.add_argument(
    "--profile",
    choices=["legacy_exact", "paper_ref_v1", "deploy_v2", "deploy_v3", "deploy_v4", "deploy_v5"],
    default="deploy_v2",
)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--num_steps", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default=None)
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
from agents.rsl_rl_ppo_cfg import BROVVelPPORunnerCfg


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_LABELS = (
    [f"q_e[{i}]" for i in range(4)]
    + [f"v_e_b[{i}]" for i in range(3)]
    + [f"omega_b[{i}]" for i in range(3)]
    + [f"z_v[{i}]" for i in range(3)]
    + [f"z_q[{i}]" for i in range(3)]
)
_PERCENTILES = [0, 0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 100]


def main() -> None:
    print("[DEBUG] building env_cfg", flush=True)
    env_cfg = apply_training_profile(BROVVelEnvCfg(), args.profile)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    print("[DEBUG] constructing BROVVelEnv", flush=True)
    env = BROVVelEnv(cfg=env_cfg, render_mode=None)
    print("[DEBUG] BROVVelEnv constructed", flush=True)
    # attach_guidance()를 호출하지 않음 -> _current_v_d_b()가 _deploy_scheduler(학습
    # 커맨드 분포)를 그대로 사용 -> 이게 학습 중 정책이 실제로 본 관측 분포.

    agent_cfg = BROVVelPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    print("[DEBUG] wrapping env", flush=True)
    wrapped = RslRlVecEnvWrapper(env)
    print("[DEBUG] constructing runner", flush=True)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("[DEBUG] loading checkpoint", flush=True)
    runner.load(args.checkpoint)
    print("[DEBUG] getting inference policy", flush=True)
    policy = runner.get_inference_policy(device=env.device)
    print(f"[INFO] 체크포인트 로드 완료: {args.checkpoint}", flush=True)

    print("[DEBUG] resetting env", flush=True)
    obs_dict, _ = env.reset()
    print("[DEBUG] reset done, starting rollout", flush=True)
    collected: list[torch.Tensor] = []
    reset_count = 0
    with torch.inference_mode():
        for i in range(args.num_steps):
            actions = policy(obs_dict)
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            collected.append(obs_dict["policy"].clone().cpu())
            reset_count += int((terminated | truncated).sum().item())
            if i % 20 == 0:
                print(f"[DEBUG] step {i}/{args.num_steps}", flush=True)

    print("[DEBUG] rollout done, concatenating", flush=True)
    obs_all = torch.cat(collected, dim=0).numpy()
    env.close()

    stats = {}
    for idx, label in enumerate(_LABELS):
        col = obs_all[:, idx]
        stats[label] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "percentiles": {str(p): float(np.percentile(col, p)) for p in _PERCENTILES},
        }

    out = {
        "schema": "brov_training_obs_distribution_v1",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "profile": args.profile,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "reset_count": reset_count,
        "num_samples": int(obs_all.shape[0]),
        "labels": _LABELS,
        "stats": stats,
    }
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "logs", "training_obs_distribution.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(out, stream, indent=2, sort_keys=True)
    print(f"[INFO] 저장 완료: {output_path} ({obs_all.shape[0]}개 샘플, reset {reset_count}회)")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise

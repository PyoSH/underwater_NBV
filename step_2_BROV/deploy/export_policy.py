"""
학습된 체크포인트 → TorchScript 정책 export (IsaacLab 컨테이너 안에서 실행)
=============================================================================
`deploy/policy_runner.py`(topside PC, IsaacLab 미설치)가 로드할 standalone
`policy.pt`를 만든다. `OnPolicyRunner.export_policy_to_jit()`가 만드는 모듈은
`forward(obs: Tensor(N,16)) -> Tensor(N,6)` 형태의 결정론적(평균, 샘플링 없음)
추론만 하는 순수 TorchScript라 topside에서 IsaacLab/rsl_rl 없이 `torch.jit.load()`
만으로 돌아간다 (rsl_rl 5.0.1 소스로 직접 확인함 — `MLPModel.as_jit()`이
obs_normalizer+mlp+deterministic_output만 감싼 `_TorchMLPModel`을 반환).

사용법 (컨테이너 안, train.py/test_policy.py와 동일 실행 방식)
--------------------------------------------------------------
python.sh deploy/export_policy.py --checkpoint logs/brov_vel/model_299.pt --headless
→ deploy/exported/policy.pt 생성
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BROVVelEnv 정책 체크포인트 → TorchScript export")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL model_*.pt 체크포인트 경로")
parser.add_argument("--out_dir", type=str, default=None, help="기본값: deploy/exported/")
parser.add_argument("--filename", type=str, default="policy.pt")
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
args.headless = True   # export는 렌더링 불필요

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from importlib.metadata import version as _pkg_version

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from envs.vel_env_cfg import BROVVelEnvCfg
from envs.vel_env import BROVVelEnv
from agents.rsl_rl_ppo_cfg import BROVVelPPORunnerCfg


def main() -> None:
    env_cfg = BROVVelEnvCfg()
    env_cfg.scene.num_envs = 1
    env = BROVVelEnv(cfg=env_cfg, render_mode=None)

    agent_cfg = BROVVelPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "exported")
    runner.export_policy_to_jit(out_dir, args.filename)
    print(f"[INFO] TorchScript 정책 export 완료: {os.path.join(out_dir, args.filename)}")
    print("[INFO] obs 순서(16-dim): q_e(4), v_e_b(3), omega_b(3), z_v(3), z_q(3) — envs/vel_env.py 참조")
    print("[INFO] action 순서(6-dim): surge,sway,heave,roll,pitch,yaw — envs/vel_env_cfg.py 참조")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

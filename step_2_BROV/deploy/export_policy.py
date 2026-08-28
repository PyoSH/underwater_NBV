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
python.sh deploy/export_policy.py --checkpoint logs/brov_vel/model_299.pt \
  --profile legacy_exact --headless
→ deploy/exported/policy.pt 생성
"""

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BROVVelEnv 정책 체크포인트 → TorchScript export")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL model_*.pt 체크포인트 경로")
parser.add_argument("--out_dir", type=str, default=None, help="기본값: deploy/exported/")
parser.add_argument("--filename", type=str, default="policy.pt")
parser.add_argument(
    "--profile",
    choices=["legacy_exact", "paper_ref_v1", "deploy_v2", "deploy_v3", "deploy_v4", "deploy_v5", "deploy_v6", "deploy_v6b"],
    required=True,
    help="checkpoint가 학습된 observation/action contract (혼용 방지용 필수)",
)
parser.add_argument(
    "--allow_unverified_checkpoint",
    action="store_true",
    help="manifest가 없는 legacy checkpoint export를 의도적으로 허용",
)
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
args.headless = True   # export는 렌더링 불필요

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from importlib.metadata import version as _pkg_version

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


def _load_and_verify_manifest(checkpoint: str, profile: str) -> dict:
    checkpoint_abs = os.path.abspath(checkpoint)
    manifest_path = os.path.join(os.path.dirname(checkpoint_abs), "artifact_manifest.json")
    if not os.path.isfile(manifest_path):
        if not args.allow_unverified_checkpoint:
            raise RuntimeError(
                f"checkpoint manifest missing: {manifest_path}; legacy export requires "
                "--allow_unverified_checkpoint"
            )
        return {
            "profile": profile,
            "checkpoint_sha256": _sha256(checkpoint_abs),
            "verified": False,
        }
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("profile") != profile:
        raise RuntimeError(
            f"checkpoint profile mismatch: manifest={manifest.get('profile')!r}, "
            f"requested={profile!r}"
        )
    checkpoint_sha = _sha256(checkpoint_abs)
    if (
        os.path.basename(manifest.get("checkpoint", "")) == os.path.basename(checkpoint_abs)
        and manifest.get("checkpoint_sha256")
        and manifest["checkpoint_sha256"] != checkpoint_sha
    ):
        raise RuntimeError("checkpoint SHA256 does not match artifact manifest")
    manifest["checkpoint_sha256"] = checkpoint_sha
    manifest["manifest_path"] = manifest_path
    manifest["manifest_sha256"] = _sha256(manifest_path)
    manifest["verified"] = True
    return manifest


def main() -> None:
    training_manifest = _load_and_verify_manifest(args.checkpoint, args.profile)
    env_cfg = apply_training_profile(BROVVelEnvCfg(), args.profile)
    env_cfg.seed = int(training_manifest.get("seed", 42))
    env_cfg.scene.num_envs = 1
    env = BROVVelEnv(cfg=env_cfg, render_mode=None)

    agent_cfg = BROVVelPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "exported")
    runner.export_policy_to_jit(out_dir, args.filename)
    policy_path = os.path.join(out_dir, args.filename)
    metadata = {
        "schema": "brov_torchscript_policy_v2",
        "policy": os.path.abspath(policy_path),
        "policy_sha256": _sha256(policy_path),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": training_manifest["checkpoint_sha256"],
        "checkpoint_contract_verified": training_manifest["verified"],
        "training_manifest": training_manifest.get("manifest_path"),
        "training_manifest_sha256": training_manifest.get("manifest_sha256"),
        "profile": args.profile,
        "observation_contract": env_cfg.observation_contract,
        "action_contract": env_cfg.action_contract,
        "command_profile": env_cfg.command_profile,
        "reward_profile": env_cfg.reward_profile,
        # 보상 가중치는 **학습 manifest에서** 읽는다. env_cfg는 profile 기본값으로
        # 새로 만든 객체라 --rew_w_action 같은 CLI 덮어쓰기를 모른다. 여기서
        # env_cfg.rew_w_action을 쓰면 w_a=0.017로 학습된 checkpoint가 0.3으로
        # 기록되어 artifact 출처가 거짓이 된다.
        "rew_w_action": training_manifest.get("rew_w_action", env_cfg.rew_w_action),
        "seed": env_cfg.seed,
        "input_dim": env_cfg.observation_space,
        "output_dim": env_cfg.action_space,
    }
    # MK2 contract enrichment: brov_ros2/brov_control/brov_control/
    # policy_contract.py::resolve_policy_artifact_contract() requires these
    # fields (action_order, wrench_scale, policy_frame, allocation_frame,
    # vehicle_model_sha256) with strict equality before policy_node_mk2 will
    # load the artifact. Previously this enrichment happened by hand outside
    # this repo -- doing it here makes every MK2 export self-contained.
    if env_cfg.action_contract == "explicit_flu_zup_to_sname_frd_v1":
        # Must never diverge from brov_ros2's WRENCH_SCALE constant -- f_max
        # IS that constant by construction (see vel_env_cfg.py module
        # docstring on why action-envelope rescaling was rejected instead).
        _mk2_wrench_scale = (85.0, 85.0, 120.0, 26.0, 14.0, 22.0)
        if tuple(env_cfg.f_max) != _mk2_wrench_scale:
            raise RuntimeError(
                f"f_max {tuple(env_cfg.f_max)} != expected MK2 WRENCH_SCALE "
                f"{_mk2_wrench_scale} -- fix before export, this must match "
                "brov_ros2/brov_control/brov_control/policy_contract.py"
            )
        vehicle_model_path = os.path.join(
            os.path.dirname(__file__), "vendor", "brov2_heavy.yaml"
        )
        metadata.update(
            {
                "action_order": ["surge", "sway", "heave", "roll", "pitch", "yaw"],
                "policy_frame": "body_flu_zup",
                "allocation_frame": "body_frd_sname",
                "wrench_scale": list(_mk2_wrench_scale),
                "runtime_action_clip": [-1.0, 1.0],
                "required_executable": "policy_node_mk2",
                "vehicle_model_sha256": _sha256(vehicle_model_path),
            }
        )
    metadata_path = policy_path + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    print(f"[INFO] TorchScript 정책 export 완료: {policy_path}")
    print(f"[INFO] 계약 metadata: {metadata_path}")
    print("[INFO] obs 순서(16-dim): q_e(4), v_e_b(3), omega_b(3), z_v(3), z_q(3) — envs/vel_env.py 참조")
    print("[INFO] action 순서(6-dim): surge,sway,heave,roll,pitch,yaw — envs/vel_env_cfg.py 참조")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

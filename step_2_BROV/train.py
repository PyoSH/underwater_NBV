"""
BROV2 속도 컨트롤러 RSL-RL 학습 런처
======================================
`BROVVelEnv`(Sim2Swim 저수준 6DOF 속도/자세 컨트롤러, arXiv:2512.08656)를
RSL-RL PPO로 학습한다. `validate_physics.py`와 동일한 AppLauncher 패턴을
따르고, RSL-RL API(OnPolicyRunner/RslRlVecEnvWrapper)는 `Project_BROV/
custom_workflows/play.py`(레거시, 실제 동작 확인된 코드)의 사용례를 참고했다
— 단 이 프로젝트는 gym 레지스트리(`gym.make`)를 쓰지 않고 `env.py`/
`validate_physics.py`처럼 환경 클래스를 직접 인스턴스화한다.

isaac-lab-base 컨테이너 실측 결과, 설치된 rsl-rl-lib(5.0.1)이 legacy 코드
작성 시점보다 최신이라 `RslRlPpoActorCriticCfg`(구 스키마) 대신 `actor`/
`critic`을 `RslRlMLPModelCfg`로 직접 구성 + `handle_deprecated_rsl_rl_cfg()`
마이그레이션 호출이 필요했고(`agents/rsl_rl_ppo_cfg.py` 참조), `isaaclab_rl`의
`export_policy_as_jit/onnx`는 새 rsl-rl-lib 정책 객체 구조와 아직 안 맞아서
(업스트림 버전 불일치) `state_dict` 직접 저장으로 대체했다.

경로 추종(los_guidance.py)은 이 학습 루프에 관여하지 않는다 — 여기서는 속도/
자세 추종만 학습하고, 경로 추종은 정책 고정 후 별도 평가/배포 스크립트에서
LOSGuidance로 v_d^b/q_d를 생성해 이 정책에 먹인다.

사용법
------
python train.py --num_envs 512 --max_iterations 300 [--headless]
python train.py --resume [--headless]   # 가장 최근 체크포인트에서 재개
python train.py --logger wandb --log_project_name brov_vel [--headless]
                                         # wandb 로깅 (WANDB_API_KEY/WANDB_USERNAME 환경변수 필요,
                                         # 계정은 `wandb login`으로 사전 인증)
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BROV2 속도 컨트롤러 RSL-RL 학습")
parser.add_argument("--num_envs", type=int, default=None,
                     help="기본값: velEnvCfg.py의 scene.num_envs(512)")
parser.add_argument("--max_iterations", type=int, default=None,
                     help="기본값: BROVVelPPORunnerCfg.max_iterations")
parser.add_argument("--num_steps_per_env", type=int, default=None,
                     help="PPO rollout horizon override (policy steps per environment)")
parser.add_argument("--save_interval", type=int, default=None,
                     help="checkpoint 저장 iteration 간격 override")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--rew_w_action", type=float, default=None,
    help="논문 Table 1의 w_a (Eq.8 행동항 가중, 기본 0.3)를 덮어쓴다. "
         "이 값은 정상상태 추종률을 직접 결정한다 — 행동항은 정상상태에서도 "
         "비용이 0이 되지 않는 반면 속도항은 오차 0 근방에서 기울기가 0이라, "
         "둘의 균형점이 정책이 수렴할 속도가 된다. "
         "manifest에 기록되므로 checkpoint에서 역추적할 수 있다.")
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument(
    "--profile",
    choices=["legacy_exact", "paper_ref_v1", "deploy_v2", "deploy_v3", "deploy_v4", "deploy_v5", "deploy_v5_pitch_fmax_diag", "deploy_v6", "deploy_v6b"],
    default="deploy_v2",
    help="MDP/observation/action contract; new training defaults to deploy_v2",
)
parser.add_argument("--resume", action="store_true", help="가장 최근 체크포인트에서 재개")
parser.add_argument("--log_root", type=str,
                     default=os.path.join(os.path.dirname(__file__), "logs"))
parser.add_argument("--logger", type=str, default=None, choices=["tensorboard", "neptune", "wandb"],
                     help="기본값: BROVVelPPORunnerCfg.logger(tensorboard)")
parser.add_argument("--log_project_name", type=str, default=None,
                     help="wandb/neptune 프로젝트 이름. wandb 계정(entity)은 WANDB_USERNAME 환경변수로 지정")
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 기동 이후에만 import 가능 (isaaclab/rsl_rl이 Kit 앱 기동을 전제로 함) ──
sys.path.insert(0, os.path.dirname(__file__))
from importlib.metadata import version as _pkg_version

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from envs.vel_env_cfg import BROVVelEnvCfg, apply_training_profile
from envs.vel_env import BROVVelEnv
from agents.rsl_rl_ppo_cfg import BROVVelPPORunnerCfg
from robots.dynamics.brov2.thruster import BROV2ThrusterModel


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*git_args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *git_args], cwd=os.path.dirname(__file__), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_manifest(
    path: str,
    env_cfg: BROVVelEnvCfg,
    agent_cfg: BROVVelPPORunnerCfg,
    *,
    checkpoint: str | None = None,
    exported: str | None = None,
) -> None:
    root = os.path.dirname(__file__)
    source_files = {
        "envs/vel_env.py": os.path.join(root, "envs/vel_env.py"),
        "envs/vel_env_cfg.py": os.path.join(root, "envs/vel_env_cfg.py"),
        "envs/observation_contract.py": os.path.join(root, "envs/observation_contract.py"),
        "envs/desired_states.py": os.path.join(root, "envs/desired_states.py"),
        "action_frame_contract.py": os.path.join(root, "action_frame_contract.py"),
        "robots/dynamics/brov2/thruster.py": os.path.join(
            root, "../robots/dynamics/brov2/thruster.py"
        ),
        # The thrust curve now lives in measured data, not in source constants,
        # so the .npz has to be part of the provenance chain -- regenerating it
        # with a different noise floor would otherwise change the plant without
        # changing any hash the manifest records.
        "robots/dynamics/brov2/t200_table.py": os.path.join(
            root, "../robots/dynamics/brov2/t200_table.py"
        ),
        "robots/dynamics/brov2/t200_table.npz": os.path.join(
            root, "../robots/dynamics/brov2/t200_table.npz"
        ),
        "robots/dynamics/brov2/thruster_dynamics.py": os.path.join(
            root, "../robots/dynamics/brov2/thruster_dynamics.py"
        ),
        "robots/dynamics/brov2/mass_randomization.py": os.path.join(
            root, "../robots/dynamics/brov2/mass_randomization.py"
        ),
        "agents/rsl_rl_ppo_cfg.py": os.path.join(root, "agents/rsl_rl_ppo_cfg.py"),
        "train.py": os.path.join(root, "train.py"),
    }
    manifest = {
        "schema": "brov_stage3_training_artifact_v1",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "profile": env_cfg.training_profile,
        "observation_contract": env_cfg.observation_contract,
        "action_contract": env_cfg.action_contract,
        "command_profile": env_cfg.command_profile,
        "reward_profile": env_cfg.reward_profile,
        "rew_w_action": env_cfg.rew_w_action,
        "policy_dt_s": env_cfg.sim.dt * env_cfg.decimation,
        "episode_length_s": env_cfg.episode_length_s,
        "num_envs": env_cfg.scene.num_envs,
        "seed": agent_cfg.seed,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
        "mass_scale_range": list(env_cfg.dr_mass_scale_range),
        # Actuator plant. The thrust curve is voltage-dependent measured data
        # and the dynamics model is selectable, so both change what the policy
        # is trained against and neither is recoverable from the profile name.
        "thruster_nominal_voltage_v": BROV2ThrusterModel.NOMINAL_VOLTAGE,
        "thruster_dynamics_model": "third_order",
        "source_sha256": {
            label: _sha256(source_path)
            for label, source_path in source_files.items()
            if os.path.isfile(source_path)
        },
    }
    if checkpoint and os.path.isfile(checkpoint):
        manifest["checkpoint"] = os.path.abspath(checkpoint)
        manifest["checkpoint_sha256"] = _sha256(checkpoint)
    if exported and os.path.isfile(exported):
        manifest["exported_state_dict"] = os.path.abspath(exported)
        manifest["exported_state_dict_sha256"] = _sha256(exported)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)


def _validate_resume_contract(
    manifest_path: str,
    env_cfg: BROVVelEnvCfg,
    agent_cfg: BROVVelPPORunnerCfg,
) -> None:
    """Reject cross-profile/seed resumes before overwriting provenance."""

    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"--resume requires an existing manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as stream:
        previous = json.load(stream)
    expected = {
        "profile": env_cfg.training_profile,
        "observation_contract": env_cfg.observation_contract,
        "action_contract": env_cfg.action_contract,
        "command_profile": env_cfg.command_profile,
        "reward_profile": env_cfg.reward_profile,
        "rew_w_action": env_cfg.rew_w_action,
        "policy_dt_s": env_cfg.sim.dt * env_cfg.decimation,
        "seed": agent_cfg.seed,
    }
    mismatches = {
        key: {"previous": previous.get(key), "requested": value}
        for key, value in expected.items()
        if previous.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"resume contract mismatch: {mismatches}")


def main() -> None:
    env_cfg = apply_training_profile(BROVVelEnvCfg(), args.profile)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    # RSL-RL and the Isaac environment must share the effective seed.  Keeping
    # a concrete default also makes the value written to the manifest true.
    env_cfg.seed = args.seed
    if args.rew_w_action is not None:
        env_cfg.rew_w_action = args.rew_w_action

    agent_cfg = BROVVelPPORunnerCfg()
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    agent_cfg.seed = args.seed
    if args.logger is not None:
        agent_cfg.logger = args.logger
    # 프로젝트 이름은 wandb/neptune일 때만 의미 있음 (cli_args.py의 update_rsl_rl_cfg와 동일 조건)
    if agent_cfg.logger in {"wandb", "neptune"} and args.log_project_name:
        agent_cfg.wandb_project = args.log_project_name
        agent_cfg.neptune_project = args.log_project_name

    # rsl-rl-lib>=5.0.0: RslRlMLPModelCfg의 폐기 필드(stochastic 등)가 기본값(MISSING)이어도
    # to_dict()에 그대로 직렬화돼 MLPModel.__init__()이 거부한다 — 실제 설치 버전을 넘겨
    # 마이그레이션 함수로 정리해야 함 (isaac-lab-base 컨테이너에서 KeyError/TypeError로 확인).
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))

    log_dir = os.path.join(args.log_root, agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    manifest_path = os.path.join(log_dir, "artifact_manifest.json")
    if args.resume:
        _validate_resume_contract(manifest_path, env_cfg, agent_cfg)
    _write_manifest(manifest_path, env_cfg, agent_cfg)

    # deploy_v6b only: the curriculum needs the total step budget to compute
    # its ramp, and must be set before BROVVelEnv is constructed.
    if env_cfg.enable_action_envelope_curriculum:
        env_cfg.action_envelope_curriculum_total_steps = (
            agent_cfg.max_iterations * agent_cfg.num_steps_per_env
        )

    env = BROVVelEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args.resume:
        # 문자열 sorted()는 안 됨 — "model_299.pt" < "model_50.pt" (사전식 비교라 '2'<'5')로
        # 더 이전 체크포인트를 "최신"으로 잘못 고르는 경우가 생긴다. 파일명의 정수만 뽑아 비교.
        ckpts = [
            f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")
        ]
        if ckpts:
            resume_path = os.path.join(
                log_dir, max(ckpts, key=lambda f: int(f[len("model_"):-len(".pt")]))
            )
            print(f"[INFO] 체크포인트에서 재개: {resume_path}")
            runner.load(resume_path)
        else:
            print(f"[INFO] --resume 지정했지만 {log_dir}에 체크포인트 없음 — 처음부터 시작")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ── 배포용 정책 내보내기 (Sim2Swim의 zero-shot sim2real 목표에 맞춰 jit+onnx 둘 다) ──
    # rsl-rl-lib 5.0.1: get_inference_policy()가 이제 순수 MLPModel을 직접 반환하는데,
    # isaaclab_rl.exporter의 export_policy_as_jit/onnx는 아직 구버전 ActorCritic 래퍼
    # (.actor/.student 속성 보유)를 전제로 해서 "Policy does not have an actor/student
    # module" 에러가 남 — isaaclab_rl이 rsl-rl-lib 5.x를 완전히 못 따라간 업스트림
    # 버전 불일치로 보임(isaac-lab-base 컨테이너에서 실제 확인). state_dict 직접
    # 저장으로 우회하고, 실패해도 학습 결과 자체는 잃지 않도록 예외를 흡수한다.
    policy = runner.get_inference_policy(device=agent_cfg.device)
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)
    try:
        exported_path = os.path.join(export_dir, "policy_state_dict.pt")
        torch.save(policy.state_dict(), exported_path)
        print(f"[INFO] 정책 state_dict 저장 완료: {export_dir}")
    except Exception as e:
        print(f"[WARN] 정책 내보내기 실패(학습 결과는 정상 저장됨, log_dir 체크포인트 참고): {e}")

    checkpoints = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.startswith("model_") and name.endswith(".pt")
    ]
    latest_checkpoint = (
        max(checkpoints, key=lambda p: int(os.path.basename(p)[len("model_"):-len(".pt")]))
        if checkpoints else None
    )
    exported_path = os.path.join(export_dir, "policy_state_dict.pt")
    _write_manifest(
        manifest_path,
        env_cfg,
        agent_cfg,
        checkpoint=latest_checkpoint,
        exported=exported_path,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

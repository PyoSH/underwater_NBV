"""
BROVVelEnv RSL-RL PPO 러너 설정
=================================
`Project_BROV/agents/rsl_rl_ppo_cfg.py`의 죽은 `BROV2PPORunnerCfg`(사용처 없음,
`brov_env.py`도 실행 불가 레거시)를 참고해 `step_2_BROV`용으로 새로 작성.

네트워크 크기(`[64,64]`, elu)는 Sim2Swim(arXiv:2512.08656)의 "2-layer MLP"
서술에 맞춘 것 — 정확한 hidden dim은 논문에 없어서 legacy 설정값을 그대로 채용.

`num_steps_per_env`/`max_iterations`은 `BROVVelEnvCfg.episode_length_s=5.0`
(=policy 125 step) 기준 초기값이며 실측 후 조정이 필요하다 — 논문의 "2048 env,
80초 수렴"은 특정 GPU(A2000) 기준이라 우리 환경/하드웨어에 그대로 적용 안 됨.

**rsl-rl 버전 주의**: 이 컨테이너의 설치 버전은 rsl-rl-lib==5.0.1로, `policy=
RslRlPpoActorCriticCfg(...)` (구 스키마, Project_BROV legacy 코드가 쓰던 방식)가
deprecated이고 `OnPolicyRunner`가 자동 마이그레이션도 안 해줘서 그대로 쓰면
`KeyError: 'class_name'`으로 죽는다 (isaac-lab-base 컨테이너에서 실제 확인).
`actor`/`critic`을 `RslRlMLPModelCfg`로 직접 구성하는 신 스키마를 쓴다 —
`/workspace/isaaclab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` 원본 대조.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BROVVelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 300
    save_interval = 50
    experiment_name = "brov_vel"
    empirical_normalization = False
    # 환경 관측이 "policy" 그룹 하나뿐이라 actor/critic 둘 다 거기서 읽는다.
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,   # critic은 결정론적 value 추정 — 분포 불필요
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

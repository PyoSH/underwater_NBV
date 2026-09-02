"""
BROV2 6DOF 속도/자세 컨트롤러 환경 설정
==========================================
DirectRLEnvCfg 서브클래스. Sim2Swim(arXiv:2512.08656) 방식 저수준 RL 환경 —
경로 추종이 아니라 body-frame 속도(v_d^b) + 자세(q_d) 명령 추종만 학습한다.
경로 추종은 이 환경 밖(3D LOS guidance, guidance/los_guidance.py)에서 고전 제어로 처리한다.

관측 벡터 (16-dim)
------------------
q_e(4)      : 쿼터니언 오차 q̄_d ⊗ q [w,x,y,z]
v_e_b(3)    : body-frame 선속도 오차 v^b - v_d^b [m/s]
ω_b(3)      : body-frame 각속도 [rad/s]
z_v(3)      : v_e_b 적분 상태
z_q(3)      : q_e vector part 적분 상태

행동 벡터 (6-dim)
-----------------
surge,sway,heave,roll,pitch,yaw 스케일 [-1, 1] — F_max로 스케일 후
할당행렬 B의 pseudo-inverse로 8-thruster PWM에 할당한다 (robots/dynamics/brov2/thruster.py 참조).

deploy_v6 액션 envelope 클램프 (2026-08-18)
--------------------------------------------
실기 1차 시험에서 학습은 [-1,1] 전권한으로 이뤄지지만 배포측
(rl_controller_mk2_real_v1.yaml)은 같은 raw action을 축별로 15~30%만
(action_abs_limit=[0.3,0.3,0.3,0.2,0.15,0.2]) 허용한다는 구조적 불일치가
확인됐다. deploy_v6는 이 클램프를 학습 중 물리적으로 재현한다(action을
F_max 곱하기 전에 envelope로 clamp) — F_max/WRENCH_SCALE 자체는 절대
건드리지 않는다. WRENCH_SCALE(brov_ros2 policy_contract.py)은 전역
상수이자 모든 아티팩트 metadata에 strict-equality로 체크되므로, action
스케일을 재정의해 [-1,1]을 envelope에 매핑하는 방식(F_max를 프로필별로
바꾸는 것과 동치)은 배포측 계약을 깨뜨린다 — 그래서 채택하지 않았다.
"""

import os
import sys

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.scene_cfg import BROVSceneCfg


@configclass
class BROVVelEnvCfg(DirectRLEnvCfg):
    """BROV2 6DOF 속도/자세 컨트롤러 강화학습 환경 설정."""

    # ── 시뮬레이션 ──────────────────────────────────────────────────────────────
    # Physics remains 100 Hz; render at the 25 Hz policy rate when rendering is
    # explicitly enabled.  Headless training does not need four render attempts
    # per control step.
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=4)

    # ── 씬 ─────────────────────────────────────────────────────────────────────
    scene: BROVSceneCfg = BROVSceneCfg(num_envs=512, env_spacing=5.0)

    # Training must not build or update debug markers.  In the previous run this
    # path converted every environment's thruster arrows to Python lists on each
    # control step even with ``--headless``, dominating rollout time.  Evaluation
    # scripts may opt in explicitly when a rendered diagnostic is required.
    debug_vis: bool = False

    # Contract/profile labels are persisted with each Stage-3 artifact.  The
    # default is the deploy candidate requested by this project; old checkpoints
    # must be evaluated with ``apply_training_profile(..., "legacy_exact")``.
    training_profile: str = "deploy_v2"
    observation_contract: str = "brov_velocity_observation_v2"
    action_contract: str = "explicit_flu_zup_to_sname_frd_v1"
    command_profile: str = "deploy_v2"
    reward_profile: str = "deploy_v2"

    # ── 에피소드 ────────────────────────────────────────────────────────────────
    episode_length_s: float = 5.0      # Sim2Swim: 짧은 에피소드 + 대규모 병렬화
    decimation      : int   = 4        # 정책 dt = 4 × (1/100) = 0.04 s

    # ── RL 공간 ─────────────────────────────────────────────────────────────────
    observation_space: int = 16   # q_e(4) + v_e_b(3) + ω_b(3) + z_v(3) + z_q(3)
    action_space     : int = 6    # surge,sway,heave,roll,pitch,yaw

    # ── 초기 수심 ───────────────────────────────────────────────────────────────
    starting_depth: float = 10.0   # 경로가 없어 경계에서 충분히 떨어진 곳에서 시작

    # ── 액션 할당 (robots.dynamics.brov2.thruster.build_allocation_matrix + inverse_thrust) ──
    # F_max — von Benzon et al. 2022 (JMSE 10,1898) Table 4 실측 최대추력값.
    # (surge, sway, heave, roll, pitch, yaw) = [N, N, N, N·m, N·m, N·m]
    f_max: tuple = (85.0, 85.0, 120.0, 26.0, 14.0, 22.0)

    # ── desired-state profiles ──────────────────────────────────────────────────
    # Paper: Eq.9 defines a Frenet-Serret attitude trajectory; body velocity is
    # sampled independently on S^2 at exactly 0.5 m/s.  deploy_v2 adds one
    # mission-shaped command transition in every 5 s episode.
    cmd_omega : float = 0.2
    cmd_coeffs: tuple = (0.5, 0.5, 0.3)   # [a, b, c]
    paper_command_speed: float = 0.5
    command_transition_time_range_s: tuple = (2.0, 3.0)
    deploy_speed_bins: tuple = (0.0, 0.1, 0.5)

    # deploy_v3: mission-scale multi-leg curriculum (envs/desired_states.py
    # DeployV3Scheduler).  episode_length_s is overridden to
    # deploy_v3_episode_length_s by apply_training_profile() below — this
    # keeps deploy_v2's 5 s default untouched for existing checkpoints.
    deploy_v3_episode_length_s        : float = 30.0
    deploy_v3_leg_duration_range_s    : tuple = (3.0, 8.0)
    deploy_v3_new_attitude_probability: float = 0.5
    # Sized for far more than the 30 s training episode: test_policy.py
    # inflates episode_length_s to duration+5 for every profile (so a
    # timeout reset never interrupts a guided eval run), which can reach
    # ~65 s+ even though the scheduler itself is never sampled once
    # attach_guidance() is active.  48 legs covers up to (48-2)*3=138 s at
    # the minimum 3 s leg duration -- comfortable headroom, and the per-leg
    # state is a few floats per env so the extra table size is free.
    deploy_v3_max_legs                : int   = 48

    # Reset z_v/z_q on every command-leg transition instead of only at
    # episode reset.  False preserves deploy_v2's exact existing behavior
    # (carry the integral through its one mid-episode transition); deploy_v3
    # turns this on to mirror the deployment-side waypoint-transition reset
    # (bumpless transfer) -- see project_step2_brov_retrain_spec memory for
    # why this alone is not sufficient without the episode-length change.
    reset_integral_on_command_transition: bool = False

    # deploy_v4 (spec items B+C on top of v3): DVL sensor realism domain
    # randomization.  Only v_e_b/z_v are affected (q_e/omega_b stay on the
    # IMU path) -- see envs/dvl_realism.py.  Defaults mirror the measured
    # Water Linked A50 characteristics already used in the Gazebo Stage-2
    # harness (stage2_sitl_dvl_injector.py).
    enable_dvl_realism        : bool  = False
    dvl_rate_hz_range         : tuple = (5.0, 15.0)
    dvl_noise_std_range_mps   : tuple = (0.0, 0.006)
    dvl_delay_s_range         : tuple = (0.0, 0.15)

    # deploy_v4: thruster voltage-sag (shared across the vehicle's 8
    # thrusters -- one battery) and per-thruster manufacturing/wear variance
    # (independent per thruster) domain randomization -- see
    # robots/dynamics/brov2/thruster.py BROV2ThrusterModel.randomize().
    dr_enable_thrust_scale          : bool  = False
    dr_thrust_voltage_scale_range   : tuple = (0.85, 1.0)
    dr_thrust_individual_scale_range: tuple = (0.90, 1.10)

    # Runtime-compatible integral state transition.  Isaac always integrates a
    # fresh sample at fixed 0.04 s; ROS may freeze stale samples using the same
    # pure-Torch contract.
    integral_velocity_limit: float = 5.0
    integral_attitude_limit: float = 5.0

    # ── 보상 가중치 (Sim2Swim Table 1, Eq.5-8) ─────────────────────────────────
    # r = w_quat·exp(-‖q_e_vec‖²) + w_vel·exp(-‖v_e_b‖²) + w_omega·exp(-‖ω_b‖²)
    #   + w_quat·exp(-∠(q_d,q))                                    (Eq.7, 별도 항, 제곱 없음)
    #   + w_action·exp(-‖a‖)
    rew_w_quat  : float = 0.4    # orientation error — Eq.6의 q_e 항과 Eq.7의 r_q 항 둘 다에 재사용
    rew_w_vel   : float = 0.2
    rew_w_omega : float = 0.05
    rew_w_action: float = 0.3

    # deploy_v2 keeps a coarse paper velocity term for exploration and adds a
    # precision term at the deployment tolerance.  Small negative penalties
    # target the observed tick-to-tick saturation without dominating tracking.
    deploy_rew_w_quat: float = 0.2
    deploy_rew_w_vel_coarse: float = 0.2
    deploy_rew_w_vel_precision: float = 0.6
    deploy_rew_vel_sigma: float = 0.10
    deploy_rew_w_omega: float = 0.05
    deploy_penalty_action_l2: float = 0.01
    deploy_penalty_action_delta_l2: float = 0.02
    deploy_penalty_thruster_clamp_l2: float = 0.05
    # Penalizes only the pre-clamp actor output beyond [-1,1] -- the other
    # three penalties above are computed from the already-clamped action and
    # so cannot see how far past the bound the raw output actually was.
    # deploy_v5 measured (via TorchScript replay of deploy_v4's own Gazebo
    # bags) sum-of-squares overflow per step averaging ~0.3-0.36 (p95
    # ~1.8-2.1), concentrated almost entirely in pitch/yaw. Default 0.0 keeps
    # deploy_v2/v3/v4 behavior unchanged.
    deploy_penalty_raw_overflow_l2: float = 0.0

    # deploy_v6: physically reproduce the real-vehicle action envelope
    # (rl_controller_mk2_real_v1.yaml's action_abs_limit) during training.
    # Applied in _pre_physics_step BEFORE the [-1,1]-clamped action is used
    # anywhere else -- this is a training-side clamp only; f_max/WRENCH_SCALE
    # are never touched (see module docstring above for why option-2
    # rescaling was rejected).
    #
    # pitch=0.25 (not the deployed-today 0.15) per the 2026-08-18 3rd-round
    # real-bag diagnosis: pitch's deployed torque budget (0.15*14.0=2.10
    # N*m) was BELOW the max trim moment dr_cob_radius=0.015 can produce
    # (2.16 N*m) -- i.e. the deployed envelope could not always counter even
    # the *training-distribution* CoB disturbance, independent of any real
    # hardware effect (ballast on vs off reproduced the same failure). This
    # value is a training target, not yet a fact about the deployed vehicle
    # -- brov_ros2/brov_control/config/rl_controller_mk2_real_v1.yaml's own
    # pitch action_abs_limit must be raised to 0.25 to match before this
    # checkpoint's real-vehicle behavior means what training assumed.
    deploy_v6_action_abs_limit: tuple = (0.3, 0.3, 0.3, 0.2, 0.25, 0.2)
    enable_action_envelope_clamp: bool = False
    # Item-D-style overflow penalty (deploy_v5 precedent), but keyed to the
    # deployment envelope instead of the actor's own [-1,1] contract bound --
    # gives the actor gradient signal to stop wasting output beyond what has
    # any physical effect once enable_action_envelope_clamp is active.
    deploy_penalty_envelope_overflow_l2: float = 0.0

    # deploy_v6b fallback only (see apply_training_profile): ramp the
    # envelope from [1,1,1,1,1,1] down to deploy_v6_action_abs_limit over
    # the first (1 - hold_fraction) of training, then hold at the target for
    # the remainder so the exported checkpoint is trained under the exact
    # target envelope, not a mid-ramp one. Off by default; only used if a
    # direct deploy_v6 run fails Gazebo validation.
    enable_action_envelope_curriculum: bool = False
    action_envelope_curriculum_total_steps: int = 0
    action_envelope_curriculum_hold_fraction: float = 0.5

    # deploy_v6 (2026-08-18 3rd-round diagnosis): per-axis integrator
    # anti-windup. Real-bag analysis of the deploy_v5 failure found only
    # qint_y (pitch attitude-error integral, z_q[1]) pinned at the -5.0
    # limit (15.5% of samples) while vint_z (heave velocity integral)
    # stayed well inside bound (-1.0) -- the depth oscillation (6.14s
    # period, +/-0.224m) was a downstream symptom of pitch integral
    # windup, not an independent cause. Conditional integration (halt a
    # z_v/z_q axis's integration on ticks where that axis's own action is
    # clamped away by the envelope) was chosen over the other two options
    # the diagnosis offered: back-calculation anti-windup would need new
    # per-axis gain state with no deployment-side precedent to mirror, and
    # narrowing integral_velocity_limit/integral_attitude_limit changes the
    # observation contract itself (both z_v/z_q's declared range), which
    # cannot be changed unilaterally on the training side -- conditional
    # integration keeps the +/-5.0 bound and the 16-D contract exactly as
    # documented, only changing *when* each axis accumulates. This mirrors
    # the existing deploy_penalty_raw_overflow_l2/envelope_overflow pattern
    # of keying a training-side behavior off the same "raw action exceeds a
    # bound" signal. NOT applied to v2/v5 (enable_action_envelope_clamp
    # gates whether an envelope exists to overflow against at all).
    #
    # brov_ros2/brov_base/brov_base/observation.py's _z_q/_z_v update is an
    # INDEPENDENT reimplementation of this same integration rule (not an
    # import of this module) -- it must be updated with the identical
    # per-axis conditional-integration logic, or the exact train/deploy
    # integral-behavior mismatch this change fixes for pitch will reappear.
    enable_attitude_integral_antiwindup: bool = False
    enable_velocity_integral_antiwindup: bool = False

    # deploy_v6: probability that a DeployV3-style leg retarget additionally
    # couples the new attitude target to the new leg's velocity direction
    # (mirrors guidance/los_guidance.py's heading_mode="align", where q_d is
    # a deterministic function of v_d_world every tick and both jump
    # together at a waypoint transition -- DeployV3Scheduler samples them
    # independently, which is the gap this closes).
    deploy_v6_los_coupled_retarget_probability: float = 0.5

    # ── 행동 지연 / 관측 신선도 / 행동 이력 (DELAY_TRAINING_PLAN.md, 2026-09-02) ──
    # 실기 수조 세션이 배포 진동의 근본원인을 dead time τ=80 ms + 정책 포화
    # (relay) 로 확정했다. 학습 환경에는 이 지연이 구조적으로 없어서(τ=0,
    # phase margin +64° vs 실기 −24°) 정책이 실기 안정 문턱을 넘는 이득을
    # 배웠다. 아래 세 스위치가 그 격차를 학습 환경 안으로 들여온다 —
    # 구현은 envs/action_delay.py (순수 torch, 단위시험 있음).
    #
    # 주입값이 실측 80 ms 가 아니라 중심 60 ms 인 이유(이중 계상 보정): 실측
    # 80 ms 는 명령→자이로 전체로 실기 액추에이터 응답을 포함한다. 학습
    # 환경에는 이미 von Benzon 3차 추진기 동특성이 있고 2 Hz 에서 위상 −15°
    # ≈ 등가지연 21 ms 를 공급하므로, 전송 몫 60 ms(= 80 − 21) 만 주입한다.
    #
    # 기본값은 전부 off — 기존 프로파일(paper_ref_v1/deploy_*)의 학습 결과가
    # 1 bit 도 변하면 안 된다(회귀 확인 대상).
    enable_action_delay   : bool  = False
    action_delay_ms_range : tuple = (40.0, 80.0)   # 에피소드마다 uniform, 중심 60 ms
    # 매 정책 스텝 env 별 이 확률로 직전 스텝 관측을 대신 공급(그 틱은 z_v/z_q
    # 적분도 정지 — 배포측 stale/duplicate 표본 처리와 같은 규칙).
    # 근거: 실기 attitude_age 분포의 15.1% 가 40~50 ms = 1 틱 묵음.
    obs_stale_probability : float = 0.0
    # 관측에 덧붙일 최근 실행 행동 개수. 0 이면 16-D 계약 그대로.
    # 지연 상한 80 ms = 정책 2스텝이므로 2 개면 증강 등가 정리를 충족한다
    # (Katsikopoulos & Engelbrecht 2003; Walsh et al. 2008).
    action_history_length : int   = 0

    # ── 안전 경계 (env origin 기준, 초과 시 terminated) ─────────────────────────
    # 경로 추종이 없는 태스크라 waypoint 기반 경계 대신 널찍한 안전판만 둔다.
    max_bound: float = 20.0

    # ── domain randomization ─────────────────────────────────────────────────────
    # The paper discloses uniform mass/volume randomization but not its bounds.
    # ±5% is the explicit project assumption matching the 600 g (~5%) ballast
    # trial.  Inertia is scaled by the same mass ratio.
    dr_mass_scale_range     : tuple = (0.95, 1.05)
    dr_enable_mass          : bool = True
    dr_volume_range         : tuple = (0.01320, 0.01613)   # [m^3] nominal 0.014665 ±10%
    dr_cob_radius           : float = 0.015                 # [m] 구 반경 (부피 기준 균등 샘플)
    dr_added_mass_rot_range : tuple = (0.60, 1.40)           # nominal 대비 배율 (±40%, Kṗ/Mq̇/Nṙ)


def apply_training_profile(cfg: BROVVelEnvCfg, profile: str) -> BROVVelEnvCfg:
    """Apply one named Stage-3 contract without silently mixing old policies."""

    if profile == "legacy_exact":
        cfg.training_profile = profile
        cfg.observation_contract = "legacy_exact_0p5"
        cfg.action_contract = "legacy_model_299_no_t6"
        cfg.command_profile = "legacy_eq9_velocity"
        cfg.reward_profile = "paper_eq5_8"
        cfg.dr_enable_mass = False
    elif profile == "paper_ref_v1":
        cfg.training_profile = profile
        cfg.observation_contract = "brov_velocity_observation_v2"
        cfg.action_contract = "explicit_flu_zup_to_sname_frd_v1"
        cfg.command_profile = "paper_ref_v1"
        cfg.reward_profile = "paper_eq5_8"
        cfg.dr_enable_mass = True
    elif profile == "paper_delay_v1":
        # 설계 A (DELAY_TRAINING_PLAN.md §2) — 계약 유지 ablation 대조군.
        # paper_ref_v1 과 관측/보상/명령이 전부 동일하고 행동 지연 + 관측
        # 신선도 jitter 만 켠다. 관측이 16-D 그대로이므로 정책이 할 수 있는
        # 적응은 이득 강하뿐이고, 그것이 이 run 의 관심사다.
        cfg = apply_training_profile(cfg, "paper_ref_v1")
        cfg.training_profile = profile
        cfg.enable_action_delay = True
        cfg.action_delay_ms_range = (40.0, 80.0)
        cfg.obs_stale_probability = 0.15
    elif profile == "paper_delay_hist_v1":
        # 설계 B (DELAY_TRAINING_PLAN.md §2) — MDP 유지 본안.
        # A 와 지연/신선도가 동일하고 관측만 28-D 로 늘린다:
        # [기존 16] + [a_{t-1}(6)] + [a_{t-2}(6)], a 는 실행 행동(탐색 노이즈
        # 포함, clip 이후, 지연버퍼에 들어간 바로 그 값).
        # observation_contract 를 새 이름으로 바꿔 기존 16-D artifact 와
        # 절대 혼용되지 않게 계약 검사로 막는다.
        cfg = apply_training_profile(cfg, "paper_delay_v1")
        cfg.training_profile = profile
        cfg.observation_contract = "brov_velocity_observation_v3_hist2"
        cfg.action_history_length = 2
        cfg.observation_space = 16 + cfg.action_history_length * cfg.action_space
    elif profile == "deploy_v2":
        cfg.training_profile = profile
        cfg.observation_contract = "brov_velocity_observation_v2"
        cfg.action_contract = "explicit_flu_zup_to_sname_frd_v1"
        cfg.command_profile = "deploy_v2"
        cfg.reward_profile = "deploy_v2"
        cfg.dr_enable_mass = True
    elif profile == "deploy_v3":
        cfg.training_profile = profile
        cfg.observation_contract = "brov_velocity_observation_v2"
        cfg.action_contract = "explicit_flu_zup_to_sname_frd_v1"
        cfg.command_profile = "deploy_v3"
        cfg.reward_profile = "deploy_v2"   # reuse deploy_v2's reward; see spec item D
        cfg.dr_enable_mass = True
        cfg.episode_length_s = cfg.deploy_v3_episode_length_s
        cfg.reset_integral_on_command_transition = True
    elif profile == "deploy_v4":
        cfg.training_profile = profile
        cfg.observation_contract = "brov_velocity_observation_v2"
        cfg.action_contract = "explicit_flu_zup_to_sname_frd_v1"
        cfg.command_profile = "deploy_v3"   # same multi-leg curriculum as v3
        cfg.reward_profile = "deploy_v2"
        cfg.dr_enable_mass = True
        cfg.episode_length_s = cfg.deploy_v3_episode_length_s
        cfg.reset_integral_on_command_transition = True
        cfg.enable_dvl_realism = True
        cfg.dr_enable_thrust_scale = True
    elif profile == "deploy_v5":
        cfg.training_profile = profile
        cfg.observation_contract = "brov_velocity_observation_v2"
        cfg.action_contract = "explicit_flu_zup_to_sname_frd_v1"
        cfg.command_profile = "deploy_v3"   # same multi-leg curriculum as v3/v4
        cfg.reward_profile = "deploy_v2"
        cfg.dr_enable_mass = True
        cfg.episode_length_s = cfg.deploy_v3_episode_length_s
        cfg.reset_integral_on_command_transition = True
        cfg.enable_dvl_realism = True
        cfg.dr_enable_thrust_scale = True
        # Item D: penalize raw (pre-clamp) actor overflow beyond [-1,1].
        # Calibrated from a TorchScript replay of deploy_v4's own Gazebo
        # bags -- see deploy_penalty_raw_overflow_l2 docstring above.
        cfg.deploy_penalty_raw_overflow_l2 = 0.15
    elif profile == "deploy_v6":
        # Inherits A+B+C+D from deploy_v5 unchanged, then adds items 1+4
        # from the 2026-08-18 real-vehicle-failure retrain spec: physical
        # action-envelope clamp + LOS-coupled attitude retargets.
        cfg = apply_training_profile(cfg, "deploy_v5")
        cfg.training_profile = profile
        cfg.command_profile = "deploy_v6"
        cfg.enable_action_envelope_clamp = True
        # Starting point = deploy_v5's calibrated raw_overflow weight; retune
        # from the smoke run's envelope-overflow diagnostic if action std
        # collapses or the term dominates total reward magnitude.
        cfg.deploy_penalty_envelope_overflow_l2 = 0.15
        cfg.enable_attitude_integral_antiwindup = True
        cfg.enable_velocity_integral_antiwindup = True
    elif profile == "deploy_v6b":
        # Curriculum fallback -- only use if a direct deploy_v6 run fails
        # Gazebo validation. train.py must set
        # action_envelope_curriculum_total_steps before constructing the env.
        cfg = apply_training_profile(cfg, "deploy_v6")
        cfg.training_profile = profile
        cfg.enable_action_envelope_curriculum = True
    elif profile == "deploy_v5_pitch_fmax_diag":
        # Diagnostic-only, never for real-vehicle export: doubles pitch
        # F_max (14 -> 28 N*m) on top of deploy_v5, isolating whether
        # deploy_v5's remaining pitch-axis saturation is a genuine
        # torque-budget ceiling. 28 N*m exceeds the von Benzon et al. 2022
        # measured real-hardware value -- this profile's checkpoint must
        # never be exported into an MK2 artifact bundle or accepted by
        # policy_contract.py's MK2_ACCEPTED_PROFILES.
        cfg = apply_training_profile(cfg, "deploy_v5")
        cfg.training_profile = profile
        cfg.f_max = (85.0, 85.0, 120.0, 26.0, 28.0, 22.0)
    else:
        raise ValueError(
            f"unknown training profile {profile!r}; expected legacy_exact, "
            "paper_ref_v1, paper_delay_v1, paper_delay_hist_v1, deploy_v2, "
            "deploy_v3, deploy_v4, deploy_v5, deploy_v5_pitch_fmax_diag, "
            "deploy_v6, or deploy_v6b"
        )
    return cfg

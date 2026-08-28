"""
BROV2 6DOF 속도/자세 컨트롤러 RL 환경
========================================
IsaacLab DirectRLEnv 기반. Sim2Swim(arXiv:2512.08656) 저수준 컨트롤러 재현.

계층 구조
---------
[Waypoints] → 3D LOS Guidance(고전 제어, guidance/los_guidance.py) → v_d^b, q_d
                                                            │
                                                    BROVVelEnv(RL, 이 파일)
                                                            │  6-dim wrench
                                                    B_pinv 할당 → 8-thruster PWM
                                                            │
                                    robots/dynamics/ (step_2/step_3 공유 물리, 변경 없음)

이 환경은 "경로 추종"을 학습하지 않는다 — body-frame 속도(v_d^b) + 자세(q_d)
명령을 얼마나 잘 추종하는지만 학습한다. 경로 추종은 정책 고정 후 LOS guidance
레이어가 담당 (envs/traj_env.py의 BROVTrajEnv는 end-to-end 대안으로 별도 유지).
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import (
    VisualizationMarkers,
    RED_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from envs.vel_env_cfg import BROVVelEnvCfg
from envs.observation_contract import (
    build_velocity_observation,
    canonicalize_quaternion,
)
from envs.desired_states import (
    DeployV2Config,
    DeployV2Scheduler,
    DeployV3Config,
    DeployV3Scheduler,
    DeployV6Config,
    DeployV6Scheduler,
    PaperReferenceBatch,
    PaperReferenceConfig,
)
from envs.dvl_realism import DVLRealismConfig, DVLRealismModel
from action_frame_contract import (
    T6_DIAGONAL,
    build_policy_action_to_sname_frd_multiplier,
)
from robots.dynamics.brov2.thruster import BROV2ThrusterModel, build_allocation_matrix
from robots.dynamics.brov2.mass_randomization import randomize_articulation_mass
from robots.dynamics.fossen import Hydrodynamics
from robots.dynamics.brov2.params import load_brov2_yaml, coBM_vector_ned, thruster_pos_dir_ned
from guidance.los_guidance import _heading_from_direction  # 방향벡터 → 자세 쿼터니언 (목표속도 화살표용)

_THRUST_ARROW_SCALE = 0.02   # [m/N] 화살표 길이 = 추력[N] * 이 값 (traj_env.py와 동일)
_THRUST_ARROW_COLOR = (1.0, 0.35, 0.0, 1.0)   # RGBA, 주황
_THRUST_ARROW_WIDTH = 3.0
_ARROWHEAD_FRACTION = 0.25
_ARROWHEAD_MAX_LEN  = 0.03

_VEL_ARROW_SPEED_REF = 0.5   # [m/s] 이 속력일 때 화살표 길이 = 마커 기본 길이(scale=1.0)


def _sample_from_sphere(n: int, r: float, device) -> torch.Tensor:
    """반경 r 구 내부 균등(부피 기준) 샘플. Project_BROV/brov_env.py 패턴 재사용.

    단순 `r*rand()`는 중심부에 밀집되는 오류가 있어 `r*rand()^(1/3)`로 스케일한다.
    """
    coords = torch.randn(n, 3, device=device)
    coords = coords / coords.norm(dim=1, keepdim=True)
    radii = r * torch.rand(n, 1, device=device).pow(1 / 3)
    return radii * coords


class BROVVelEnv(DirectRLEnv):
    """BROV2 Heavy 6DOF 속도/자세 컨트롤러 환경."""

    cfg: BROVVelEnvCfg

    def __init__(self, cfg: BROVVelEnvCfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode)

        self._robot: Articulation = self.scene.articulations["robot"]
        self._policy_dt = cfg.sim.dt * cfg.decimation

        yaml_params = load_brov2_yaml()
        hydro_coef  = yaml_params["hydro_coef"]
        cob_vector  = coBM_vector_ned(yaml_params)
        thr_pos, thr_dir = thruster_pos_dir_ned(yaml_params)

        self._thruster = BROV2ThrusterModel(
            self.num_envs, cfg.sim.dt, self.device, pos=thr_pos, dir=thr_dir,
        )
        self._hydro = Hydrodynamics(
            self.num_envs, cfg.sim.dt, self.device,
            volume            = yaml_params["volume"],
            cob_vector        = cob_vector,
            water_density     = yaml_params["environment"]["fluid_density"],
            added_mass        = hydro_coef["added_mass"],
            linear_damping    = hydro_coef["linear_damping"],
            quadratic_damping = hydro_coef["quadratic_damping"],
            # M_total = M_RB + M_A 를 만들기 위해 필요하다. Hydrodynamics가
            # added mass를 암묵적으로 풀기 때문 — compute() docstring 참조.
            rigid_mass        = yaml_params["expect"]["mass"],
            rigid_inertia     = yaml_params["expect"]["inertia"],
        )
        self._rigid_mass = float(yaml_params["expect"]["mass"])

        # 할당행렬 B(6,8) → pseudo-inverse(8,6). YAML 위치/방향에서 매번 계산
        # (하드코딩 금지 — coBM/hydro_coef와 동일한 단일 정본 원칙).
        B = build_allocation_matrix(self._thruster._pos, self._thruster._dir)
        self._B = B.to(self.device)                                      # (6,8)
        self._B_pinv = torch.linalg.pinv(self._B)                         # (8,6)
        self._f_max  = torch.tensor(cfg.f_max, device=self.device)       # (6,)
        # deploy_v6: physical action-envelope clamp (see _pre_physics_step).
        # f_max/WRENCH_SCALE are never touched -- only the action that
        # multiplies it is bounded, before that multiplication happens.
        self._action_envelope = torch.tensor(
            cfg.deploy_v6_action_abs_limit, device=self.device
        )
        self._action_to_sname_multiplier = (
            build_policy_action_to_sname_frd_multiplier(
                cfg.f_max,
                contract=cfg.action_contract,
                dtype=self._f_max.dtype,
                device=self.device,
            )
        )
        self._sname_to_zup_sign = torch.tensor(
            T6_DIAGONAL, dtype=self._f_max.dtype, device=self.device
        )
        lower_force, upper_force = self._thruster.force_limits_n
        self._thruster_force_scale = max(abs(lower_force), abs(upper_force))

        # 도메인 랜덤화 기준값 (nominal, hydro.randomize() 오프셋/배율 기준)
        self._nominal_added_mass_rot = torch.tensor(
            hydro_coef["added_mass"][3:], device=self.device
        )   # (3,) Kṗ, Mq̇, Nṙ

        # 속도/자세 명령 + 오차 적분 버퍼
        self._v_d_b    = torch.zeros(self.num_envs, 3, device=self.device)
        self._q_d      = torch.zeros(self.num_envs, 4, device=self.device)
        self._cmd_quat = torch.zeros(self.num_envs, 4, device=self.device)   # 속도궤적 방향 랜덤화용
        self._z_v      = torch.zeros(self.num_envs, 3, device=self.device)
        self._z_q      = torch.zeros(self.num_envs, 3, device=self.device)
        # Integrals advance once per unique policy sample.  This keeps reset
        # observations at z=0 and makes duplicate getter calls idempotent,
        # matching the source-sample gate used by brov_ros2.
        self._last_integrated_episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._actions  = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._raw_actions = torch.zeros_like(self._actions)
        # [-1,1]-clamped actor output, before any deploy_v6 envelope clamp --
        # matches real /brov/action_raw exactly (see _pre_physics_step).
        self._pre_envelope_actions = torch.zeros_like(self._actions)
        self._force_requested = torch.zeros(self.num_envs, 8, device=self.device)
        self._force_limited = torch.zeros_like(self._force_requested)
        self._wrench_requested_zup = torch.zeros(self.num_envs, 6, device=self.device)
        self._wrench_achieved_zup = torch.zeros_like(self._wrench_requested_zup)
        self._mass_scale = torch.ones(self.num_envs, 1, device=self.device)
        self._command_transition_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._command_transition_mode = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._command_reversal_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        command_seed = int(cfg.seed if cfg.seed is not None else 42)
        self._paper_reference = None
        self._deploy_scheduler = None
        if cfg.command_profile == "paper_ref_v1":
            self._paper_reference = PaperReferenceBatch(
                self.num_envs,
                device=self.device,
                seed=command_seed,
                config=PaperReferenceConfig(
                    speed_mps=cfg.paper_command_speed,
                    trajectory_coefficients=tuple(cfg.cmd_coeffs),
                    trajectory_omega_rad_s=cfg.cmd_omega,
                    episode_length_s=cfg.episode_length_s,
                ),
            )
        elif cfg.command_profile == "deploy_v2":
            self._deploy_scheduler = DeployV2Scheduler(
                self.num_envs,
                device=self.device,
                seed=command_seed,
                config=DeployV2Config(
                    episode_length_s=cfg.episode_length_s,
                    transition_time_range_s=tuple(
                        cfg.command_transition_time_range_s
                    ),
                    speed_bins_mps=tuple(cfg.deploy_speed_bins),
                    exact_reversal=True,
                    policy_dt_s=self._policy_dt,
                    trajectory_coefficients=tuple(cfg.cmd_coeffs),
                    trajectory_omega_rad_s=cfg.cmd_omega,
                ),
            )
        elif cfg.command_profile == "deploy_v3":
            self._deploy_scheduler = DeployV3Scheduler(
                self.num_envs,
                device=self.device,
                seed=command_seed,
                config=DeployV3Config(
                    episode_length_s=cfg.episode_length_s,
                    leg_duration_range_s=tuple(cfg.deploy_v3_leg_duration_range_s),
                    speed_bins_mps=tuple(cfg.deploy_speed_bins),
                    new_attitude_probability=cfg.deploy_v3_new_attitude_probability,
                    policy_dt_s=self._policy_dt,
                    trajectory_coefficients=tuple(cfg.cmd_coeffs),
                    trajectory_omega_rad_s=cfg.cmd_omega,
                    max_legs=cfg.deploy_v3_max_legs,
                ),
            )
        elif cfg.command_profile == "deploy_v6":
            self._deploy_scheduler = DeployV6Scheduler(
                self.num_envs,
                device=self.device,
                seed=command_seed,
                config=DeployV6Config(
                    episode_length_s=cfg.episode_length_s,
                    leg_duration_range_s=tuple(cfg.deploy_v3_leg_duration_range_s),
                    speed_bins_mps=tuple(cfg.deploy_speed_bins),
                    new_attitude_probability=cfg.deploy_v3_new_attitude_probability,
                    policy_dt_s=self._policy_dt,
                    trajectory_coefficients=tuple(cfg.cmd_coeffs),
                    trajectory_omega_rad_s=cfg.cmd_omega,
                    max_legs=cfg.deploy_v3_max_legs,
                    los_coupled_retarget_probability=(
                        cfg.deploy_v6_los_coupled_retarget_probability
                    ),
                ),
            )
        elif cfg.command_profile != "legacy_eq9_velocity":
            raise ValueError(f"unsupported command profile {cfg.command_profile!r}")

        # deploy_v4 (spec item B): DVL sensor realism. Only v_e_b/z_v pass
        # through this -- q_e/omega_b stay on the IMU path (see
        # envs/dvl_realism.py).
        self._dvl_realism = None
        if cfg.enable_dvl_realism:
            self._dvl_realism = DVLRealismModel(
                self.num_envs,
                device=self.device,
                seed=command_seed,
                config=DVLRealismConfig(
                    rate_hz_range=tuple(cfg.dvl_rate_hz_range),
                    noise_std_range_mps=tuple(cfg.dvl_noise_std_range_mps),
                    delay_s_range=tuple(cfg.dvl_delay_s_range),
                    policy_dt_s=self._policy_dt,
                ),
            )

        # Debug drawing is deliberately lazy.  ``--headless`` only disables the
        # window; it does not make ``draw_lines(...tolist())`` free.  The previous
        # implementation updated 8 thruster arrows for every parallel environment
        # during training, so visualization is now an explicit evaluation option.
        self._draw = None
        self.set_debug_vis(cfg.debug_vis)

    def _setup_scene(self) -> None:
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _current_action_envelope(self) -> torch.Tensor:
        """deploy_v6b only: ramp from [1]*6 down to the target envelope.

        Holds at the exact target envelope for the last
        ``action_envelope_curriculum_hold_fraction`` of training so the
        exported checkpoint is trained under the real target, not a
        mid-ramp one. No-op (returns the fixed target) unless
        ``enable_action_envelope_curriculum`` is set.
        """
        if not self.cfg.enable_action_envelope_curriculum:
            return self._action_envelope
        total = max(1, self.cfg.action_envelope_curriculum_total_steps)
        ramp_total = max(
            1.0, total * (1.0 - self.cfg.action_envelope_curriculum_hold_fraction)
        )
        progress = min(1.0, self.common_step_counter / ramp_total)
        full = torch.ones_like(self._action_envelope)
        return (1.0 - progress) * full + progress * self._action_envelope

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions.copy_(self._actions)
        self._raw_actions = actions.clone()
        self._pre_envelope_actions = actions.clamp(-1.0, 1.0)
        if self.cfg.enable_action_envelope_clamp:
            envelope = self._current_action_envelope()
            self._actions = self._pre_envelope_actions.clamp(-envelope, envelope)
        else:
            self._actions = self._pre_envelope_actions

    def _apply_action(self) -> None:
        """6-dim wrench → B_pinv 할당 → 8-thruster PWM → 추력/유체역학 계산 후 외력 적용."""
        # The policy speaks FLU/Z-up.  The allocation matrix is SNAME/FRD.
        # Cache the six signed scale factors at construction so this 100 Hz
        # hot path has no validation or device-to-host synchronization.
        tau_cmd_sname = self._actions * self._action_to_sname_multiplier
        self._wrench_requested_zup.copy_(
            tau_cmd_sname * self._sname_to_zup_sign
        )
        f_desired = (self._B_pinv @ tau_cmd_sname.unsqueeze(-1)).squeeze(-1)
        self._force_requested.copy_(f_desired)
        self._force_limited.copy_(self._thruster.clamp_thrust(f_desired))
        pwm = self._thruster.inverse_thrust(self._force_limited)

        f_thrust, t_thrust = self._thruster.compute(pwm)
        self._wrench_achieved_zup.copy_(torch.cat((f_thrust, t_thrust), dim=-1))
        # Hydrodynamics가 ν̇ 를 풀려면 **이 모듈 밖에서 몸체에 작용하는 전부**가
        # 필요하다 — 추력과 중력. 빠뜨리면 added mass가 그만큼 어긋난다.
        # 중력은 PhysX가 따로 적용하므로 여기서는 ν̇ 를 푸는 데만 쓴다.
        g_world = torch.zeros_like(f_thrust)
        g_world[:, 2] = -self._rigid_mass * 9.81
        g_body = math_utils.quat_apply(
            math_utils.quat_conjugate(self._robot.data.root_quat_w), g_world
        )
        other_wrench_b = torch.cat((f_thrust + g_body, t_thrust), dim=-1)
        f_hydro, t_hydro = self._hydro.compute(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            other_wrench_b,
        )

        total_forces  = (f_thrust + f_hydro).unsqueeze(1)
        total_torques = (t_thrust + t_hydro).unsqueeze(1)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=total_forces, torques=total_torques, body_ids=[0]
        )

        if self.cfg.debug_vis:
            self._visualize_thrust_arrows()

    def _ensure_debug_draw(self) -> None:
        """Lazily create debug-draw resources only for an opted-in visual run."""
        if self._draw is not None:
            return

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw

        t3 = torch.tensor([1.0, -1.0, -1.0], device=self.device)
        self._thruster_pos_zup = self._thruster._pos * t3
        self._thruster_dir_zup = self._thruster._dir * t3
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def _visualize_thrust_arrows(self) -> None:
        """스러스터 8개의 실제 추력을 world-space 화살표(화살촉 포함)로 그린다.

        `envs/traj_env.py`의 `BROVTrajEnv._visualize_debug_overlays()`에서 추력 화살표
        부분만 이식 — 위치/방향은 이미 아는 상수(`_thruster_pos_zup`/`_dir_zup`), 크기만 매
        스텝 `robots/dynamics/brov2/thruster.py`가 계산한 실제 추력값을 그대로 쓴다(COB 점 표시는
        `traj_env.py` 쪽에만 있음 — velEnv는 CoB가 env별로 랜덤화돼 있어 여기서는 생략).
        """
        if self._draw is None:
            return

        root_pos  = self._robot.data.root_pos_w                 # (N,3)
        root_quat = self._robot.data.root_quat_w                # (N,4)
        thrust    = self._thruster._last_thrust                 # (N,8)

        pos_b = self._thruster_pos_zup.unsqueeze(0).expand(self.num_envs, -1, -1)   # (N,8,3)
        dir_b = self._thruster_dir_zup.unsqueeze(0).expand(self.num_envs, -1, -1)   # (N,8,3)
        quat_ex = root_quat.unsqueeze(1).expand(-1, 8, -1)                          # (N,8,4)

        start_w = root_pos.unsqueeze(1) + math_utils.quat_apply(quat_ex, pos_b)     # (N,8,3)
        end_w   = start_w + math_utils.quat_apply(quat_ex, dir_b) * (
            thrust.unsqueeze(-1) * _THRUST_ARROW_SCALE
        )

        active = thrust.abs() > 1e-6   # 데드밴드로 추력=0인 스러스터는 화살표 생략
        shaft_starts = start_w[active]
        shaft_ends   = end_w[active]
        m = shaft_starts.shape[0]

        self._draw.clear_lines()
        if m > 0:
            line_starts = [shaft_starts]
            line_ends   = [shaft_ends]

            vec = shaft_ends - shaft_starts
            length = vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            fdir = vec / length

            ref_z = torch.tensor([0., 0., 1.], device=self.device).expand_as(fdir)
            ref_x = torch.tensor([1., 0., 0.], device=self.device).expand_as(fdir)
            use_alt = (fdir * ref_z).sum(-1, keepdim=True).abs() > 0.95
            ref = torch.where(use_alt, ref_x, ref_z)

            side1 = torch.cross(fdir, ref, dim=-1)
            side1 = side1 / side1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            side2 = torch.cross(fdir, side1, dim=-1)

            head_len = length.clamp(max=_ARROWHEAD_MAX_LEN / _ARROWHEAD_FRACTION) * _ARROWHEAD_FRACTION
            head_w = head_len * 0.5
            head_base = shaft_ends - fdir * head_len

            for side in (side1, -side1, side2, -side2):
                line_starts.append(shaft_ends)
                line_ends.append(head_base + side * head_w)

            all_starts = torch.cat(line_starts, dim=0).tolist()
            all_ends   = torch.cat(line_ends, dim=0).tolist()
            n_total = len(all_starts)
            colors  = [_THRUST_ARROW_COLOR] * n_total
            widths  = [_THRUST_ARROW_WIDTH] * m + [_THRUST_ARROW_WIDTH * 0.7] * (n_total - m)

            self._draw.draw_lines(all_starts, all_ends, colors, widths)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """현재/목표 자세, 목표 속도 화살표 — Project_BROV/brov_env.py의
        VisualizationMarkers 패턴을 그대로 따름 (IsaacLab DirectRLEnv 내장 훅)."""
        if debug_vis:
            self._ensure_debug_draw()
            if not hasattr(self, "cur_att_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/cur_attitude"
                marker_cfg.markers["arrow"].scale = (0.15, 0.15, 1.0)
                self.cur_att_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "tgt_att_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/tgt_attitude"
                marker_cfg.markers["arrow"].scale = (0.15, 0.15, 1.0)
                self.tgt_att_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "tgt_vel_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/tgt_velocity"
                marker_cfg.markers["arrow"].scale = (0.1, 0.1, 1.0)
                self.tgt_vel_visualizer = VisualizationMarkers(marker_cfg)

            self.cur_att_visualizer.set_visibility(True)
            self.tgt_att_visualizer.set_visibility(True)
            self.tgt_vel_visualizer.set_visibility(True)
        else:
            for name in ("cur_att_visualizer", "tgt_att_visualizer", "tgt_vel_visualizer"):
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """렌더 스텝마다 자동 호출됨(IsaacLab 내장). 로봇의 현재 위치에 현재/목표
        자세 화살표를 겹쳐 그리고, 목표 속도는 방향+크기(길이)로 별도 화살표."""
        pos  = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w

        self.cur_att_visualizer.visualize(translations=pos, orientations=quat)
        self.tgt_att_visualizer.visualize(translations=pos, orientations=self._q_d)

        v_world = math_utils.quat_apply(quat, self._v_d_b)
        speed   = v_world.norm(dim=-1)
        vel_quat = _heading_from_direction(v_world, self.device)
        vel_scales = torch.ones(self.num_envs, 3, device=self.device)
        vel_scales[:, 0] = (speed / _VEL_ARROW_SPEED_REF).clamp(min=0.1)   # 화살표 길이 = 속력 비례
        self.tgt_vel_visualizer.visualize(translations=pos, orientations=vel_quat, scales=vel_scales)

    def attach_guidance(self, guidance) -> None:
        """평가/배포용 — `LOSGuidance` 등 외부 유도기를 연결한다.

        연결되면 `_current_v_d_b()`가 학습용 랜덤 궤적 대신 매 스텝
        `guidance.compute(pos_env, root_quat_w)`로 v_d_b/q_d를 받아온다.
        `_get_observations()`/`_get_rewards()`는 무수정 — 호출 지점(`_current_v_d_b`)
        하나만 갈아끼우는 구조라, obs를 두 번 계산해서 z_v/z_q 적분이 중복
        누적되는 문제가 애초에 생기지 않는다. `guidance`는 `compute(pos_env,
        root_quat_w) -> (v_d_b, q_d)`와 `reset(env_ids)`만 있으면 됨(LOSGuidance
        인터페이스). 학습(`train.py`)에서는 절대 호출하지 않음 — 평가 스크립트 전용.
        """
        self._guidance = guidance

    def _current_v_d_b(self) -> torch.Tensor:
        """Return the selected desired-state contract at the current policy tick.

        ``legacy_eq9_velocity`` preserves model_299 exactly.  ``paper_ref_v1``
        uses Eq.9 for Frenet--Serret attitude and an independent exact 0.5 m/s
        S² velocity.  ``deploy_v2`` adds one balanced stop/restart/reversal
        transition.  Attached LOS guidance always takes precedence for eval.
        """
        guidance = getattr(self, "_guidance", None)
        if guidance is not None:
            pos_env = self._robot.data.root_pos_w - self.scene.env_origins
            v_d_b, q_d = guidance.compute(pos_env, self._robot.data.root_quat_w)
            self._q_d[:] = q_d
            self._command_transition_mask.zero_()
            self._command_transition_mode.fill_(-1)
            self._command_reversal_mask.zero_()
            return v_d_b

        t = self.episode_length_buf.float() * self._policy_dt
        if self._paper_reference is not None:
            v_d_b, q_d = self._paper_reference.sample(t)
            self._q_d.copy_(q_d)
            self._command_transition_mask.zero_()
            self._command_transition_mode.fill_(-1)
            self._command_reversal_mask.zero_()
            return v_d_b

        if self._deploy_scheduler is not None:
            sample = self._deploy_scheduler.sample(t)
            self._q_d.copy_(sample.desired_quaternion)
            self._command_transition_mask.copy_(sample.transition_mask)
            self._command_transition_mode.copy_(sample.transition_mode)
            self._command_reversal_mask.copy_(sample.reversal_mask)
            los_coupled_mask = getattr(sample, "los_coupled_mask", None)
            if los_coupled_mask is not None:
                # deploy_v6 item 4: mirror guidance/los_guidance.py's
                # heading_mode="align", where v_d_b is the world-frame LOS
                # direction rotated into the vehicle's CURRENT attitude
                # every tick -- not a static per-leg body-frame value. Uses
                # live root_quat_w, exactly like LOSGuidance.compute() does.
                q = self._robot.data.root_quat_w
                v_body_coupled = math_utils.quat_apply(
                    math_utils.quat_conjugate(q), sample.world_direction
                )
                return torch.where(
                    los_coupled_mask.unsqueeze(-1), v_body_coupled, sample.velocity_body
                )
            return sample.velocity_body

        # Frozen model_299 compatibility path.  This is the historical
        # misinterpretation of Eq.9 and must never be used for a new artifact.
        a, b, c = self.cfg.cmd_coeffs
        w = self.cfg.cmd_omega
        template = torch.stack(
            [torch.full_like(t, a), b * torch.sin(w * t), c * torch.cos(w * t)],
            dim=-1,
        )   # (N,3)
        return math_utils.quat_apply(self._cmd_quat, template)

    def _get_observations(self) -> dict:
        self._v_d_b = self._current_v_d_b()

        # deploy_v3 only (guarded by cfg flag -- deploy_v2's single mid-episode
        # transition keeps carrying the integral through it, unchanged, so old
        # checkpoints stay reproducible).  Mirrors the deployment-side
        # waypoint-transition reset (bumpless transfer): z_v/z_q jump to 0
        # exactly on the tick the active command leg changes.  This is safe
        # because z_v=0 is a value the policy has seen at every episode start;
        # per project_step2_brov_retrain_spec (memory), this alone only
        # accounts for a minority of the observed windup (~27% in the real
        # Case-A bag) -- the episode-length extension is what makes the
        # majority (sustained single-leg accumulation) trainable.
        if self.cfg.reset_integral_on_command_transition and bool(
            self._command_transition_mask.any()
        ):
            reset_mask = self._command_transition_mask.unsqueeze(-1)
            self._z_v = torch.where(reset_mask, torch.zeros_like(self._z_v), self._z_v)
            self._z_q = torch.where(reset_mask, torch.zeros_like(self._z_q), self._z_q)

        q = self._robot.data.root_quat_w
        q_e = math_utils.quat_mul(math_utils.quat_conjugate(self._q_d), q)   # q̄_d ⊗ q
        omega_b = self._robot.data.root_ang_vel_b

        # deploy_v4 (spec item B): the OBSERVED v_e_b uses a delayed/held/
        # noised "DVL measurement" of body velocity instead of the perfect
        # PhysX value.  Delay is applied to the measurement itself, not to
        # v_e_b -- v_d_b keeps updating in real time (guidance has no sensor
        # lag), matching brov_base/observation.py's real structure exactly.
        # _get_rewards() below independently recomputes v_e_b from the true
        # PhysX velocity, so only the *observation* -- never the training
        # signal -- sees this realism model.
        dvl_fresh_mask = None
        if self._dvl_realism is not None:
            t_episode = self.episode_length_buf.float() * self._policy_dt
            v_measured, dvl_fresh_mask = self._dvl_realism.step(
                self._robot.data.root_lin_vel_b, t_episode
            )
        else:
            v_measured = self._robot.data.root_lin_vel_b
        v_e_b = v_measured - self._v_d_b

        if self.cfg.observation_contract == "legacy_exact_0p5":
            self._z_v = self._z_v + v_e_b * self._policy_dt
            self._z_q = self._z_q + q_e[:, 1:] * self._policy_dt
            obs = torch.cat([q_e, v_e_b, omega_b, self._z_v, self._z_q], dim=-1)
        else:
            integrate_mask = (
                self.episode_length_buf != self._last_integrated_episode_step
            )
            integrate_velocity = (
                integrate_mask if dvl_fresh_mask is None
                else integrate_mask & dvl_fresh_mask
            )
            integrate_attitude = integrate_mask
            # deploy_v6 (2026-08-18 3rd-round diagnosis): per-axis
            # conditional-integration anti-windup. Halts a z_v/z_q axis's
            # integration on ticks where its own action is being clamped
            # away by the envelope -- reuses the same pre_envelope_actions
            # vs action_envelope comparison as the envelope_overflow reward
            # term in _get_rewards. brov_base/observation.py's independent
            # z_v/z_q reimplementation needs this same rule for parity.
            if self.cfg.enable_velocity_integral_antiwindup or (
                self.cfg.enable_attitude_integral_antiwindup
            ):
                not_saturated = (
                    self._pre_envelope_actions.abs() <= self._action_envelope
                )   # (N,6) bool: surge,sway,heave,roll,pitch,yaw
                if self.cfg.enable_velocity_integral_antiwindup:
                    integrate_velocity = (
                        integrate_velocity.unsqueeze(-1) & not_saturated[:, 0:3]
                    )
                if self.cfg.enable_attitude_integral_antiwindup:
                    integrate_attitude = (
                        integrate_mask.unsqueeze(-1) & not_saturated[:, 3:6]
                    )
            obs, self._z_v, self._z_q = build_velocity_observation(
                quaternion_error_wxyz=q_e,
                velocity_error_body=v_e_b,
                angular_velocity_body=omega_b,
                integral_velocity=self._z_v,
                integral_attitude=self._z_q,
                dt=self._policy_dt,
                integrate=integrate_mask,
                integrate_velocity=integrate_velocity,
                integrate_attitude=integrate_attitude,
                integral_velocity_limit=self.cfg.integral_velocity_limit,
                integral_attitude_limit=self.cfg.integral_attitude_limit,
            )
            self._last_integrated_episode_step.copy_(self.episode_length_buf)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Sim2Swim Eq.5-8: r = Σᵢrᵢ(q_e,v_e_b,ω_b) + r_q(각도, 별도 항) + r_a."""
        cfg = self.cfg

        q = self._robot.data.root_quat_w
        q_e = math_utils.quat_mul(math_utils.quat_conjugate(self._q_d), q)
        if self.cfg.observation_contract != "legacy_exact_0p5":
            q_e = canonicalize_quaternion(q_e)
        v_e_b = self._robot.data.root_lin_vel_b - self._v_d_b
        omega_b = self._robot.data.root_ang_vel_b

        if cfg.reward_profile == "paper_eq5_8":
            return (
                cfg.rew_w_quat   * torch.exp(-(q_e[:, 1:] ** 2).sum(-1))
                + cfg.rew_w_vel   * torch.exp(-(v_e_b ** 2).sum(-1))
                + cfg.rew_w_omega * torch.exp(-(omega_b ** 2).sum(-1))
                + cfg.rew_w_quat  * torch.exp(-math_utils.quat_error_magnitude(self._q_d, q))
                + cfg.rew_w_action* torch.exp(-self._actions.norm(dim=-1))
            )

        vel_error_sq = (v_e_b ** 2).sum(-1)
        action_delta = self._actions - self._prev_actions
        clamp_residual = (
            self._force_requested - self._force_limited
        ) / self._thruster_force_scale
        # raw_overflow sees the actor's pre-clamp output -- self._actions is
        # already clamped by _pre_physics_step, so it cannot distinguish a
        # raw output that barely exceeded [-1,1] from one that overshot it
        # by several units. Both existing action penalties above are blind
        # to this (see MK2_SIM2SIM_DEPLOY_RESULT.md sec.6/8 and
        # project_step2_brov_retrain_spec memory item D).
        raw_overflow = (self._raw_actions.abs() - 1.0).clamp_min(0.0)
        # deploy_v6 item 1: the actor's [-1,1]-clamped-but-pre-envelope
        # output beyond the deployment envelope has zero marginal physical
        # effect once enable_action_envelope_clamp makes self._actions the
        # envelope-limited value -- without this term the actor has no
        # reason to stop producing it (mirrors deploy_v5's raw_overflow
        # penalty, item D, keyed to the envelope instead of 1.0).
        envelope_overflow = (
            self._pre_envelope_actions.abs() - self._action_envelope
        ).clamp_min(0.0)
        return (
            cfg.deploy_rew_w_quat * torch.exp(-(q_e[:, 1:] ** 2).sum(-1))
            + cfg.deploy_rew_w_quat * torch.exp(-math_utils.quat_error_magnitude(self._q_d, q))
            + cfg.deploy_rew_w_vel_coarse * torch.exp(-vel_error_sq)
            + cfg.deploy_rew_w_vel_precision
            * torch.exp(-vel_error_sq / (cfg.deploy_rew_vel_sigma ** 2))
            + cfg.deploy_rew_w_omega * torch.exp(-(omega_b ** 2).sum(-1))
            - cfg.deploy_penalty_action_l2 * (self._actions ** 2).sum(-1)
            - cfg.deploy_penalty_action_delta_l2 * (action_delta ** 2).sum(-1)
            - cfg.deploy_penalty_thruster_clamp_l2 * (clamp_residual ** 2).sum(-1)
            - cfg.deploy_penalty_raw_overflow_l2 * (raw_overflow ** 2).sum(-1)
            - cfg.deploy_penalty_envelope_overflow_l2 * (envelope_overflow ** 2).sum(-1)
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_env = self._robot.data.root_pos_w - self.scene.env_origins
        out_of_bounds = (pos_env.abs() > cfg.max_bound).any(dim=-1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids_t)

        default_state = self._robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] += self.scene.env_origins[env_ids]
        default_state[:, 2]   = self.cfg.starting_depth

        self._robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        # Per-profile desired state.  Only the frozen compatibility profile
        # consumes the historical process-global random orientations.
        if self._paper_reference is not None:
            self._paper_reference.reset(env_ids_t)
        elif self._deploy_scheduler is not None:
            self._deploy_scheduler.reset(env_ids_t)
        else:
            self._cmd_quat[env_ids_t] = math_utils.random_orientation(
                n, device=self.device
            )
            self._q_d[env_ids_t] = math_utils.random_orientation(
                n, device=self.device
            )
        self._command_transition_mask[env_ids_t] = False
        self._command_transition_mode[env_ids_t] = -1
        self._command_reversal_mask[env_ids_t] = False
        self._z_v[env_ids_t] = 0.0
        self._z_q[env_ids_t] = 0.0
        self._last_integrated_episode_step[env_ids_t] = 0
        self._actions[env_ids_t] = 0.0
        self._prev_actions[env_ids_t] = 0.0
        self._raw_actions[env_ids_t] = 0.0
        self._pre_envelope_actions[env_ids_t] = 0.0
        self._force_requested[env_ids_t] = 0.0
        self._force_limited[env_ids_t] = 0.0
        self._wrench_requested_zup[env_ids_t] = 0.0
        self._wrench_achieved_zup[env_ids_t] = 0.0

        # Paper-disclosed mass randomization.  Inertia uses the identical
        # ratio and every reset starts from PhysX nominal values (no drift).
        if self.cfg.dr_enable_mass:
            mass_result = randomize_articulation_mass(
                self._robot,
                env_ids_t,
                relative_range=self.cfg.dr_mass_scale_range,
            )
            self._mass_scale[env_ids_t] = mass_result.scale.to(self.device)
        else:
            mass_result = randomize_articulation_mass(
                self._robot,
                env_ids_t,
                relative_range=(1.0, 1.0),
            )
            self._mass_scale[env_ids_t] = mass_result.scale.to(self.device)

        # ── 도메인 랜덤화: volume/CoB/added-mass ──
        vol_lo, vol_hi = self.cfg.dr_volume_range
        volume = math_utils.sample_uniform(vol_lo, vol_hi, (n,), self.device)

        cob_offset = _sample_from_sphere(n, self.cfg.dr_cob_radius, self.device)

        am_lo, am_hi = self.cfg.dr_added_mass_rot_range
        am_scale = math_utils.sample_uniform(am_lo, am_hi, (n,), self.device)
        added_mass_rot = self._nominal_added_mass_rot.unsqueeze(0) * am_scale.unsqueeze(-1)

        self._hydro.randomize(
            env_ids_t, volume=volume, cob_offset=cob_offset, added_mass_rot=added_mass_rot,
        )

        # deploy_v4 (spec item C): voltage sag (shared across all 8
        # thrusters -- one battery) x per-thruster manufacturing/wear
        # variance (independent per thruster).  1.0 = datasheet curve as-is.
        if self.cfg.dr_enable_thrust_scale:
            volt_lo, volt_hi = self.cfg.dr_thrust_voltage_scale_range
            voltage_scale = math_utils.sample_uniform(volt_lo, volt_hi, (n, 1), self.device)
            ind_lo, ind_hi = self.cfg.dr_thrust_individual_scale_range
            individual_scale = math_utils.sample_uniform(ind_lo, ind_hi, (n, 8), self.device)
            self._thruster.randomize(
                env_ids_t, thrust_scale=voltage_scale * individual_scale
            )

        self._thruster.reset(env_ids_t)
        self._hydro.reset(env_ids_t)
        if self._dvl_realism is not None:
            self._dvl_realism.reset(env_ids_t)

        guidance = getattr(self, "_guidance", None)
        if guidance is not None:
            guidance.reset(env_ids_t)

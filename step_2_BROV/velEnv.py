"""
BROV2 6DOF 속도/자세 컨트롤러 RL 환경
========================================
IsaacLab DirectRLEnv 기반. Sim2Swim(arXiv:2512.08656) 저수준 컨트롤러 재현.

계층 구조
---------
[Waypoints] → 3D LOS Guidance(고전 제어, 추후 구현) → v_d^b, q_d
                                                            │
                                                    BROVVelEnv(RL, 이 파일)
                                                            │  6-dim wrench
                                                    B_pinv 할당 → 8-thruster PWM
                                                            │
                                            hydrodynamics.py (기존, 변경 없음)

이 환경은 "경로 추종"을 학습하지 않는다 — body-frame 속도(v_d^b) + 자세(q_d)
명령을 얼마나 잘 추종하는지만 학습한다. 경로 추종은 정책 고정 후 LOS guidance
레이어가 담당 (env.py의 BROVTrajEnv는 end-to-end 대안으로 별도 유지).
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

# 추력 벡터 디버그 시각화용 (env.py의 BROVTrajEnv._visualize_debug_overlays()와 동일 방식)
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.util.debug_draw")
from isaacsim.util.debug_draw import _debug_draw

sys.path.insert(0, os.path.dirname(__file__))
from velEnvCfg import BROVVelEnvCfg
from hydrodynamics import BROV2ThrusterModel, BROV2Hydrodynamics, build_allocation_matrix
from env import _load_brov2_yaml, _coBM_vector_ned, _thruster_pos_dir_ned  # 단일 정본 재사용
from los_guidance import _heading_from_direction  # 방향벡터 → 자세 쿼터니언 (목표속도 화살표용)

_THRUST_ARROW_SCALE = 0.02   # [m/N] 화살표 길이 = 추력[N] * 이 값 (env.py와 동일)
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

        yaml_params = _load_brov2_yaml()
        hydro_coef  = yaml_params["hydro_coef"]
        cob_vector  = _coBM_vector_ned(yaml_params)
        thr_pos, thr_dir = _thruster_pos_dir_ned(yaml_params)

        self._thruster = BROV2ThrusterModel(
            self.num_envs, cfg.sim.dt, self.device, pos=thr_pos, dir=thr_dir,
        )
        self._hydro = BROV2Hydrodynamics(
            self.num_envs, cfg.sim.dt, self.device,
            volume            = yaml_params["volume"],
            cob_vector        = cob_vector,
            water_density     = yaml_params["environment"]["fluid_density"],
            added_mass        = hydro_coef["added_mass"],
            linear_damping    = hydro_coef["linear_damping"],
            quadratic_damping = hydro_coef["quadratic_damping"],
        )

        # 할당행렬 B(6,8) → pseudo-inverse(8,6). YAML 위치/방향에서 매번 계산
        # (하드코딩 금지 — coBM/hydro_coef와 동일한 단일 정본 원칙).
        B = build_allocation_matrix(self._thruster._pos, self._thruster._dir)
        self._B_pinv = torch.linalg.pinv(B).to(self.device)              # (8,6)
        self._f_max  = torch.tensor(cfg.f_max, device=self.device)       # (6,)

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
        self._actions  = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        # 추력 벡터 디버그 시각화 — 스러스터 위치/방향(SNAME) → Z-up body frame으로 미리 변환
        # (env.py의 BROVTrajEnv와 동일 방식 — 실제 생성된 self._thruster 인스턴스 값 사용).
        t3 = torch.tensor([1., -1., -1.], device=self.device)
        self._thruster_pos_zup = self._thruster._pos * t3   # (8,3)
        self._thruster_dir_zup = self._thruster._dir * t3   # (8,3)
        self._draw = _debug_draw.acquire_debug_draw_interface()

        # 현재/목표 자세·목표 속도 화살표(IsaacLab 내장 debug-vis 훅) — 활성화
        self.set_debug_vis(True)

    def _setup_scene(self) -> None:
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """6-dim wrench → B_pinv 할당 → 8-thruster PWM → 추력/유체역학 계산 후 외력 적용."""
        tau_cmd   = self._f_max * self._actions                              # (N,6)
        f_desired = (self._B_pinv @ tau_cmd.unsqueeze(-1)).squeeze(-1)       # (N,8) [N]
        pwm = self._thruster.inverse_thrust(f_desired)

        f_thrust, t_thrust = self._thruster.compute(pwm)
        f_hydro, t_hydro = self._hydro.compute(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
        )

        total_forces  = (f_thrust + f_hydro).unsqueeze(1)
        total_torques = (t_thrust + t_hydro).unsqueeze(1)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=total_forces, torques=total_torques, body_ids=[0]
        )

        self._visualize_thrust_arrows()

    def _visualize_thrust_arrows(self) -> None:
        """스러스터 8개의 실제 추력을 world-space 화살표(화살촉 포함)로 그린다.

        `env.py`의 `BROVTrajEnv._visualize_debug_overlays()`에서 추력 화살표 부분만
        이식 — 위치/방향은 이미 아는 상수(`_thruster_pos_zup`/`_dir_zup`), 크기만 매
        스텝 `hydrodynamics.py`가 계산한 실제 추력값을 그대로 쓴다(COB 점 표시는
        `env.py` 쪽에만 있음 — velEnv는 CoB가 env별로 랜덤화돼 있어 여기서는 생략).
        """
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
        """Sim2Swim Eq.9: v_d^b(t) = q_cmd ⊗ [a, b·sin(ωt), c·cos(ωt)].

        q_cmd는 에피소드마다 랜덤 샘플되는 방향(자세와는 별개) — 템플릿 곡선의
        모양은 고정하고 방향만 env마다 다양화한다. `attach_guidance()`로 외부
        유도기가 연결된 경우엔 이 자동생성 대신 유도기 출력을 그대로 쓴다
        (q_d도 여기서 함께 갱신 — 유도기가 자세 명령까지 책임지므로).
        """
        guidance = getattr(self, "_guidance", None)
        if guidance is not None:
            pos_env = self._robot.data.root_pos_w - self.scene.env_origins
            v_d_b, q_d = guidance.compute(pos_env, self._robot.data.root_quat_w)
            self._q_d[:] = q_d
            return v_d_b

        t = self.episode_length_buf.float() * self._policy_dt
        a, b, c = self.cfg.cmd_coeffs
        w = self.cfg.cmd_omega
        template = torch.stack(
            [torch.full_like(t, a), b * torch.sin(w * t), c * torch.cos(w * t)],
            dim=-1,
        )   # (N,3)
        return math_utils.quat_apply(self._cmd_quat, template)

    def _get_observations(self) -> dict:
        self._v_d_b = self._current_v_d_b()

        q = self._robot.data.root_quat_w
        q_e = math_utils.quat_mul(math_utils.quat_conjugate(self._q_d), q)   # q̄_d ⊗ q
        v_e_b = self._robot.data.root_lin_vel_b - self._v_d_b
        omega_b = self._robot.data.root_ang_vel_b

        self._z_v = self._z_v + v_e_b * self._policy_dt
        self._z_q = self._z_q + q_e[:, 1:] * self._policy_dt   # vector part만 적분

        obs = torch.cat([q_e, v_e_b, omega_b, self._z_v, self._z_q], dim=-1)   # 4+3+3+3+3=16
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Sim2Swim Eq.5-8: r = Σᵢrᵢ(q_e,v_e_b,ω_b) + r_q(각도, 별도 항) + r_a."""
        cfg = self.cfg

        q = self._robot.data.root_quat_w
        q_e = math_utils.quat_mul(math_utils.quat_conjugate(self._q_d), q)
        v_e_b = self._robot.data.root_lin_vel_b - self._v_d_b
        omega_b = self._robot.data.root_ang_vel_b

        return (
            cfg.rew_w_quat   * torch.exp(-(q_e[:, 1:] ** 2).sum(-1))                       # Eq.6, o_i=q_e
            + cfg.rew_w_vel   * torch.exp(-(v_e_b ** 2).sum(-1))                           # Eq.6, o_i=v_e^b
            + cfg.rew_w_omega * torch.exp(-(omega_b ** 2).sum(-1))                         # Eq.6, o_i=ω^b
            + cfg.rew_w_quat  * torch.exp(-math_utils.quat_error_magnitude(self._q_d, q))  # Eq.7, r_q (제곱 없음)
            + cfg.rew_w_action* torch.exp(-self._actions.norm(dim=-1))                     # Eq.8
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

        # 속도/자세 명령 재샘플 — 에피소드당 1회, episode_length_s 동안 고정
        self._cmd_quat[env_ids_t] = math_utils.random_orientation(n, device=self.device)
        self._q_d[env_ids_t]      = math_utils.random_orientation(n, device=self.device)
        self._z_v[env_ids_t] = 0.0
        self._z_q[env_ids_t] = 0.0

        # ── 도메인 랜덤화 1단계 (volume/CoB/added_mass, mass는 2단계로 연기) ──
        vol_lo, vol_hi = self.cfg.dr_volume_range
        volume = math_utils.sample_uniform(vol_lo, vol_hi, (n,), self.device)

        cob_offset = _sample_from_sphere(n, self.cfg.dr_cob_radius, self.device)

        am_lo, am_hi = self.cfg.dr_added_mass_rot_range
        am_scale = math_utils.sample_uniform(am_lo, am_hi, (n,), self.device)
        added_mass_rot = self._nominal_added_mass_rot.unsqueeze(0) * am_scale.unsqueeze(-1)

        self._hydro.randomize(
            env_ids_t, volume=volume, cob_offset=cob_offset, added_mass_rot=added_mass_rot,
        )

        self._thruster.reset(env_ids_t)
        self._hydro.reset(env_ids_t)

        guidance = getattr(self, "_guidance", None)
        if guidance is not None:
            guidance.reset(env_ids_t)

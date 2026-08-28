"""
BROV2 궤적 추종 RL 환경
========================
IsaacLab DirectRLEnv 기반.

동역학
------
MARINEGYM 방식 수중 유체역학 (robots/dynamics/ — step_2/step_3 공유):
  - robots.dynamics.brov2.thruster.BROV2ThrusterModel : PWM → body-frame 추력/토크
  - robots.dynamics.fossen.Hydrodynamics               : 부력 · 항력 · 추가질량 · Coriolis (Fossen NED 내부 계산)

궤적 추종
---------
env origin 기준 상대 좌표로 waypoint 리스트를 생성한다.
로봇이 waypoint_reach_threshold 이내에 진입하면 다음 waypoint 로 전환한다.
"""

from __future__ import annotations

import math
import sys
import os
from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate
import isaaclab.utils.math as math_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from envs.traj_env_cfg import BROVTrajEnvCfg
from robots.dynamics.brov2.thruster import BROV2ThrusterModel
from robots.dynamics.fossen import Hydrodynamics
from robots.dynamics.brov2.params import load_brov2_yaml, coBM_vector_ned, thruster_pos_dir_ned

# 추력 벡터 디버그 시각화용 (draw_lines(starts, ends, colors, widths), draw_points)
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.util.debug_draw")
from isaacsim.util.debug_draw import _debug_draw

# COM/관성은 직접 그리지 않고 PhysX 내장 "Body Mass Axes" 디버그 시각화를 사용한다
# (physxDebugView.py의 "Physics Debug" 패널과 동일한 기능을 코드로 켜는 것).
# COB는 PhysX에 없는 우리만의 개념이라 draw_points로 별도 표시.
from omni.physx import get_physx_visualization_interface

_THRUST_ARROW_SCALE = 0.02   # [m/N] 화살표 길이 = 추력[N] * 이 값
_THRUST_ARROW_COLOR = (1.0, 0.35, 0.0, 1.0)   # RGBA, 주황
_THRUST_ARROW_WIDTH = 3.0
_ARROWHEAD_FRACTION = 0.25   # 화살촉 길이 = 화살표 전체 길이의 이 비율
_ARROWHEAD_MAX_LEN  = 0.03   # [m] 화살촉 길이 상한

_COB_COLOR  = (0.0, 1.0, 1.0, 1.0)   # 시안
_COB_SIZE   = 12.0

# 로봇 고유 파라미터 YAML 로더 — robots/dynamics/brov2/params.py로 승격됨(2026-07).
# step_2_BROV뿐 아니라 향후 step_3도 재사용하므로 traj_env.py엔 더 이상 안 둠.


class BROVTrajEnv(DirectRLEnv):
    """BROV2 Heavy 궤적 추종 환경."""

    cfg: BROVTrajEnvCfg

    def __init__(self, cfg: BROVTrajEnvCfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode)

        self._robot: Articulation = self.scene.articulations["robot"]

        phys_dt = cfg.sim.dt
        yaml_params = load_brov2_yaml()
        hydro_coef  = yaml_params["hydro_coef"]
        cob_vector  = coBM_vector_ned(yaml_params)
        thr_pos, thr_dir = thruster_pos_dir_ned(yaml_params)
        self._thruster = BROV2ThrusterModel(
            self.num_envs, phys_dt, self.device, pos=thr_pos, dir=thr_dir,
        )
        self._volume        = yaml_params["volume"]
        self._water_density = yaml_params["environment"]["fluid_density"]
        self._hydro    = Hydrodynamics(
            self.num_envs, phys_dt, self.device,
            volume            = self._volume,
            cob_vector        = cob_vector,
            water_density     = self._water_density,
            added_mass        = hydro_coef["added_mass"],
            linear_damping    = hydro_coef["linear_damping"],
            quadratic_damping = hydro_coef["quadratic_damping"],
            # M_total = M_RB + M_A 를 만들기 위해 필요하다. Hydrodynamics가
            # added mass를 암묵적으로 풀기 때문 — compute() docstring 참조.
            rigid_mass        = yaml_params["expect"]["mass"],
            rigid_inertia     = yaml_params["expect"]["inertia"],
        )
        self._rigid_mass = float(yaml_params["expect"]["mass"])

        # 추력 벡터 디버그 시각화 — 스러스터 위치/방향(SNAME) → Z-up body frame으로 미리 변환.
        # 반드시 실제 생성된 self._thruster의 인스턴스 값(_pos/_dir, YAML에서 로드됨)을
        # 써야 한다 — 클래스 상수(BROV2ThrusterModel._POS/_DIR)는 YAML 없이 직접 생성할
        # 때만 쓰이는 fallback이라 시각화가 실제 물리와 어긋날 수 있다.
        t3 = torch.tensor([1., -1., -1.], device=self.device)
        self._thruster_pos_zup = self._thruster._pos * t3   # (8,3)
        self._thruster_dir_zup = self._thruster._dir * t3   # (8,3)
        self._draw = _debug_draw.acquire_debug_draw_interface()

        # COB(부력중심) 표시용 — buoyancy:comToCob(SNAME)를 다시 Z-up으로 되돌림 (T3 self-inverse)
        self._com_to_cob_zup = torch.tensor(
            [cob_vector[0], -cob_vector[1], -cob_vector[2]], device=self.device
        )

        # COM/관성은 직접 안 그리고 PhysX 내장 "Body Mass Axes" 시각화를 켠다
        vis = get_physx_visualization_interface()
        vis.enable_visualization(True)
        vis.set_visualization_scale(1.0)   # 기본 스케일이 0이라 이걸 안 하면 아무것도 안 보임
        vis.set_visualization_parameter("BodyMassAxes", True)

        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        self._waypoints = torch.zeros(
            self.num_envs, cfg.num_waypoints, 3, device=self.device
        )
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_wp_dist = torch.zeros(self.num_envs, device=self.device)

        self._generate_trajectories()

    def _setup_scene(self) -> None:
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _generate_trajectories(self) -> None:
        cfg = self.cfg
        N   = cfg.num_waypoints
        R   = cfg.trajectory_radius

        angles = torch.linspace(0, 2 * math.pi, N + 1, device=self.device)[:-1]
        x = R * torch.cos(angles)
        y = R * torch.sin(angles)

        if cfg.trajectory_type == "circle":
            z = torch.zeros(N, device=self.device)
        elif cfg.trajectory_type == "helix":
            z = torch.linspace(0.0, cfg.trajectory_height, N, device=self.device)
        else:
            raise ValueError(f"Unknown trajectory_type: '{cfg.trajectory_type}'.")

        wps = torch.stack([x, y, z], dim=-1)
        self._waypoints[:] = wps.unsqueeze(0).expand(self.num_envs, -1, -1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """추진기 + 수중 동역학 계산 후 루트 바디에 외력 적용."""
        f_thrust, t_thrust = self._thruster.compute(self._actions)

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

        self._visualize_debug_overlays()

    def _visualize_debug_overlays(self) -> None:
        """스러스터 8개의 실제 추력을 world-space 화살표(화살촉 포함)로 그리고,
        COB(부력중심) 위치를 점으로 표시한다. COM/관성은 PhysX 내장 "Body Mass Axes"
        시각화가 대신 담당한다 (__init__에서 활성화).

        base_link의 현재 world pose(위치+회전)만 쓰고, 스러스터가 실제 rigid body/joint로
        존재하는지 여부와는 완전히 무관하다 — 위치/방향은 이미 알고 있는 상수(_POS/_DIR)이고,
        크기만 매 스텝 robots/dynamics/brov2/thruster.py가 계산한 실제 추력값을 그대로 사용한다.
        """
        root_pos  = self._robot.data.root_pos_w                 # (N,3)
        root_quat = self._robot.data.root_quat_w                # (N,4)
        thrust    = self._thruster._last_thrust                 # (N,8)

        pos_b = self._thruster_pos_zup.unsqueeze(0).expand(self.num_envs, -1, -1)   # (N,8,3)
        dir_b = self._thruster_dir_zup.unsqueeze(0).expand(self.num_envs, -1, -1)   # (N,8,3)
        quat_ex = root_quat.unsqueeze(1).expand(-1, 8, -1)                          # (N,8,4)

        start_w = root_pos.unsqueeze(1) + quat_apply(quat_ex, pos_b)                # (N,8,3)
        end_w   = start_w + quat_apply(quat_ex, dir_b) * (thrust.unsqueeze(-1) * _THRUST_ARROW_SCALE)

        # 추력이 할당되지 않은(데드밴드 내 PWM → 정확히 0.0) 스러스터는 화살표 자체를 그리지 않음
        active = thrust.abs() > 1e-6   # (N,8)
        shaft_starts = start_w[active]   # (M,3)
        shaft_ends   = end_w[active]     # (M,3)
        m = shaft_starts.shape[0]

        self._draw.clear_lines()
        if m > 0:
            line_starts = [shaft_starts]
            line_ends   = [shaft_ends]

            # 화살촉: 화살표 축에 수직인 두 방향(side1, side2)을 구해서 4방향 wing을 그린다.
            # 임의 축과 정확히 평행해서 외적이 퇴화하는 경우를 대비해 참조축 두 개 중
            # 덜 평행한 쪽을 골라 사용한다.
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

        # COB(부력중심) 점 표시
        cob_w = root_pos + quat_apply(root_quat, self._com_to_cob_zup.unsqueeze(0).expand(self.num_envs, -1))
        self._draw.clear_points()
        self._draw.draw_points(
            cob_w.tolist(),
            [_COB_COLOR] * self.num_envs,
            [_COB_SIZE] * self.num_envs,
        )

    def _get_observations(self) -> dict:
        wp_world = self._current_waypoint_world()

        delta_world = wp_world - self._robot.data.root_pos_w
        delta_b = quat_apply(
            quat_conjugate(self._robot.data.root_quat_w),
            delta_world,
        )

        wp_dist  = torch.norm(delta_world, dim=-1, keepdim=True)
        wp_dir_b = delta_b / (wp_dist + 1e-6)

        pos_env = self._robot.data.root_pos_w - self.scene.env_origins

        obs = torch.cat([
            pos_env,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            wp_dir_b,
            wp_dist,
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        wp_world = self._current_waypoint_world()
        wp_dist  = torch.norm(wp_world - self._robot.data.root_pos_w, dim=-1)

        rew_progress = cfg.rew_scale_progress * (self._prev_wp_dist - wp_dist)
        self._prev_wp_dist = wp_dist.detach()

        reached = (wp_dist < cfg.waypoint_reach_threshold)
        rew_waypoint = cfg.rew_scale_waypoint * reached.float()
        self._wp_idx = torch.where(
            reached,
            (self._wp_idx + 1) % cfg.num_waypoints,
            self._wp_idx,
        )

        rew_action = -cfg.rew_scale_action * torch.norm(self._actions, dim=-1)

        body_z = torch.zeros(self.num_envs, 3, device=self.device)
        body_z[:, 2] = 1.0
        up_world = quat_apply(self._robot.data.root_quat_w, body_z)
        rew_upright = cfg.rew_scale_upright * up_world[:, 2]

        return rew_progress + rew_waypoint + rew_action + rew_upright

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg     = self.cfg
        pos_env = self._robot.data.root_pos_w - self.scene.env_origins

        out_of_bounds = (
            (torch.abs(pos_env[:, 0]) > cfg.max_bound_x) |
            (torch.abs(pos_env[:, 1]) > cfg.max_bound_y) |
            (torch.abs(pos_env[:, 2]) > cfg.max_bound_z)
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        default_state = self._robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] += self.scene.env_origins[env_ids]
        default_state[:, 2]   = self.cfg.starting_depth

        self._robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        self._wp_idx[env_ids] = 0
        wp_world = self._current_waypoint_world(env_ids)
        self._prev_wp_dist[env_ids] = torch.norm(
            wp_world - default_state[:, :3], dim=-1
        )

        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        self._thruster.reset(env_ids_t)
        self._hydro.reset(env_ids_t)

    def _current_waypoint_world(
        self,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if env_ids is None:
            idx   = self._wp_idx
            env_i = torch.arange(self.num_envs, device=self.device)
            return self._waypoints[env_i, idx] + self.scene.env_origins

        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        idx       = self._wp_idx[env_ids_t]
        return self._waypoints[env_ids_t, idx] + self.scene.env_origins[env_ids_t]

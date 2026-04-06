"""
BROV2 궤적 추종 RL 환경
========================
IsaacLab DirectRLEnv 기반.

동역학
------
MARINEGYM 방식 수중 유체역학 (hydrodynamics.py):
  - BROV2ThrusterModel  : PWM → body-frame 추력/토크
  - BROV2Hydrodynamics  : 부력 · 항력 · 추가질량 · Coriolis

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

# 같은 패키지 내 모듈
sys.path.insert(0, os.path.dirname(__file__))
from envCfg import BROVTrajEnvCfg
from hydrodynamics import BROV2ThrusterModel, BROV2Hydrodynamics


class BROVTrajEnv(DirectRLEnv):
    """BROV2 Heavy 궤적 추종 환경 (MarineGym 동역학 적용)."""

    cfg: BROVTrajEnvCfg

    # ──────────────────────────────────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, cfg: BROVTrajEnvCfg, render_mode: str | None = None):
        # DirectRLEnv.__init__ 내부에서:
        #   1) InteractiveScene 이 BROVSceneCfg 에 정의된 에셋을 모두 스폰
        #   2) _setup_scene() 호출 (clone + filter)
        super().__init__(cfg, render_mode)

        # ── 로봇 핸들 (InteractiveScene 자동 생성) ──────────────────────────
        # BROVSceneCfg.robot: ArticulationCfg → scene.articulations["robot"]
        self._robot: Articulation = self.scene.articulations["robot"]

        # ── 수중 동역학 모듈 ────────────────────────────────────────────────
        # 물리 스텝 dt 기준 (정책 dt 아님)
        phys_dt = cfg.sim.dt
        self._thruster = BROV2ThrusterModel(self.num_envs, phys_dt, self.device)
        self._hydro    = BROV2Hydrodynamics(
            self.num_envs, phys_dt, self.device,
            volume       = cfg.volume,
            cob_offset   = cfg.cob_offset,
            water_density= cfg.water_density,
        )

        # ── 버퍼 ────────────────────────────────────────────────────────────
        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        # waypoints: (num_envs, num_waypoints, 3) — env origin 기준 상대 좌표
        self._waypoints = torch.zeros(
            self.num_envs, cfg.num_waypoints, 3, device=self.device
        )
        # 현재 목표 waypoint 인덱스
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 이전 스텝의 waypoint 까지 거리 (progress 보상 계산용)
        self._prev_wp_dist = torch.zeros(self.num_envs, device=self.device)

        # ── 궤적 생성 ───────────────────────────────────────────────────────
        self._generate_trajectories()

    # ──────────────────────────────────────────────────────────────────────────
    # 씬 구성
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        """
        환경 복제 및 충돌 필터 설정.

        BROVSceneCfg 에 정의된 에셋(robot, seafloor, dome_light)은
        InteractiveScene 이 이미 스폰했으므로 여기서는 클로닝만 수행한다.
        """
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    # ──────────────────────────────────────────────────────────────────────────
    # 궤적 생성
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_trajectories(self) -> None:
        """
        각 환경의 waypoint 리스트 생성 (env origin 기준 상대 좌표).

        trajectory_type
        ---------------
        "circle" : 수평 원형 궤적 (고도 일정)
        "helix"  : 나선형 궤적 (고도 점진 변화)

        TODO: 랜덤 궤적, B-spline 기반 연속 궤적 등으로 확장 가능.
        """
        cfg = self.cfg
        N   = cfg.num_waypoints
        R   = cfg.trajectory_radius

        # 균등 분할 각도
        angles = torch.linspace(0, 2 * math.pi, N + 1, device=self.device)[:-1]  # (N,)

        x = R * torch.cos(angles)
        y = R * torch.sin(angles)

        if cfg.trajectory_type == "circle":
            z = torch.zeros(N, device=self.device)
        elif cfg.trajectory_type == "helix":
            z = torch.linspace(0.0, cfg.trajectory_height, N, device=self.device)
        else:
            raise ValueError(f"Unknown trajectory_type: '{cfg.trajectory_type}'. "
                             "Choose 'circle' or 'helix'.")

        # waypoints (N, 3) → (1, N, 3) → (num_envs, N, 3)
        wps = torch.stack([x, y, z], dim=-1)                        # (N, 3)
        self._waypoints[:] = wps.unsqueeze(0).expand(self.num_envs, -1, -1)

    # ──────────────────────────────────────────────────────────────────────────
    # RL 인터페이스
    # ──────────────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """행동 클리핑 후 버퍼 저장 (물리 스텝 직전)."""
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """
        추진기 + 수중 동역학 계산 후 로봇에 외력 적용.

        흐름
        ----
        1. BROV2ThrusterModel  → PWM → body-frame 추력/토크
        2. BROV2Hydrodynamics  → 부력/항력/추가질량/Coriolis (body frame)
        3. 합산 후 set_external_force_and_torque 호출

        Note
        ----
        IsaacLab Articulation.set_external_force_and_torque 는
        local (body) frame 기준 힘을 받는다.
        body_ids=[0] = 루트 바디(base_link) 에만 적용.
        실제 body index 는 USD 구조에 따라 다를 수 있으므로 확인 필요.
        """
        # --- 추진기 힘/토크 (body frame) ---
        f_thrust, t_thrust = self._thruster.compute(self._actions)

        # --- 수중 동역학 힘/토크 (body frame) ---
        f_hydro, t_hydro = self._hydro.compute(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
        )

        # --- 합산 및 적용 ---
        total_forces  = (f_thrust + f_hydro).unsqueeze(1)    # (num_envs, 1, 3)
        total_torques = (t_thrust + t_hydro).unsqueeze(1)    # (num_envs, 1, 3)

        # body_ids=[0]: 루트 바디(base_link) 에만 적용
        self._robot.set_external_force_and_torque(
            total_forces, total_torques, body_ids=[0]
        )

    def _get_observations(self) -> dict:
        """
        관측 벡터 구성 (17-dim).

        [pos_env(3), quat(4), lin_vel_b(3), ang_vel_b(3), wp_dir_b(3), wp_dist(1)]
        """
        # 현재 waypoint (world frame)
        wp_world = self._current_waypoint_world()          # (num_envs, 3)

        # 로봇 → waypoint 벡터 (world frame)
        delta_world = wp_world - self._robot.data.root_pos_w   # (num_envs, 3)

        # body frame 으로 변환
        delta_b = quat_apply(
            quat_conjugate(self._robot.data.root_quat_w),
            delta_world,
        )  # (num_envs, 3)

        wp_dist   = torch.norm(delta_world, dim=-1, keepdim=True)  # (num_envs, 1)
        wp_dir_b  = delta_b / (wp_dist + 1e-6)                     # 정규화 방향벡터

        # env origin 기준 로봇 위치
        pos_env = self._robot.data.root_pos_w - self.scene.env_origins   # (num_envs, 3)

        obs = torch.cat([
            pos_env,                            # 3
            self._robot.data.root_quat_w,       # 4
            self._robot.data.root_lin_vel_b,    # 3
            self._robot.data.root_ang_vel_b,    # 3
            wp_dir_b,                           # 3
            wp_dist,                            # 1
        ], dim=-1)   # → 17

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        보상 계산.

        1. rew_progress   : 이전 스텝 대비 waypoint 접근량 (dense)
        2. rew_waypoint   : waypoint 도달 보너스 + 다음 waypoint 로 전환
        3. rew_action     : 액션 크기 페널티 (에너지 효율 유도)
        4. rew_upright    : 자세 유지 (body Z 축이 world Z 와 정렬)
        """
        cfg = self.cfg

        # 현재 waypoint 까지 거리
        wp_world = self._current_waypoint_world()
        wp_dist  = torch.norm(wp_world - self._robot.data.root_pos_w, dim=-1)  # (num_envs,)

        # 1. 진행 보상
        rew_progress = cfg.rew_scale_progress * (self._prev_wp_dist - wp_dist)
        self._prev_wp_dist = wp_dist.detach()

        # 2. Waypoint 도달 보너스
        reached = (wp_dist < cfg.waypoint_reach_threshold)
        rew_waypoint = cfg.rew_scale_waypoint * reached.float()
        # 도달한 환경만 다음 waypoint 로 전환 (순환)
        self._wp_idx = torch.where(
            reached,
            (self._wp_idx + 1) % cfg.num_waypoints,
            self._wp_idx,
        )

        # 3. 액션 정규화
        rew_action = -cfg.rew_scale_action * torch.norm(self._actions, dim=-1)

        # 4. 자세 유지 (body Z 축을 world Z 방향과 정렬)
        body_z_world = torch.zeros(self.num_envs, 3, device=self.device)
        body_z_world[:, 2] = 1.0
        up_world = quat_apply(self._robot.data.root_quat_w, body_z_world)  # (num_envs, 3)
        rew_upright = cfg.rew_scale_upright * up_world[:, 2]   # cos(기울기)

        return rew_progress + rew_waypoint + rew_action + rew_upright

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        종료 조건.

        terminated : 경계 이탈 (env origin 기준)
        truncated  : 에피소드 시간 초과
        """
        cfg     = self.cfg
        pos_env = self._robot.data.root_pos_w - self.scene.env_origins   # (num_envs, 3)

        out_of_bounds = (
            (torch.abs(pos_env[:, 0]) > cfg.max_bound_x) |
            (torch.abs(pos_env[:, 1]) > cfg.max_bound_y) |
            (torch.abs(pos_env[:, 2]) > cfg.max_bound_z)
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """
        지정 환경 리셋.

        순서
        ----
        1. super()._reset_idx  (episode_length_buf 등 내부 버퍼 초기화)
        2. 로봇 루트 상태 재설정  (위치·방향·속도)
        3. 관절 상태 초기화
        4. Waypoint 인덱스·이전 거리 초기화
        5. 동역학 모듈 상태 초기화
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # -- 루트 상태 재설정 --
        default_state = self._robot.data.default_root_state[env_ids].clone()
        # env origin 오프셋 반영
        default_state[:, :3] += self.scene.env_origins[env_ids]
        # 수심 설정 (cfg.starting_depth → world Z)
        default_state[:, 2] = self.cfg.starting_depth

        self._robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        # -- 관절 상태 초기화 (추진기 관절 속도 = 0) --
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )

        # -- Waypoint 인덱스 및 이전 거리 초기화 --
        self._wp_idx[env_ids] = 0
        wp_world = self._current_waypoint_world(env_ids)
        self._prev_wp_dist[env_ids] = torch.norm(
            wp_world - default_state[:, :3], dim=-1
        )

        # -- 동역학 모듈 초기화 --
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        self._thruster.reset(env_ids_t)
        self._hydro.reset(env_ids_t)

    # ──────────────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _current_waypoint_world(
        self,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """
        현재 목표 waypoint 의 world-frame 좌표 반환.

        waypoints 는 env origin 기준 상대 좌표로 저장되어 있으므로
        scene.env_origins 를 더해 world 좌표로 변환한다.
        """
        if env_ids is None:
            idx     = self._wp_idx                                        # (num_envs,)
            env_i   = torch.arange(self.num_envs, device=self.device)
            wps_rel = self._waypoints[env_i, idx]                         # (num_envs, 3)
            return wps_rel + self.scene.env_origins

        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        idx       = self._wp_idx[env_ids_t]
        wps_rel   = self._waypoints[env_ids_t, idx]
        return wps_rel + self.scene.env_origins[env_ids_t]

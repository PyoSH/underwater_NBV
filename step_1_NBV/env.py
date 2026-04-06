from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import os

import omni.usd
from pxr import UsdLux

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from envCfg import OceanEnvCfg

class OceanEnv(DirectRLEnv):
    """카메라와 조명을 이동시키며 대상 물체를 탐색하는 병렬 RL 환경."""

    cfg: OceanEnvCfg

    # ── 초기화 ───────────────────────────────────────────────────────────────
    def __init__(self, cfg: OceanEnvCfg, render_mode: str | None = None):
        if cfg.debug_vis:
            cfg.scene.camera.enable_viewport = True
            cfg.scene.camera.viewport_env_id = 0
            
        super().__init__(cfg, render_mode)

        # RigidObject 핸들 (InteractiveScene 이 자동 생성)
        self._sensor_rig   = self.scene["sensor_rig"]
        self._light_rig = self.scene["light_rig"]

        self._camera = self.scene["camera"]

        rock_local = torch.tensor([0.0, 0.0, -3.0], device=self.device)
        self.rock_pos = self.scene.env_origins + rock_local  # (num_envs, 3)
        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        self._setup_vis_markers()        

    # ── 방향 시각화 마커 ──────────────────────────────────────────────────────

    def _setup_vis_markers(self) -> None:
        """debug_draw 인터페이스 초기화 (Isaac Sim 버전별 경로 순차 시도).

        debug_draw 는 뷰포트 오버레이에만 선을 그림.
        USD 프림을 생성하지 않으므로 카메라 센서 이미지에 영향 없음.
        모듈을 찾지 못하면 self._draw = None 으로 시각화 비활성화.
        """
        self._draw = None

        _candidates = [
            # Isaac Sim 2023.x
            lambda: __import__(
                "omni.isaac.debug_draw", fromlist=["_debug_draw"]
            )._debug_draw.acquire_debug_draw_interface(),
            # Isaac Sim 4.x (isaacsim 패키지)
            lambda: __import__(
                "isaacsim.util.debug_draw", fromlist=["_debug_draw"]
            )._debug_draw.acquire_debug_draw_interface(),
            # Kit 106+
            lambda: __import__(
                "omni.debugdraw", fromlist=["get_debug_draw_interface"]
            ).get_debug_draw_interface(),
        ]

        for _try in _candidates:
            try:
                self._draw = _try()
                return
            except (ModuleNotFoundError, AttributeError):
                continue

        print("[OceanNBVEnv] debug_draw 모듈 없음 → 방향 시각화 비활성화")

    def _update_vis_markers(self) -> None:
        """카메라·조명의 X/Y/Z 3축을 뷰포트에 선분으로 실시간 표시.

        cfg.debug_vis_env_id:
            -1 → 전체 num_envs
            ≥0 → 해당 인덱스 환경만
        """
        if self._draw is None:
            return

        self._draw.clear_lines()

        L   = 0.30   # 화살표 길이 [m]
        W   = 3.0    # 선 굵기 

        # 표준 RGB 축 색상: +X=빨강, +Y=초록, +Z=파랑 (카메라·조명 공통)
        AXIS_COLS = [
            (1.0, 0.0, 0.0, 1.0),  # +X: 빨강
            (0.0, 1.0, 0.0, 1.0),  # +Y: 초록
            (0.0, 0.0, 1.0, 1.0),  # +Z: 파랑
        ]
        # 로컬 축 벡터 (body frame)
        AXES = [
            torch.tensor([[1., 0., 0.]], device=self.device),  # +X
            torch.tensor([[0., 1., 0.]], device=self.device),  # +Y
            torch.tensor([[0., 0., 1.]], device=self.device),  # +Z
        ]

        eid = self.cfg.debug_vis_env_id
        indices = [eid] if eid >= 0 else list(range(self.num_envs))

        starts, ends, colors, widths = [], [], [], []

        for i in indices:
            cp = self.cam_pos[i].cpu()
            cq = self.cam_orient[i:i+1]
            lp = self.light_pos[i].cpu()
            lq = self.light_orient[i:i+1]

            for ax, col in zip(AXES, AXIS_COLS):
                # 카메라 축
                aw = quat_apply(cq, ax)[0].cpu()
                starts.append(cp.tolist())
                ends.append((cp + aw * L).tolist())
                colors.append(col); widths.append(W)

                # 조명 축
                aw = quat_apply(lq, ax)[0].cpu()
                starts.append(lp.tolist())
                ends.append((lp + aw * L).tolist())
                colors.append(col); widths.append(W)

        if starts:
            self._draw.draw_lines(starts, ends, colors, widths)

    # ── 씬 구성 ──────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        """Camera, SphereLight, DirectionCone 자식 프림을 각 환경에 추가.

        OceanSceneCfg 의 에셋(Seafloor, Rock, CameraRig, LightRig)은
        InteractiveScene 이 이미 스폰했으므로 추가 자식만 생성.
        """
        stage = omni.usd.get_context().get_stage()

        for env_idx in range(self.num_envs):
            env_ns = f"/World/envs/env_{env_idx}"
            self._add_light_children(stage, f"{env_ns}/LightRig/SphereLight")

    def _add_light_children(self, stage, light_prim_path: str) -> None:
        # ── SphereLight + ShapingAPI ──────────────────────────────────────
        light = stage.GetPrimAtPath(light_prim_path)

        shaping = UsdLux.ShapingAPI.Apply(light)
        shaping.GetShapingConeAngleAttr().Set(40.0)   # 반각 [도]
        shaping.GetShapingConeSoftnessAttr().Set(0.1)

    # ── 행동 적용 ─────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """행동 클리핑 후 저장 (물리 스텝 직전)."""
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """RigidObject GPU tensor API 로 선속도·각속도 설정.

        action[0:3]  = 카메라 선속도  [vx, vy, vz]
        action[3:6]  = 카메라 각속도  [wx, wy, wz]
        action[6:9]  = 조명 선속도    [vx, vy, vz]
        action[9:12] = 조명 각속도    [wx, wy, wz]
        """
        lv = self.cfg.max_velocity
        av = self.cfg.max_angular_velocity

        # (num_envs, 6): [lin_vel(3), ang_vel(3)]
        cam_vel = torch.cat([
            self._actions[:, 0:3] * lv,
            self._actions[:, 3:6] * av,
        ], dim=-1)
        light_vel = torch.cat([
            self._actions[:, 6:9]  * lv,
            self._actions[:, 9:12] * av, # roll은 안하고자 함.
        ], dim=-1)

        self._sensor_rig.write_root_velocity_to_sim(cam_vel)
        self._light_rig.write_root_velocity_to_sim(light_vel)

    # ── 상태 프로퍼티 ─────────────────────────────────────────────────────────

    @property
    def cam_pos(self) -> torch.Tensor:
        """카메라 리그 월드 위치 (num_envs, 3)."""
        return self._sensor_rig.data.root_pos_w

    @property
    def cam_orient(self) -> torch.Tensor:
        """카메라 리그 월드 자세 쿼터니언 [w,x,y,z] (num_envs, 4)."""
        return self._sensor_rig.data.root_quat_w

    @property
    def light_pos(self) -> torch.Tensor:
        """조명 리그 월드 위치 (num_envs, 3)."""
        return self._light_rig.data.root_pos_w

    @property
    def light_orient(self) -> torch.Tensor:
        """조명 리그 월드 자세 쿼터니언 [w,x,y,z] (num_envs, 4)."""
        return self._light_rig.data.root_quat_w

    # ── 관측 ─────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        관측 벡터
        """
        cam_to_rock = self.rock_pos - self.cam_pos
        obs = torch.cat(
            [self.cam_pos, self.cam_orient,
             self.light_pos, self.light_orient,
             cam_to_rock],
            dim=-1,
        )

        # 방향 마커 실시간 업데이트 (PhysX runtime 데이터 사용)
        if self.cfg.debug_vis:
            self._update_vis_markers()

        return {"policy": obs}

    # ── 보상 ─────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        reward_scalar = cfg.w_distance * 0.3 + cfg.w_direction * 0.3 + cfg.w_baseline * 0.3

        return torch.full((self.num_envs,), reward_scalar, device=self.device)
    
    # ── 종료 조건 ─────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        terminated: 카메라가 workspace_radius 밖으로 이탈
        truncated:  에피소드 길이 초과
        """
        dist_cam   = torch.norm(self.cam_pos - self.rock_pos, dim=-1)
        terminated = dist_cam > self.cfg.workspace_radius
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── 리셋 ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)

        cfg = self.cfg
        n   = len(env_ids)
 
        # rock 월드 위치 (해당 env 들만)
        rock_np = self.rock_pos[env_ids].cpu().numpy()  # (n, 3)
 
        # ── 카메라 위치: 구면 좌표 샘플링 (reset_radius_min ~ reset_radius_max) ──
        r     = np.random.uniform(cfg.reset_radius_min, cfg.reset_radius_max, n)
        theta = np.random.uniform(cfg.reset_theta_min,  cfg.reset_theta_max,  n)
        phi   = np.random.uniform(0.0, 2.0 * np.pi, n)
 
        offsets = r[:, None] * np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], axis=1)  # (n, 3)
 
        cam_np = rock_np + offsets  # (n, 3) 월드 좌표
 
        # ── 조명 위치: look-at 방향과 수직인 방향으로 baseline 오프셋 ──────────
        look_dirs = rock_np - cam_np
        look_dirs /= np.linalg.norm(look_dirs, axis=1, keepdims=True) + 1e-8
 
        ref = np.tile([0.0, 0.0, 1.0], (n, 1))
        perp = np.cross(look_dirs, ref)
        perp_norm = np.linalg.norm(perp, axis=1, keepdims=True)
        # look_dir 이 +Z 와 평행한 경우 fallback: +X 기준
        fallback_mask = (perp_norm[:, 0] < 1e-6)
        if fallback_mask.any():
            fallback_ref = np.tile([1.0, 0.0, 0.0], (fallback_mask.sum(), 1))
            perp[fallback_mask] = np.cross(look_dirs[fallback_mask], fallback_ref)
            perp_norm[fallback_mask] = np.linalg.norm(
                perp[fallback_mask], axis=1, keepdims=True
            )
        perp /= perp_norm + 1e-8
        light_np = cam_np + perp * cfg.light_baseline  # (n, 3)
 
        # ── 카메라·조명 look-at 쿼터니언 (+X 전방이 rock 을 향하도록) ──────────
        cam_quats   = np.stack([self._look_at_quat(cam_np[i],   rock_np[i]) for i in range(n)])
        light_quats = np.stack([self._look_at_quat(light_np[i], rock_np[i]) for i in range(n)])
 
        # ── RigidObject 상태 쓰기: [pos(3), quat(4), lin_vel(3), ang_vel(3)] ──
        cam_state   = torch.zeros(n, 13, device=self.device)
        light_state = torch.zeros(n, 13, device=self.device)
 
        cam_state[:, 0:3] = torch.tensor(cam_np,     dtype=torch.float32, device=self.device)
        cam_state[:, 3:7] = torch.tensor(cam_quats,  dtype=torch.float32, device=self.device)
        # lin_vel, ang_vel 은 zeros 로 초기화됨
 
        light_state[:, 0:3] = torch.tensor(light_np,    dtype=torch.float32, device=self.device)
        light_state[:, 3:7] = torch.tensor(light_quats, dtype=torch.float32, device=self.device)
 
        env_ids_t = torch.tensor(list(env_ids), dtype=torch.int64, device=self.device)
        self._sensor_rig.write_root_state_to_sim(cam_state,   env_ids=env_ids_t)
        self._light_rig.write_root_state_to_sim(light_state,  env_ids=env_ids_t)

        if self.cfg.water_dr_enabled:
            self._randomize_water_params()
    
    def _randomize_water_params(self) -> None:
        dr = self.cfg.water_dr

        def rand_tuple(mn, mx):
            return tuple(float(np.random.uniform(mn[i], mx[i])) for i in range(3))

        self._camera.set_water_params(
            backscatter_value = rand_tuple(dr.backscatter_value_min,
                                        dr.backscatter_value_max),
            atten_coeff       = rand_tuple(dr.atten_coeff_min,
                                        dr.atten_coeff_max),
            backscatter_coeff = rand_tuple(dr.backscatter_coeff_min,
                                        dr.backscatter_coeff_max),
        )
        
    def _look_at_quat(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """
        +X = forward(to_pos 방향),  +Z = up 기준 roll 고정.
        반환: [w, x, y, z]
        """
        forward = to_pos - from_pos
        norm = np.linalg.norm(forward)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        forward = forward / norm

        up = np.array([0.0, 0.0, 1.0])

        # forward ≈ ±Z 일 때 up 벡터 fallback
        if abs(np.dot(forward, up)) > 1.0 - 1e-6:
            up = np.array([0.0, 1.0, 0.0])

        # 직교 기저 구성 (body frame: X=forward, Y=left, Z=up)
        right   = np.cross(forward, up);  right   /= np.linalg.norm(right)
        up_ortho = np.cross(right, forward)           # 재정규화 불필요하지만 안전하게:
        up_ortho /= np.linalg.norm(up_ortho)

        # 회전행렬 → 쿼터니언 (Shepperd method)
        # 열: [forward, -right, up_ortho]  ← X=forward, Y=-right(=left), Z=up
        R = np.stack([forward, -right, up_ortho], axis=1)  # (3,3), 열=축

        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s

        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)   # 수치 오차 보정
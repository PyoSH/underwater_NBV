from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import os

import omni.usd
from pxr import UsdLux
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply
from env_utils import EnvUtilsMixin
from env_reward import EnvRewardMixin
from envCfg import OceanEnvCfg
from pathlib import Path

class OceanEnv(EnvUtilsMixin,EnvRewardMixin,DirectRLEnv):
    """카메라와 조명을 이동시키며 대상 물체를 탐색하는 병렬 RL 환경."""
    cfg: OceanEnvCfg

    # ── 초기화 ───────────────────────────────────────────────────────────────
    def __init__(self, cfg: OceanEnvCfg, render_mode: str | None = None):
        if cfg.debug_vis:
            # cfg.scene.camera.enable_viewport = True
            cfg.scene.camera.viewport_env_id = 0
            
        super().__init__(cfg, render_mode)

        self._image_buffer = torch.zeros((self.num_envs, self.cfg.visual.num_seq_actor, self.cfg.visual.h, self.cfg.visual.w), device=self.device)
        self._depth_buffer = torch.zeros((self.num_envs, self.cfg.visual.num_seq_critic, self.cfg.visual.h, self.cfg.visual.w), device=self.device)

        self._sph_theta   = torch.zeros(self.num_envs, device=self.device)
        self._sph_phi     = torch.zeros(self.num_envs, device=self.device)
        self._sph_psi     = torch.zeros(self.num_envs, device=self.device)
        self._light_level = torch.full((self.num_envs,), cfg.light_level_init,
                                       dtype=torch.long, device=self.device)
        
        Nx, Ny, Nz          = self.cfg.tsdf.vol_dim
        self._tsdf_vol      = torch.zeros(self.num_envs, Nx, Ny, Nz, device=self.device)
        self._weight_vol    = torch.zeros(self.num_envs, Nx, Ny, Nz, device=self.device)
        self._vol_origin    = torch.zeros(self.num_envs, 3,          device=self.device)
        self._total_surf_voxels = torch.ones(self.num_envs,          device=self.device)

        
        self._prev_coverage = torch.zeros(self.num_envs, device=self.device)
        self._prev_contrast = torch.zeros(self.num_envs, device=self.device)
        self.curr_coverage  = torch.zeros(self.num_envs, device=self.device)
        self.curr_contrast  = torch.zeros(self.num_envs, device=self.device)

        # RigidObject 핸들 (InteractiveScene 이 자동 생성)
        self._sensor_rig    = self.scene["sensor_rig"]
        self._camera        = self.scene["camera"]

        rock_local      = torch.tensor([0.0, 0.0, -3.0], device=self.device)
        self.rock_pos   = self.scene.env_origins + rock_local  # (num_envs, 3)
        self._actions   = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        
        self._prev_cam_pos = torch.zeros(self.num_envs, 3, device=self.device) # this var doesn't use right now, but it will be in reward function.

        self._setup_vis_markers()        
        self._debug_save_dir = Path("./debug_obs")
        self._debug_save_dir.mkdir(parents=True, exist_ok=True)
        self._debug_save_every = 1 
        self._debug_frame_idx = 0
        
        self._debug_seq_dir = Path("./debug_seq")
        self._debug_seq_dir.mkdir(parents=True, exist_ok=True)
        self._debug_seq_every = 6
        self._debug_seq_step = 0

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

            for ax, col in zip(AXES, AXIS_COLS):
                # 카메라 축
                aw = quat_apply(cq, ax)[0].cpu()
                starts.append(cp.tolist())
                ends.append((cp + aw * L).tolist())
                colors.append(col); widths.append(W)

        if starts:
            self._draw.draw_lines(starts, ends, colors, widths)

    # ── 씬 구성 ──────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        stage = omni.usd.get_context().get_stage()

        for env_idx in range(self.num_envs):
            env_ns = f"/World/envs/env_{env_idx}"
            for idx in range(2):
                self._add_light_children(stage, f"{env_ns}/SensorRig/SphereLight_{idx}")

    def _add_light_children(self, stage, light_prim_path: str) -> None:
        # ── SphereLight + ShapingAPI ──────────────────────────────────────
        light = stage.GetPrimAtPath(light_prim_path)

        shaping = UsdLux.ShapingAPI.Apply(light)
        shaping.GetShapingConeAngleAttr().Set(40.0)   # 반각 [도]
        shaping.GetShapingConeSoftnessAttr().Set(0.1)

    def _pre_physics_step(self, action: torch.Tensor) -> None: # 실제로 쓰이는 함수인가???
        """action을 저장한다. 실제 적용은 _apply_action()에서 수행."""
        self._actions = action.clone()

    def _apply_action(self) -> None:
        pose_idx = self._actions[:, 0:6].argmax(dim=-1) 
        light_idx= self._actions[:, 6:9].argmax(dim=-1)

        pose_active = self._actions[:, 0:6].any(dim=-1)

        deltas = torch.tensor([self.cfg.delta_theta, self.cfg.delta_phi, self.cfg.delta_psi],
                              device = self.device)
        axis = pose_idx //2
        sign = torch.where(pose_idx %2 ==0,
                           torch.ones_like(pose_idx, dtype=torch.float),
                           torch.full_like(pose_idx, -1.0, dtype=torch.float)) # (num_envs,) 짝수 = +1, 홀수 = -1
        delta_mat = torch.zeros(self.num_envs, 3, device=self.device)
        
        delta_mat.scatter_(1,
                           axis.unsqueeze(1),
                           (sign*deltas[axis]).unsqueeze(1))
        # delta_mat *= pose_active.float().unsequeeze(1)

        self._sph_theta += delta_mat[:, 0]
        self._sph_phi   += delta_mat[:, 1]
        self._sph_psi   += delta_mat[:, 2]

        self._sph_theta = self._sph_theta % (2*math.pi)
        self._sph_phi.clamp_(self.cfg.phi_min, self.cfg.phi_max)
        self._sph_psi.clamp_(self.cfg.psi_min, self.cfg.psi_max)

        prev_light_level = self._light_level.clone()
        delta_light = light_idx.long() - 1
        self._light_level = (prev_light_level + delta_light).clamp(1, 8)

        # sphere -> cartesian coordinate 변환
        curr_theta = self._sph_theta
        curr_phi   = self._sph_phi
        curr_psi   = self._sph_psi

        offset = torch.stack([
            curr_psi * torch.sin(curr_phi) * torch.cos(curr_theta),
            curr_psi * torch.sin(curr_phi) * torch.sin(curr_theta),
            curr_psi * torch.cos(curr_phi),
        ], dim=-1)
        
        cam_pos_new = self.rock_pos + offset
        cam_quat_new = self._look_at_quat(cam_pos_new, self.rock_pos)
                                                                                                                        
        # sensor_rig 상태 세트                                                                                         
        state = torch.zeros(self.num_envs, 13, device=self.device)                                                     
        state[:, 0:3] = cam_pos_new                                                                                    
        state[:, 3:7] = cam_quat_new
        self._sensor_rig.write_root_state_to_sim(state)
                                                                                                                        
        # 조명 intensity 업데이트
        self._update_light_intensity(self._light_level) 

    def _update_light_intensity(self, next_light_level:torch.Tensor)->None:
        print(f"Ordered light level: {next_light_level}")
        stage = omni.usd.get_context().get_stage()
        for env_idx in range(self.num_envs):
            intensity = float(next_light_level[env_idx].item() * self.cfg.light_intensity_per_level)
            env_ns = f"/World/envs/env_{env_idx}"
            for i in range(2):
                path = f"{env_ns}/SensorRig/SphereLight_{i}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    continue

                attr = prim.GetAttribute("inputs:intensity")
                if attr.IsValid():
                    attr.Set(intensity)

    # ── 상태 프로퍼티 ─────────────────────────────────────────────────────────

    @property
    def cam_pos(self) -> torch.Tensor:
        """카메라 리그 월드 위치 (num_envs, 3)."""
        return self._sensor_rig.data.root_pos_w

    @property
    def cam_orient(self) -> torch.Tensor:
        """카메라 리그 월드 자세 쿼터니언 [w,x,y,z] (num_envs, 4)."""
        return self._sensor_rig.data.root_quat_w        

    # ── 관측 ─────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        # image buffer updating - need to be implemented
        raw_rgb     = self._camera.data.output["uw_rgb"][:, :, :, :3]          # (num_envs, h, w, 3)
        # raw_rgb     = self._camera.data.output["rgba"][:, :, :, :3]          # (num_envs, h, w, 3)
        raw_depth   = self._camera.data.output["distance_to_camera"]   # (num_envs, h, w) ?? sure?

        curr_obs    = torch.mean(raw_rgb.float(), dim=-1)/255.0 # 3 color scale -> grayscale & pixel value normalization
        curr_state  = raw_depth.squeeze(-1)

        curr_obs = F.interpolate(
            curr_obs.unsqueeze(1),
            size = (self.cfg.visual.h, self.cfg.visual.w),
            mode = "bilinear", align_corners=False
        ).squeeze(1)

        curr_state = F.interpolate(
            curr_state.unsqueeze(1),
            size = (self.cfg.visual.h, self.cfg.visual.w),
            mode = "nearest"
        ).squeeze(1)

        self._image_buffer = torch.roll(self._image_buffer, shifts=-1, dims=1)
        self._image_buffer[:, -1, :, :] = curr_obs

        self._depth_buffer = torch.roll(self._depth_buffer, shifts=-1, dims=1)
        self._depth_buffer[:, -1, :, :] = curr_state # depth map은 (num_envs, h, w)아닌지?

        curr_contrast = self._compute_patch_contrast(curr_obs)
        self.curr_contrast = curr_contrast

        # number data vector (normalizaiton needed)
        scalar_obs = torch.stack([
            self._sph_theta / (2*math.pi),
            (self._sph_phi - self.cfg.phi_min) / (self.cfg.phi_max - self.cfg.phi_min),
            (self._sph_psi - self.cfg.psi_min) / (self.cfg.psi_max - self.cfg.psi_min),
            curr_contrast,
            (self._light_level.float()-1.0)/7.0,
        ], dim=-1)

        # 방향 마커 실시간 업데이트 (PhysX runtime 데이터 사용)
        if self.cfg.debug_vis:
            self._update_vis_markers()

        # self._save_debug_obs(raw_rgb, raw_depth, curr_obs, curr_state)
        # self._save_debug_sequence()

        return {
            "policy": self._image_buffer,     # Actor: 광학 이미지 시퀀스 (6, 84, 84)
            "extra_info": scalar_obs,         # Actor: 5개 수치 데이터
            "critic": self._depth_buffer      # Critic: GT Depth 시퀀스 (Privileged)
        }

    
    # ── 보상 ─────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        # self._voxelize_gt_mesh(list(range(self.num_envs)))
        # print(f"[S1] total_surf_voxels : {self._total_surf_voxels}")                    
        # print(f"[S1] vol_origin        : {self._vol_origin}")                           
        # print(f"[S1] rock_pos          : {self.rock_pos}")      
        # curr_coverage = self._compute_curr_coverage()
        # self.curr_coverage = curr_coverage

        # delta_coverage = curr_coverage - self._prev_coverage
        # reward_coverage = self.cfg.k_c * delta_coverage

        # terminal_mask = (curr_coverage >= self.cfg.coverage_terminal)
        # reward_coverage[terminal_mask] += 100.0 #?????

        # delta_contrast = self.curr_contrast - self._prev_contrast
        # reward_quality = self.cfg.lambda_q * delta_contrast

        # dist_moved = torch.norm(self.cam_pos - self._prev_cam_pos, dim=-1)
        # reward_penalty = self.cfg.k_x * dist_moved + self.cfg.c_step

        # self._prev_coverage = curr_coverage.clone()
        # self._prev_contrast  = self.curr_contrast.clone()
        # self._prev_cam_pos  = self.cam_pos.clone()

        # print(f"reward : {reward_coverage + reward_quality - reward_penalty}")
        retval = torch.full((self.num_envs,),10.0, device = self.device)
        # return reward_coverage + reward_quality - reward_penalty
        return retval
    
    # ── 종료 조건 ─────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        goal_reached   = self.curr_coverage >= self.cfg.coverage_terminal

        dist_cam       = torch.norm(self.cam_pos - self.rock_pos, dim=-1)
        out_of_bounds  = dist_cam > self.cfg.psi_max

        terminated     = goal_reached | out_of_bounds
        truncated      = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ── 리셋 ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)

        cfg = self.cfg
        n   = len(env_ids)
        env_ids_t = torch.tensor(env_ids, device=self.device)

        # self._sph_theta[env_ids]    = torch.rand(n, device=self.device) * 2.0 * math.pi
        # self._sph_phi[env_ids]      = torch.rand(n) * (cfg.phi_max - cfg.phi_min) + cfg.phi_min
        # self._sph_psi[env_ids]      = torch.rand(n, device=self.device) * (cfg.psi_max - cfg.psi_min) + cfg.psi_min
        self._sph_theta[env_ids]    = 0.0
        self._sph_phi[env_ids]      = math.radians(89.0)
        self._sph_psi[env_ids]      = 1.0
        
        self._light_level[env_ids]  = cfg.light_level_init

        offset = torch.stack([
            self._sph_psi[env_ids] * torch.sin(self._sph_phi[env_ids]) * torch.cos(self._sph_theta[env_ids]),
            self._sph_psi[env_ids] * torch.sin(self._sph_phi[env_ids]) * torch.sin(self._sph_theta[env_ids]),
            self._sph_psi[env_ids] * torch.cos(self._sph_phi[env_ids]),
        ], dim=-1)

        cam_pos_new = self.rock_pos[env_ids] + offset
        cam_quat_new = self._look_at_quat(cam_pos_new, self.rock_pos[env_ids])
        # cam_quat_new = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(n, -1)

        rig_state = torch.zeros(n, 13, device = self.device) # where does "13" came from??? <- 'isaacLab root_state form' hmm..
        rig_state[:, 0:3] = cam_pos_new
        rig_state[:, 3:7] = cam_quat_new
        self._sensor_rig.write_root_state_to_sim(rig_state, env_ids=env_ids_t)

        # fill buffer frames with 1st frame in k+1 times, but how can it accomplished with 0.0 ??? is this implement not conflict with _get_observation()?
        sim = sim_utils.SimulationContext.instance()

        for _ in range(10):
            sim.render()

        raw_rgb = self._camera.data.output["uw_rgb"][env_ids, :, :, :3]
        # raw_rgb = self._camera.data.output["rgba"][env_ids, :, :, :3]
        current_obs = torch.mean(raw_rgb.float(), dim=-1) / 255.0
        current_depth = self._camera.data.output["distance_to_camera"][env_ids].squeeze(-1)

        current_obs = F.interpolate(
            current_obs.unsqueeze(1),
            size=(self.cfg.visual.h, self.cfg.visual.w),
            mode="bilinear", align_corners=False
        ).squeeze(1)
        current_depth = F.interpolate(
            current_depth.unsqueeze(1),
            size=(self.cfg.visual.h, self.cfg.visual.w),
            mode="nearest"
        ).squeeze(1)

        # 2. k+1 채널에 현재 프레임을 반복해서 채우기
        self._image_buffer[env_ids] = current_obs.unsqueeze(1).expand(-1, self.cfg.visual.num_seq_actor, -1, -1)
        self._depth_buffer[env_ids] = current_depth.unsqueeze(1).expand(-1, self.cfg.visual.num_seq_critic, -1, -1)

        self._prev_coverage[env_ids] = 0.0
        self._prev_contrast[env_ids] = 0.0
        self._prev_cam_pos[env_ids]  = cam_pos_new

        if cfg.water_dr_enabled:
            self._randomize_water_params()

        self._voxelize_gt_mesh(env_ids)

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
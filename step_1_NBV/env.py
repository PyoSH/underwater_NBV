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
        self._sensor_rig   = self.scene["sensor_rig"]
        self._camera = self.scene["camera"]

        rock_local = torch.tensor([0.0, 0.0, -3.0], device=self.device)
        self.rock_pos = self.scene.env_origins + rock_local  # (num_envs, 3)
        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        
        self._prev_cam_pos = torch.zeros(self.num_envs, 3, device=self.device) # this var doesn't use right now, but it will be in reward function.

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

    def _pre_physics_step(self, action: torch.Tensor) -> None:
        """action을 저장한다. 실제 적용은 _apply_action()에서 수행."""
        self._actions = action.clone()

    def _apply_action(self) -> None:
        pose_idx = self._actions[:, 0:6].argmax(dim=-1) 
        light_idx= self._actions[:, 6:9].argmax(dim=-1) 

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
    
    def _compute_patch_contrast(self, img: torch.Tensor)->torch.Tensor:
        patches     = img.unfold(1, 14, 14).unfold(2, 14, 14) # what tensor shape is this?
        patch_std   = torch.std(patches, dim=(-1, -2))
        
        return torch.mean(patch_std, dim=(1,2))
    
    def _voxelize_gt_mesh(self, env_ids: Sequence[int]) -> None:
        vox        = self.cfg.tsdf.voxel_size
        Nx, Ny, Nz = self.cfg.tsdf.vol_dim

        for env_id in env_ids:
            verts, faces = self._load_mesh(env_id)             # world-space verts

            r1  = np.random.rand(len(faces), 1).astype(np.float32)
            r2  = np.random.rand(len(faces), 1).astype(np.float32)
            a   = 1.0 - np.sqrt(r1)
            b   = np.sqrt(r1) * (1.0 - r2)
            c   = np.sqrt(r1) * r2

            v0  = verts[faces[:, 0]]
            v1  = verts[faces[:, 1]]
            v2  = verts[faces[:, 2]]
            pts = a * v0 + b * v1 + c * v2                     # (F, 3)

            obj_min  = pts.min(axis=0)
            obj_max  = pts.max(axis=0)
            center   = (obj_min + obj_max) / 2.0
            half_ext = np.array([Nx, Ny, Nz], dtype=np.float32) * vox / 2.0
            origin   = center - half_ext

            self._vol_origin[env_id] = torch.tensor(origin, device=self.device)

            pts_t     = torch.tensor(pts, device=self.device)
            orig_t    = self._vol_origin[env_id]
            idx       = ((pts_t - orig_t) / vox).long()

            in_bounds = (
                (idx[:, 0] >= 0) & (idx[:, 0] < Nx) &
                (idx[:, 1] >= 0) & (idx[:, 1] < Ny) &
                (idx[:, 2] >= 0) & (idx[:, 2] < Nz)
            )
            idx = idx[in_bounds]

            surf_vol = torch.zeros(Nx, Ny, Nz, dtype=torch.bool, device=self.device)
            surf_vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True

            self._total_surf_voxels[env_id] = surf_vol.sum().float().clamp(min=1.0)
            self._tsdf_vol  [env_id]        = torch.zeros(Nx, Ny, Nz, device=self.device)
            self._weight_vol[env_id]        = torch.zeros(Nx, Ny, Nz, device=self.device)


    def _load_mesh(self, env_id: int):
        from pxr import Usd, UsdGeom, Gf

        stage     = omni.usd.get_context().get_stage()
        prim_path = f"/World/envs/env_{env_id}/Object"
        root_prim = stage.GetPrimAtPath(prim_path)

        mesh_prim = None
        for prim in Usd.PrimRange(root_prim):                  # full subtree
            if prim.IsA(UsdGeom.Mesh):
                mesh_prim = UsdGeom.Mesh(prim)
                break

        if mesh_prim is None:
            raise RuntimeError(f"No UsdGeom.Mesh found under: {prim_path}")

        points  = mesh_prim.GetPointsAttr().Get()
        verts   = np.array(points, dtype=np.float32)

        indices = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        counts  = np.array(mesh_prim.GetFaceVertexCountsAttr().Get(),  dtype=np.int64)
        faces   = self._triangulate(indices, counts)

        # Local → world space
        xform_cache = UsdGeom.XformCache()
        world_xform = xform_cache.GetLocalToWorldTransform(mesh_prim.GetPrim())
        ones    = np.ones((len(verts), 1), dtype=np.float32)
        verts_h = np.hstack([verts, ones])
        mat     = np.array(world_xform).reshape(4, 4).T.astype(np.float32)
        verts   = (verts_h @ mat.T)[:, :3]

        # Unit conversion (cm → m etc.)
        stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
        verts     = verts * float(stage_mpu)

        return verts, faces

    def _triangulate(self, indices: np.ndarray, counts: np.ndarray) -> np.ndarray:
        triangles = []
        offset    = 0
        for n in counts:
            v0 = indices[offset]
            for j in range(1, n - 1):
                triangles.append([v0, indices[offset + j], indices[offset + j + 1]])
            offset += n
        return np.array(triangles, dtype=np.int64)
    
    def _integrate_depth(self) -> None:
        """
        Fuses current depth maps from all envs into the batched TSDF volume.
        Fully vectorized — no Python loops over envs or voxels.
        
        Shapes:
            vox_world:  (num_envs, Nx*Ny*Nz, 3)
            vox_cam:    (num_envs, Nx*Ny*Nz, 3)
            proj_u/v:   (num_envs, Nx*Ny*Nz)
            sdf:        (num_envs, Nx*Ny*Nz)
        """
        cfg        = self.cfg.tsdf
        vox        = cfg.voxel_size
        trunc      = cfg.trunc_margin
        Nx, Ny, Nz = cfg.vol_dim
        N_vox      = Nx * Ny * Nz
        E          = self.num_envs

        K = self._camera.data.intrinsic_matrices
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]

        # ── 1. Build voxel center grid (shared across envs) ───────────────────
        # Do this once and cache — grid doesn't change between steps
        if not hasattr(self, '_vox_local'):
            xi = torch.arange(Nx, device=self.device)
            yi = torch.arange(Ny, device=self.device)
            zi = torch.arange(Nz, device=self.device)

            # (Nx, Ny, Nz, 3) voxel centers in local grid coords (origin = 0)
            gx, gy, gz = torch.meshgrid(xi, yi, zi, indexing='ij')
            self._vox_local = torch.stack([
                gx.flatten().float() * vox + vox / 2.0,
                gy.flatten().float() * vox + vox / 2.0,
                gz.flatten().float() * vox + vox / 2.0,
            ], dim=-1)                                         # (N_vox, 3)

        # ── 2. Shift local grid to world coords per env ────────────────────────
        # _vol_origin: (E, 3),  _vox_local: (N_vox, 3)
        vox_world = self._vox_local.unsqueeze(0) + \
                    self._vol_origin.unsqueeze(1)              # (E, N_vox, 3)

        # ── 3. Transform world → camera space ─────────────────────────────────
        cam_pose = self._build_cam_pose()                      # (E, 4, 4)
        R = cam_pose[:, :3, :3]                                # (E, 3, 3)
        t = cam_pose[:, :3,  3]                                # (E, 3)

        # v_cam = R @ v_world + t
        # bmm expects (E, 3, 3) @ (E, 3, N_vox) → (E, 3, N_vox)
        vox_cam = torch.bmm(R, vox_world.permute(0, 2, 1))    # (E, 3, N_vox)
        vox_cam = vox_cam + t.unsqueeze(-1)                    # (E, 3, N_vox)
        vox_cam = vox_cam.permute(0, 2, 1)                     # (E, N_vox, 3)

        vox_z = vox_cam[..., 2]                                # (E, N_vox)
        vox_x = vox_cam[..., 0]
        vox_y = vox_cam[..., 1]

        # ── 4. Project to pixel coordinates ───────────────────────────────────
        valid_z = vox_z > 1e-4                                 # in front of camera

        proj_u = (fx * vox_x / vox_z.clamp(min=1e-4) + cx)    # (E, N_vox)
        proj_v = (fy * vox_y / vox_z.clamp(min=1e-4) + cy)    # (E, N_vox)

        H = self._camera.data.output["distance_to_camera"].shape[1]
        W = self._camera.data.output["distance_to_camera"].shape[2]

        proj_u_int = proj_u.long()
        proj_v_int = proj_v.long()

        in_bounds = (
            valid_z                        &
            (proj_u_int >= 0)              &
            (proj_u_int <  W)              &
            (proj_v_int >= 0)              &
            (proj_v_int <  H)
        )                                                      # (E, N_vox) bool

        # ── 5. Sample depth image at projected pixels ──────────────────────────
        depth_img = self._camera.data.output["distance_to_camera"]
        if depth_img.dim() == 4:
            depth_img = depth_img.squeeze(-1)                      # (E, H, W)
        H, W      = depth_img.shape[1], depth_img.shape[2]        # move here, after squeeze
        depth_flat = depth_img.reshape(E, -1)

        # Clamp indices for safe gather (out-of-bounds handled by mask)
        safe_u = proj_u_int.clamp(0, W - 1)
        safe_v = proj_v_int.clamp(0, H - 1)
        pixel_idx = safe_v * W + safe_u                        # (E, N_vox)

        sampled_depth = torch.gather(depth_flat, 1, pixel_idx) # (E, N_vox)

        # ── 6. Compute SDF and truncate ────────────────────────────────────────
        sdf  = sampled_depth - vox_z                           # (E, N_vox)
        tsdf = (sdf / trunc).clamp(-1.0, 1.0)                  # (E, N_vox)

        # Only update voxels that are:
        #  - projected inside image (in_bounds)
        #  - within truncation band (sdf >= -trunc)
        update_mask = in_bounds & (sdf >= -trunc)              # (E, N_vox)

        # ── 7. Running average TSDF update ────────────────────────────────────
        w_old = self._weight_vol.reshape(E, N_vox)             # (E, N_vox)
        t_old = self._tsdf_vol  .reshape(E, N_vox)             # (E, N_vox)

        w_new = w_old + update_mask.float()                    # (E, N_vox)
        # avoid div/0 where update_mask is False (w_new == w_old there)
        t_new = torch.where(
            update_mask,
            (t_old * w_old + tsdf) / w_new.clamp(min=1e-8),
            t_old
        )                                                      # (E, N_vox)

        self._tsdf_vol   = t_new.reshape(E, Nx, Ny, Nz)
        self._weight_vol = w_new.reshape(E, Nx, Ny, Nz)
    
    def _compute_curr_coverage(self) -> torch.Tensor:
        """
        Computes coverage rate per env from the current TSDF volume.
        
        A voxel counts as 'observed surface' when:
        - weight > 0  : seen by at least one depth frame
        - |tsdf| < 1.0: near a surface (not free space or behind surface)

        Returns: coverage (num_envs,) float32, range [0, 1]
        """
        observed = (
            (self._weight_vol > 0) &
            (self._tsdf_vol.abs() < 1.0)
        )                                                      # (E, Nx, Ny, Nz) bool

        count    = observed.sum(dim=(1, 2, 3)).float()         # (E,)
        coverage = count / self._total_surf_voxels             # (E,)  normalized

        return coverage.clamp(0.0, 1.0)

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

        return {
            "policy": self._image_buffer,     # Actor: 광학 이미지 시퀀스 (6, 84, 84)
            "extra_info": scalar_obs,         # Actor: 5개 수치 데이터
            "critic": self._depth_buffer      # Critic: GT Depth 시퀀스 (Privileged)
        }
    
    # ── 보상 ─────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        self._integrate_depth()
        curr_coverage = self._compute_curr_coverage()
        self.curr_coverage = curr_coverage

        delta_coverage = curr_coverage - self._prev_coverage
        reward_coverage = self.cfg.k_c * delta_coverage

        terminal_mask = (curr_coverage >= self.cfg.coverage_terminal)
        reward_coverage[terminal_mask] += 100.0 #?????

        delta_contrast = self.curr_contrast - self._prev_contrast
        reward_quality = self.cfg.lambda_q * delta_contrast

        dist_moved = torch.norm(self.cam_pos - self._prev_cam_pos, dim=-1)
        reward_penalty = self.cfg.k_x * dist_moved + self.cfg.c_step

        self._prev_coverage = curr_coverage.clone()
        self._prev_contrast  = self.curr_contrast.clone()
        self._prev_cam_pos  = self.cam_pos.clone()

        return reward_coverage + reward_quality - reward_penalty
    
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
        # forward = -offset / (offset.norm(dim=-1, keepdim=True) + 1e-8)
        # cam_quat_new = self._forward_to_quat(forward)
        cam_quat_new = self._look_at_quat(cam_pos_new, self.rock_pos)
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

    def _forward_to_quat(self, forward: torch.Tensor) -> torch.Tensor:
        N = forward.shape[0]

        up = torch.tensor([[0., 0., 1.]], device=self.device).expand(N, -1)
        dot = (forward * up).sum(dim=-1, keepdim=True).abs()
        fallback = torch.tensor([[0., 1., 0.]], device=self.device).expand(N, -1)
        up = torch.where(dot > 1.0 - 1e-6, fallback, up)

        # 수정: up × forward 순서
        right    = torch.linalg.cross(forward, up)
        right    = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
        up_ortho = torch.linalg.cross(forward, right)

        # 수정: Isaac body frame 기준 열 배치 (X=forward, Y=left, Z=up)
        R = torch.stack([forward, -right, up_ortho], dim=-1)

        # 쿼터니언 변환은 동일
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        s = 0.5 / torch.sqrt((trace + 1.0).clamp(min=1e-8))
        w = 0.25 / s
        x = (R[:, 2, 1] - R[:, 1, 2]) * s
        y = (R[:, 0, 2] - R[:, 2, 0]) * s
        z = (R[:, 1, 0] - R[:, 0, 1]) * s

        quat = torch.stack([w, x, y, z], dim=-1)
        print(f"1. forward vector : {forward}")
        print(f"2. forward quat   : {quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)}")
        return quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    
    def _look_at_quat(self, from_pos: torch.Tensor, to_pos: torch.Tensor) -> torch.Tensor:
        """
        from_pos (N,3) → to_pos (N,3) 를 바라보는 쿼터니언 [w,x,y,z] (N,4) 반환.
        리그 body frame: +X = forward, +Y = left, +Z = up.
 
        Shepperd method 4분기 완전 구현으로 수치 안정성 확보.
        """
        N = from_pos.shape[0]
 
        # ── forward 벡터 ───────────────────────────────────────────────────
        forward = to_pos - from_pos
        forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-8)
 
        # ── up 기준벡터 및 gimbal lock fallback ────────────────────────────
        up = torch.tensor([[0., 0., 1.]], device=self.device).expand(N, -1)
        dot = (forward * up).sum(dim=-1, keepdim=True).abs()
        fallback = torch.tensor([[0., 1., 0.]], device=self.device).expand(N, -1)
        up = torch.where(dot > 1.0 - 1e-6, fallback, up)
 
        # ── 직교 기저 구성 (X=forward, Y=left, Z=up) ──────────────────────
        right    = torch.linalg.cross(forward, up)
        right    = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
        up_ortho = torch.linalg.cross(right, forward)
        up_ortho = up_ortho / (up_ortho.norm(dim=-1, keepdim=True) + 1e-8)
 
        # ── R 열 배치: col0=forward(+X), col1=-right(+Y=left), col2=up_ortho(+Z) ──
        R = torch.stack([forward, -right, up_ortho], dim=-1)  # (N, 3, 3)
 
        # ── Shepperd method 4분기 ──────────────────────────────────────────
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]          # (N,)
 
        w = torch.zeros(N, device=self.device)
        x = torch.zeros(N, device=self.device)
        y = torch.zeros(N, device=self.device)
        z = torch.zeros(N, device=self.device)
 
        # case 0: trace > 0
        m0 = trace > 0
        if m0.any():
            s     = 0.5 / torch.sqrt((trace[m0] + 1.0).clamp(min=1e-8))
            w[m0] = 0.25 / s
            x[m0] = (R[m0, 2, 1] - R[m0, 1, 2]) * s
            y[m0] = (R[m0, 0, 2] - R[m0, 2, 0]) * s
            z[m0] = (R[m0, 1, 0] - R[m0, 0, 1]) * s
 
        # case 1: R00 최대
        m1 = (~m0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        if m1.any():
            s     = 2.0 * torch.sqrt((1.0 + R[m1, 0, 0] - R[m1, 1, 1] - R[m1, 2, 2]).clamp(min=1e-8))
            w[m1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
            x[m1] = 0.25 * s
            y[m1] = (R[m1, 0, 1] + R[m1, 1, 0]) / s
            z[m1] = (R[m1, 0, 2] + R[m1, 2, 0]) / s
 
        # case 2: R11 최대
        m2 = (~m0) & (~m1) & (R[:, 1, 1] > R[:, 2, 2])
        if m2.any():
            s     = 2.0 * torch.sqrt((1.0 + R[m2, 1, 1] - R[m2, 0, 0] - R[m2, 2, 2]).clamp(min=1e-8))
            w[m2] = (R[m2, 0, 2] - R[m2, 2, 0]) / s
            x[m2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
            y[m2] = 0.25 * s
            z[m2] = (R[m2, 1, 2] + R[m2, 2, 1]) / s
 
        # case 3: R22 최대
        m3 = (~m0) & (~m1) & (~m2)
        if m3.any():
            s     = 2.0 * torch.sqrt((1.0 + R[m3, 2, 2] - R[m3, 0, 0] - R[m3, 1, 1]).clamp(min=1e-8))
            w[m3] = (R[m3, 1, 0] - R[m3, 0, 1]) / s
            x[m3] = (R[m3, 0, 2] + R[m3, 2, 0]) / s
            y[m3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s
            z[m3] = 0.25 * s
 
        quat = torch.stack([w, x, y, z], dim=-1)               # (N, 4)

        print(f"1. forward vector : {forward}")
        print(f"2. forward quat   : {quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)}")

        return quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    
    def _quat_to_rot_matrix(self, quat:torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        R = torch.stack([
            1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
                2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
                2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)                          # (num_envs, 3, 3)

        return R
    
    def _build_cam_pose(self) -> torch.Tensor:
        """
        Builds world-to-camera extrinsic matrix (E, 4, 4).

        Isaac gives us camera-in-world:
            R_wc (cam_orient): rotation from camera frame to world frame
            t_w  (cam_pos):    camera position in world

        We need world-to-camera:
            R_cw = R_wc^T          (transpose, since R is orthogonal)
            t_cw = -R_cw @ t_w    (re-express world origin in camera frame)
        """
        N    = self.num_envs
        R_wc = self._quat_to_rot_matrix(self.cam_orient)          # (E, 3, 3) cam→world
        R_cw = R_wc.transpose(1, 2)                               # (E, 3, 3) world→cam
        t_cw = -torch.bmm(R_cw, self.cam_pos.unsqueeze(-1)).squeeze(-1)  # (E, 3)

        pose = torch.eye(4, device=self.device).unsqueeze(0).expand(N, -1, -1).clone()
        pose[:, :3, :3] = R_cw
        pose[:, :3,  3] = t_cw

        return pose    
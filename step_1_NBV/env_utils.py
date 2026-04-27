from __future__ import annotations
import numpy as np
import torch
from typing import Sequence
import omni.usd
import cv2

class EnvUtilsMixin:

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

        # print(f"1. forward vector : {forward}")
        # print(f"2. forward quat   : {quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)}")

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
            self._surf_vol[env_id] = surf_vol

    def _load_mesh(self, env_id: int):
        from pxr import Usd, UsdGeom

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
    
    def _save_debug_sequence(self):
        # 너무 자주 저장하면 느리니 주기로 제어
        if self._debug_seq_step % self._debug_seq_every != 0:
            self._debug_seq_step += 1
            return

        step_dir = self._debug_seq_dir / f"step_{self._debug_seq_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        num_envs = self.num_envs
        K_img = self.cfg.visual.num_seq_actor
        K_dep = self.cfg.visual.num_seq_critic

        img_buf = self._image_buffer.detach().cpu().numpy()   # (E,K,H,W)
        dep_buf = self._depth_buffer.detach().cpu().numpy()   # (E,K,H,W)

        for env_id in range(num_envs):
            env_dir = step_dir / f"env_{env_id}"
            env_dir.mkdir(parents=True, exist_ok=True)

            # image 시퀀스 저장: time axis = 1
            for k in range(K_img):
                img = img_buf[env_id, k]  # (H,W), [0,1] 가정
                img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(env_dir / f"img_seq_{k:02d}.png"), img_u8)

            # depth 시퀀스 저장
            for k in range(K_dep):
                depth = dep_buf[env_id, k]      # (H,W), float
                depth_valid = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                dmin, dmax = depth_valid.min(), depth_valid.max()
                if dmax > dmin:
                    depth_vis = ((depth_valid - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
                else:
                    depth_vis = np.zeros_like(depth_valid, dtype=np.uint8)
                cv2.imwrite(str(env_dir / f"depth_seq_{k:02d}.png"), depth_vis)

        self._debug_seq_step += 1
    
    def _save_debug_obs(self, raw_rgb, raw_depth, curr_obs, curr_state):
        if self._debug_frame_idx % self._debug_save_every != 0:
            self._debug_frame_idx += 1
            return

        save_root = self._debug_save_dir / f"step_{self._debug_frame_idx:06d}"
        save_root.mkdir(parents=True, exist_ok=True)

        num_envs = raw_rgb.shape[0]

        for env_id in range(num_envs):
            env_dir = save_root / f"env_{env_id}"
            env_dir.mkdir(parents=True, exist_ok=True)

            rgb = raw_rgb[env_id].detach().cpu().numpy().astype(np.uint8)   # (H,W,3)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(env_dir / "raw_rgb.png"), rgb_bgr)

            depth = raw_depth[env_id].detach().cpu().numpy()
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]

            depth_valid = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            dmin, dmax = depth_valid.min(), depth_valid.max()
            if dmax > dmin:
                depth_vis = ((depth_valid - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_valid, dtype=np.uint8)
            cv2.imwrite(str(env_dir / "raw_depth.png"), depth_vis)

            obs_img = (curr_obs[env_id].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(env_dir / "policy_obs.png"), obs_img)

            critic_depth = curr_state[env_id].detach().cpu().numpy()
            critic_depth = np.nan_to_num(critic_depth, nan=0.0, posinf=0.0, neginf=0.0)
            cmin, cmax = critic_depth.min(), critic_depth.max()
            if cmax > cmin:
                critic_vis = ((critic_depth - cmin) / (cmax - cmin) * 255.0).astype(np.uint8)
            else:
                critic_vis = np.zeros_like(critic_depth, dtype=np.uint8)
            cv2.imwrite(str(env_dir / "critic_depth.png"), critic_vis)

        self._debug_frame_idx += 1
    
    def _save_surf_pc(self, env_id:int =0):
        vox = self.cfg.tsdf.voxel_size
        Nx, Ny, Nz = self.cfg.tsdf.vol_dim
        origin = self._vol_origin[env_id].cpu().numpy()

        idx = self._surf_vol[env_id].nonzero(as_tuple=False).cpu().numpy()
        pts = origin + (idx + 0.5) * vox

        path = f"./surf_cloud_env{env_id}.ply"
        with open(path, 'w') as f:                                                      
            f.write("ply\nformat ascii 1.0\n")                                          
            f.write(f"element vertex {len(pts)}\n")                                     
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")                                                     
            for p in pts:          
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")                          
        print(f"[VIZ] saved {len(pts)} points → {path}")  
    
    def _save_weight_pc(self, env_id: int = 0):
      vox    = self.cfg.tsdf.voxel_size                                               
      origin = self._vol_origin[env_id].cpu().numpy()                                 
                                                                                      
      # 관측된 voxel 인덱스 (weight > 0)                                              
      idx = (self._weight_vol[env_id] > 0).nonzero(as_tuple=False).cpu().numpy()                                                                             
      pts = origin + (idx + 0.5) * vox
                                                                                      
      path = f"./weight_cloud_env{env_id}.ply"                                        
      with open(path, 'w') as f:                                                      
          f.write("ply\nformat ascii 1.0\n")                                          
          f.write(f"element vertex {len(pts)}\n")
          f.write("property float x\nproperty float y\nproperty float z\n")           
          f.write("end_header\n")                                                     
          for p in pts:
              f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")                          
                                                                                      
      print(f"[VIZ] weight cloud saved: {len(pts)} voxels → {path}")    
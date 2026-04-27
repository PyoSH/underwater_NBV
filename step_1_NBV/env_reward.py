from __future__ import annotations
import torch

class EnvRewardMixin:
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

    def _compute_patch_contrast(self, img: torch.Tensor)->torch.Tensor:
        patches     = img.unfold(1, 14, 14).unfold(2, 14, 14) # what tensor shape is this?
        patch_std   = torch.std(patches, dim=(-1, -2))
        
        return torch.mean(patch_std, dim=(1,2))

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
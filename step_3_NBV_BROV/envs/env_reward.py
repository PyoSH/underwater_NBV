"""
step_1_NBV/env/env_reward.py를 무수정 이식 — TSDF 적분/coverage 계산은 카메라가
자유 부유(sensor_rig)든 로봇에 고정 부착이든 동일 로직(`self._camera.data.*`,
`self._build_cam_pose()`만 사용, 로봇 종류 무관). `_build_cam_pose()` 자체의
변경(`envs/env_utils.py` 참조)만으로 충분해 여기는 손댈 곳이 없다.
"""

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
        cfg = self.cfg.tsdf
        vox = cfg.voxel_size
        trunc = cfg.trunc_margin
        Nx, Ny, Nz = cfg.vol_dim
        N_vox = Nx * Ny * Nz
        E = self.num_envs

        K = self._camera.data.intrinsic_matrices
        fx = K[:, 0, 0].unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)

        if not hasattr(self, '_vox_local'):
            xi = torch.arange(Nx, device=self.device)
            yi = torch.arange(Ny, device=self.device)
            zi = torch.arange(Nz, device=self.device)

            gx, gy, gz = torch.meshgrid(xi, yi, zi, indexing='ij')
            self._vox_local = torch.stack([
                gx.flatten().float() * vox + vox / 2.0,
                gy.flatten().float() * vox + vox / 2.0,
                gz.flatten().float() * vox + vox / 2.0,
            ], dim=-1)   # (N_vox, 3)

        vox_world = self._vox_local.unsqueeze(0) + \
            self._vol_origin.unsqueeze(1)   # (E, N_vox, 3)

        cam_pose = self._build_cam_pose()   # (E, 4, 4)
        R = cam_pose[:, :3, :3]
        t = cam_pose[:, :3, 3]

        vox_cam = torch.bmm(R, vox_world.permute(0, 2, 1))
        vox_cam = vox_cam + t.unsqueeze(-1)
        vox_cam = vox_cam.permute(0, 2, 1)   # (E, N_vox, 3)

        vox_z = vox_cam[..., 2]
        vox_x = vox_cam[..., 0]
        vox_y = vox_cam[..., 1]

        valid_z = vox_z > 1e-4

        proj_u = (fx * vox_x / vox_z.clamp(min=1e-4) + cx)
        proj_v = (fy * vox_y / vox_z.clamp(min=1e-4) + cy)

        H = self._camera.data.output["distance_to_camera"].shape[1]
        W = self._camera.data.output["distance_to_camera"].shape[2]

        proj_u_int = proj_u.long()
        proj_v_int = proj_v.long()

        in_bounds = (
            valid_z &
            (proj_u_int >= 0) &
            (proj_u_int < W) &
            (proj_v_int >= 0) &
            (proj_v_int < H)
        )

        depth_img = self._camera.data.output["distance_to_camera"]
        if depth_img.dim() == 4:
            depth_img = depth_img.squeeze(-1)
        H, W = depth_img.shape[1], depth_img.shape[2]
        depth_flat = depth_img.reshape(E, -1)

        safe_u = proj_u_int.clamp(0, W - 1)
        safe_v = proj_v_int.clamp(0, H - 1)
        pixel_idx = safe_v * W + safe_u

        sampled_depth = torch.gather(depth_flat, 1, pixel_idx)

        sdf = sampled_depth - vox_z
        tsdf = (sdf / trunc).clamp(-1.0, 1.0)

        update_mask = in_bounds & (sdf >= -trunc) & (sdf <= trunc)

        w_old = self._weight_vol.reshape(E, N_vox)
        t_old = self._tsdf_vol.reshape(E, N_vox)

        w_new = w_old + update_mask.float()
        t_new = torch.where(
            update_mask,
            (t_old * w_old + tsdf) / w_new.clamp(min=1e-8),
            t_old
        )

        self._tsdf_vol = t_new.reshape(E, Nx, Ny, Nz)
        self._weight_vol = w_new.reshape(E, Nx, Ny, Nz)

    def _compute_patch_contrast(self, img: torch.Tensor) -> torch.Tensor:
        patches = img.unfold(1, 14, 14).unfold(2, 14, 14)
        patch_std = torch.std(patches, dim=(-1, -2))

        return torch.mean(patch_std, dim=(1, 2))

    def _compute_curr_coverage(self) -> torch.Tensor:
        observed = (self._weight_vol > 0) & self._surf_vol   # GT surface만 카운트
        count = observed.sum(dim=(1, 2, 3)).float()
        return (count / self._total_surf_voxels).clamp(0.0, 1.0)

    # ── Quality-weighted coverage (step_1 env_GenNBV_quality.py 이식) ─────────

    def _compute_quality(self) -> None:
        """관측된 voxel의 품질을 Beer-Lambert 감쇠로 갱신한다.

        `_integrate_depth()` 이후에 호출해야 한다(TSDF/weight가 최신이어야 함).

        `surface_mask = weight > 0` — TSDF 분류(`tsdf <= 0`)를 **조건에 넣지
        않는다**. step_1에서 이 조건을 넣었더니 GT surface voxel의 37%가
        "관측됐지만 TSDF는 free space로 분류"돼 품질 누적이 차단됐고,
        binary 0.857 vs quality 0.483이라는 괴리가 생겼다(step_1 CLAUDE.md §10,
        "해석 B"로 수정 완료). 여기서 재는 것은 재구성 확정도가 아니라
        **관측 품질**이므로 weight>0이면 누적하는 것이 맞다.

        누적은 합이 아니라 **max**다 — 같은 voxel을 반복 방문해도 품질이 무한히
        쌓이지 않고 "가장 가까이서 본 순간"만 남는다(step_1 2026-05-26 변경).
        """
        centers = (
            self._vol_origin[:, None, None, None, :]      # (E,1,1,1,3)
            + self._voxel_offset[None]                    # (1,Nx,Ny,Nz,3)
        )
        cam = self._camera_position_w()[:, None, None, None, :]
        dist = torch.norm(centers - cam, dim=-1)          # (E,Nx,Ny,Nz)

        mu = self._quality_mu.view(-1, 1, 1, 1)
        quality_new = torch.exp(-mu * dist)

        observed = self._weight_vol > 0
        self._quality_vol = torch.maximum(
            self._quality_vol, quality_new * observed.float()
        )

    def _compute_coverage_q(self) -> torch.Tensor:
        """GT surface voxel에 대한 품질 가중 coverage (raw, 상한 = Q_sat)."""
        count = (self._quality_vol * self._surf_vol.float()).sum(dim=(1, 2, 3))
        return count / self._total_surf_voxels

    def _coverage_for_reward(self) -> torch.Tensor:
        """보상·종료·커리큘럼이 공통으로 쓰는 coverage (항상 0~1 정규화).

        quality 모드에서는 `coverage_q / Q_sat`을 돌려준다 — 이 정규화 덕분에
        `coverage_terminal`이 step_1과 같은 "달성 가능 상한 대비 비율" 의미를
        유지하고, binary 기준으로 실측 보정해 둔 k_c/c_step/coverage_bonus가
        스케일 변경 없이 그대로 유효하다.
        """
        if not self.cfg.use_quality_coverage:
            return self.curr_coverage
        return (self.curr_coverage_q / self._quality_Q_sat).clamp(0.0, 1.0)

"""
ImagingSonar — Isaac Lab sensor implementation (multi-env batched).

Pipeline per update()
─────────────────────
  Isaac Lab Camera renders → distance_to_camera (N,H,W,1), normals (N,H,W,4),
                             semantic_segmentation (N,H,W,1)
       │
       ▼  _apply_sonar_pipeline()
  1. _unproject_to_pcl()     : (N,H,W,…) → pcl (N,H*W,3), normals (N,H*W,3), semantics (N,H*W)
  2. compute_intensity        : dim=(N, H*W)
  3. world2local              : dim=(N, H*W)
  4. bin_intensity            : dim=(N, H*W)  → bin_sum/count (N, R, A)
  5. average (optional)       : dim=(N, R, A)
  6. noise kernels            : dim=(N, R, A)
  7. make_sonar_map_*         : dim=(N, R, A)
  8. make_sonar_image         : dim=(N, R, A)

Output:
    sensor.data.output["sonar_map"]    wp.array  (N, R, A)  vec3 (x, y, intensity)
    sensor.data.output["sonar_image"]  torch.Tensor (N, R, A+1, 4)  uint8 RGBA
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
import omni.ui as ui

from isaaclab.sensors import Camera

from .ImagingSonar_kernels import (
    compute_intensity,
    world2local,
    bin_intensity,
    average,
    all_max,
    range_max,
    normal_2d,
    range_dependent_rayleigh_2d,
    make_sonar_map_all,
    make_sonar_map_range,
    make_sonar_image as _make_sonar_image_kernel,
)

if TYPE_CHECKING:
    from .ImagingSonarCfg import ImagingSonarCfg


class ImagingSonar(Camera):
    cfg: ImagingSonarCfg

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, cfg: ImagingSonarCfg) -> None:
        super().__init__(cfg)

    def _initialize_impl(self) -> None:
        super()._initialize_impl()

        self._device = wp.get_preferred_device()
        N = self._num_envs  # number of parallel environments

        # ── Polar meshgrid: shared across all envs ─────────────────────
        self.min_azi = float(np.deg2rad(90.0 - self.cfg.hori_fov / 2.0))
        r_np = np.arange(self.cfg.min_range, self.cfg.max_range,
                         self.cfg.range_res, dtype=np.float32)
        a_np = np.arange(
            np.deg2rad(90.0 - self.cfg.hori_fov / 2.0),
            np.deg2rad(90.0 + self.cfg.hori_fov / 2.0),
            np.deg2rad(self.cfg.angular_res),
            dtype=np.float32,
        )
        r_grid, azi_grid = np.meshgrid(r_np, a_np, indexing="ij")  # (R, A)
        self._r   = wp.array(r_grid,   dtype=wp.float32, device=self._device)
        self._azi = wp.array(azi_grid, dtype=wp.float32, device=self._device)

        R_bins, A_bins = self._r.shape
        HW = self.cfg.hori_res * int(self.cfg.hori_res / (self.cfg.hori_fov / self.cfg.vert_fov))

        # ── Persistent GPU buffers ─────────────────────────────────────
        self._bin_sum    = wp.zeros((N, R_bins, A_bins), dtype=wp.float32, device=self._device)
        self._bin_count  = wp.zeros((N, R_bins, A_bins), dtype=wp.int32,   device=self._device)
        self._binned_int = wp.zeros((N, R_bins, A_bins), dtype=wp.float32, device=self._device)
        self._sonar_map  = wp.zeros((N, R_bins, A_bins), dtype=wp.vec3,    device=self._device)
        self._sonar_img  = wp.zeros((N, R_bins, A_bins + 1, 4), dtype=wp.uint8, device=self._device)
        self._gau_noise  = wp.zeros((N, R_bins, A_bins), dtype=wp.float32, device=self._device)
        self._ray_noise  = wp.zeros((N, R_bins, A_bins), dtype=wp.float32, device=self._device)

        self._frame_id: int = 0

        # Pre-register output keys
        self.data.output["sonar_map"]   = None
        self.data.output["sonar_image"] = None

        # Viewport
        self._sonar_provider = None
        if self.cfg.enable_viewport:
            self._make_viewport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt: float, force_recompute: bool = False) -> None:
        super().update(dt, force_recompute=force_recompute)
        self._apply_sonar_pipeline()

    def set_sonar_params(self, **kwargs) -> None:
        """Runtime override of any ImagingSonarCfg processing parameter."""
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _apply_sonar_pipeline(self) -> None:

        # ── 1. Fetch Isaac Lab Camera outputs ──────────────────────────
        depth_t    = self.data.output.get("distance_to_camera")  # (N,H,W,1)|(N,H,W)
        normals_t  = self.data.output.get("normals")             # (N,H,W,4)
        sem_t      = self.data.output.get("semantic_segmentation")  # (N,H,W,1)|(N,H,W)

        if depth_t is None or normals_t is None or sem_t is None:
            return

        # ── 2. Unproject to batched flat pointcloud ────────────────────
        pcl_np, nrm_np, sem_np, view_np = self._unproject_to_pcl(
            depth_t, normals_t, sem_t
        )
        # pcl_np  : (N, H*W, 3)  float32
        # nrm_np  : (N, H*W, 3)  float32
        # sem_np  : (N, H*W)     uint32
        # view_np : (N, 4, 4)    float32

        N, HW, _ = pcl_np.shape

        # ── 3. Build per-env reflectivity LUT ─────────────────────────
        indexToRefl_np = self._build_refl_lut(sem_np)  # (N, max_id+1)

        # ── 4. Upload to Warp ──────────────────────────────────────────
        pcl_wp      = wp.array(pcl_np,         ndim=3, dtype=wp.float32, device=self._device)
        nrm_wp      = wp.array(nrm_np,         ndim=3, dtype=wp.float32, device=self._device)
        sem_wp      = wp.array(sem_np,         ndim=2, dtype=wp.uint32,  device=self._device)
        lut_wp      = wp.array(indexToRefl_np, ndim=2, dtype=wp.float32, device=self._device)
        view_wp     = wp.array(view_np,        ndim=3, dtype=wp.float32, device=self._device)

        # ── 5. Per-point intensity   dim = (N, H*W) ───────────────────
        intensity_pt = wp.empty((N, HW), dtype=wp.float32, device=self._device)
        wp.launch(
            kernel=compute_intensity,
            dim=(N, HW),
            inputs=[pcl_wp, nrm_wp, view_wp, sem_wp, lut_wp, self.cfg.attenuation],
            outputs=[intensity_pt],
            device=self._device,
        )

        # ── 6. World → local → spherical   dim = (N, H*W) ────────────
        pcl_local_wp = wp.empty((N, HW), dtype=wp.vec3, device=self._device)
        pcl_spher_wp = wp.empty((N, HW), dtype=wp.vec3, device=self._device)
        wp.launch(
            kernel=world2local,
            dim=(N, HW),
            inputs=[view_wp, pcl_wp],
            outputs=[pcl_local_wp, pcl_spher_wp],
            device=self._device,
        )

        # ── 7. Bin   dim = (N, H*W) ────────────────────────────────────
        self._bin_sum.zero_()
        self._bin_count.zero_()
        self._binned_int.zero_()

        wp.launch(
            kernel=bin_intensity,
            dim=(N, HW),
            inputs=[
                pcl_spher_wp, intensity_pt,
                wp.float32(self.cfg.min_range),
                wp.float32(self.min_azi),
                wp.float32(self.cfg.range_res),
                wp.float32(float(np.deg2rad(self.cfg.angular_res))),
            ],
            outputs=[self._bin_sum, self._bin_count],
            device=self._device,
        )

        # ── 8. Binning method   dim = (N, R, A) ───────────────────────
        bin_shape = self._bin_sum.shape  # (N, R, A)
        if self.cfg.binning_method == "mean":
            wp.launch(
                kernel=average,
                dim=bin_shape,
                inputs=[self._bin_sum, self._bin_count],
                outputs=[self._binned_int],
                device=self._device,
            )
        else:  # "sum"
            self._binned_int = self._bin_sum

        # ── 9. Noise   dim = (N, R, A) ────────────────────────────────
        self._gau_noise.zero_()
        self._ray_noise.zero_()
        self._sonar_map.zero_()

        wp.launch(
            kernel=normal_2d,
            dim=bin_shape,
            inputs=[self._frame_id, 0.0, self.cfg.gau_noise_param],
            outputs=[self._gau_noise],
            device=self._device,
        )
        wp.launch(
            kernel=range_dependent_rayleigh_2d,
            dim=bin_shape,
            inputs=[
                self._frame_id,
                self._r, self._azi,
                self.cfg.max_range,
                self.cfg.ray_noise_param,
                self.cfg.central_peak,
                self.cfg.central_std,
            ],
            outputs=[self._ray_noise],
            device=self._device,
        )

        # ── 10. Normalise + composit   dim = (N, R, A) ────────────────
        offset_f = wp.float32(self.cfg.intensity_offset)
        gain_f   = wp.float32(self.cfg.intensity_gain)

        if self.cfg.normalizing_method == "all":
            maximum = wp.zeros((N,), dtype=wp.float32, device=self._device)
            wp.launch(
                kernel=all_max,
                dim=bin_shape,
                inputs=[self._binned_int],
                outputs=[maximum],
                device=self._device,
            )
            wp.launch(
                kernel=make_sonar_map_all,
                dim=bin_shape,
                inputs=[self._r, self._azi, self._binned_int, maximum,
                        self._gau_noise, self._ray_noise, offset_f, gain_f],
                outputs=[self._sonar_map],
                device=self._device,
            )
        else:  # "range"
            maximum = wp.zeros((N, self._r.shape[0]), dtype=wp.float32, device=self._device)
            wp.launch(
                kernel=range_max,
                dim=bin_shape,
                inputs=[self._binned_int],
                outputs=[maximum],
                device=self._device,
            )
            wp.launch(
                kernel=make_sonar_map_range,
                dim=bin_shape,
                inputs=[self._r, self._azi, self._binned_int, maximum,
                        self._gau_noise, self._ray_noise, offset_f, gain_f],
                outputs=[self._sonar_map],
                device=self._device,
            )

        # ── 11. Sonar image   dim = (N, R, A) ─────────────────────────
        self._sonar_img.zero_()
        wp.launch(
            kernel=_make_sonar_image_kernel,
            dim=bin_shape,
            inputs=[self._sonar_map],
            outputs=[self._sonar_img],
            device=self._device,
        )

        # ── 12. Store outputs ──────────────────────────────────────────
        self.data.output["sonar_map"]   = self._sonar_map
        self.data.output["sonar_image"] = wp.to_torch(self._sonar_img)

        # ── 13. Viewport (env 0 only) ──────────────────────────────────
        if self._sonar_provider is not None:
            R_bins, A_bins = self._sonar_map.shape[1], self._sonar_map.shape[2]
            # sonar_img[0] = env 0 slice: (R, A+1, 4)
            env0_ptr = self._sonar_img[0].ptr
            self._sonar_provider.set_bytes_data_from_gpu(env0_ptr, [A_bins, R_bins])

        self._frame_id += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unproject_to_pcl(
        self,
        depth_t:   torch.Tensor,
        normals_t: torch.Tensor,
        sem_t:     torch.Tensor,
    ):
        """Convert Isaac Lab Camera tensors to batched flat arrays.

        Returns
        -------
        pcl      : np.ndarray (N, H*W, 3)  float32  — world positions
        normals  : np.ndarray (N, H*W, 3)  float32
        semantics: np.ndarray (N, H*W)     uint32
        view_mats: np.ndarray (N, 4, 4)    float32  — extrinsic matrices
        """
        from isaaclab.utils.math import quat_to_matrix

        # Normalise shapes
        if depth_t.ndim == 4:
            depth_t = depth_t[..., 0]        # (N, H, W)
        if sem_t.ndim == 4:
            sem_t = sem_t[..., 0]            # (N, H, W)

        N, H, W = depth_t.shape

        # Camera intrinsics (same for all envs in Isaac Lab)
        try:
            fx = float(self.data.intrinsic_matrices[0, 0, 0])
            fy = float(self.data.intrinsic_matrices[0, 1, 1])
            cx = float(self.data.intrinsic_matrices[0, 0, 2])
            cy = float(self.data.intrinsic_matrices[0, 1, 2])
        except Exception:
            fx = fy = float(W) / (2.0 * np.tan(np.deg2rad(self.cfg.hori_fov) / 2.0))
            cx, cy = W / 2.0, H / 2.0

        # Pixel grid (shared)
        us, vs = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)
        us = us.reshape(-1).astype(np.float32)             # (H*W,)
        vs = vs.reshape(-1).astype(np.float32)

        depth_np = depth_t.cpu().numpy()                    # (N, H, W)
        nrm_np   = normals_t[:, :, :, :3].cpu().numpy()    # (N, H, W, 3)
        sem_np   = sem_t.cpu().numpy().astype(np.uint32)    # (N, H, W)

        pcl_out = np.empty((N, H*W, 3), dtype=np.float32)
        nrm_out = nrm_np.reshape(N, H*W, 3).astype(np.float32)
        sem_out = sem_np.reshape(N, H*W)

        for n in range(N):
            d = depth_np[n].reshape(-1)                     # (H*W,)
            pcl_out[n, :, 0] = (us - cx) / fx * d
            pcl_out[n, :, 1] = (vs - cy) / fy * d
            pcl_out[n, :, 2] = d

        # Per-env view transforms
        view_mats = np.empty((N, 4, 4), dtype=np.float32)
        for n in range(N):
            pos  = self._data.pos_w[n].cpu().numpy()
            quat = self._data.quat_w_world[n].cpu().numpy()  # (w,x,y,z)
            R_mat = quat_to_matrix(
                torch.tensor(quat, dtype=torch.float32)
            ).numpy()
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R_mat
            T[:3,  3] = -(R_mat @ pos)
            view_mats[n] = T

        return pcl_out, nrm_out, sem_out, view_mats

    def _build_refl_lut(self, sem_np: np.ndarray) -> np.ndarray:
        """Build per-env, per-semantic-index reflectivity LUT.

        Returns
        -------
        np.ndarray (N, max_id+1)  float32   defaults to 1.0
        """
        N = sem_np.shape[0]
        max_id = int(sem_np.max()) if sem_np.size > 0 else 0
        lut = np.ones((N, max_id + 1), dtype=np.float32)

        refl_map: dict[int, float] = getattr(self.cfg, "semantic_to_reflectivity", {})
        for idx, val in refl_map.items():
            if idx <= max_id:
                lut[:, idx] = float(val)  # same mapping for all envs
        return lut

    # ------------------------------------------------------------------
    # Viewport
    # ------------------------------------------------------------------

    def _make_viewport(self) -> None:
        self._viewport_window = ui.Window("ImagingSonar Viewport", width=800, height=840)
        self._sonar_provider  = ui.ByteImageProvider()

        with self._viewport_window.frame:
            with ui.ZStack(height=720, width=720):
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.Label(
                    "Run the scenario for sonar image to appear",
                    style={"font_size": 40, "alignment": ui.Alignment.CENTER},
                    word_wrap=True,
                )
                ui.ImageWithProvider(
                    self._sonar_provider,
                    style={
                        "width": 720,
                        "height": 720,
                        "fill_policy": ui.FillPolicy.STRETCH,
                        "alignment": ui.Alignment.CENTER,
                    },
                )

    # ------------------------------------------------------------------
    # Destructor
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        if hasattr(self, "_viewport_window") and self._viewport_window is not None:
            self._viewport_window.destroy()
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
import warp as wp
import omni.ui as ui

from isaaclab.sensors import Camera
from .UWrenderer_parallel_utils import UW_render_batch
if TYPE_CHECKING:
    from .UW_Camera_cfg import UWCameraCfg

class UWCamera(Camera):
    cfg: UWCameraCfg

    def __init__(self, cfg: UWCameraCfg):
        super().__init__(cfg)

    def _initialize_impl(self):
        super()._initialize_impl()
        self._device = wp.get_preferred_device()

        N = self.num_instances

        # per-env water params: numpy (N, 3) as source-of-truth before GPU tensors are ready
        self._backscatter_value_np = np.tile(self.cfg.backscatter_value, (N, 1)).astype(np.float32)
        self._atten_coeff_np       = np.tile(self.cfg.atten_coeff,       (N, 1)).astype(np.float32)
        self._backscatter_coeff_np = np.tile(self.cfg.backscatter_coeff, (N, 1)).astype(np.float32)

        # GPU tensors — initialized lazily on first _apply_uw_render (device known then)
        self._backscatter_value_t: torch.Tensor | None = None  # (N, 3)
        self._atten_coeff_t:       torch.Tensor | None = None  # (N, 3)
        self._backscatter_coeff_t: torch.Tensor | None = None  # (N, 3)

        self._provider = None
        if self.cfg.enable_viewport:
            self._make_viewport()

    def _init_gpu_tensors(self, device: torch.device) -> None:
        self._backscatter_value_t = torch.from_numpy(self._backscatter_value_np.copy()).to(device)
        self._atten_coeff_t       = torch.from_numpy(self._atten_coeff_np.copy()).to(device)
        self._backscatter_coeff_t = torch.from_numpy(self._backscatter_coeff_np.copy()).to(device)

    def update(self, dt: float, force_recompute: bool = False):
        super().update(dt, force_recompute=force_recompute)
        self._apply_uw_render()

    def _apply_uw_render(self):
        raw_rgba = self.data.output.get("rgba")
        depth    = self.data.output.get("distance_to_camera")
        if raw_rgba is None or depth is None:
            return

        N, H, W, _ = raw_rgba.shape

        # lazy GPU init
        if self._atten_coeff_t is None:
            self._init_gpu_tensors(raw_rgba.device)

        raw_wp   = wp.from_torch(raw_rgba.contiguous(),  dtype=wp.uint8)
        depth_wp = wp.from_torch(depth.contiguous(),     dtype=wp.float32)
        bv_wp    = wp.from_torch(self._backscatter_value_t.contiguous(), dtype=wp.float32)
        ac_wp    = wp.from_torch(self._atten_coeff_t.contiguous(),       dtype=wp.float32)
        bc_wp    = wp.from_torch(self._backscatter_coeff_t.contiguous(), dtype=wp.float32)
        uw_wp    = wp.zeros((N, H, W, 4), dtype=wp.uint8, device=self._device)

        wp.launch(
            kernel=UW_render_batch,
            dim=(N, H, W),
            inputs=[raw_wp, depth_wp, bv_wp, ac_wp, bc_wp],
            outputs=[uw_wp]
        )

        self.data.output["uw_rgb"] = wp.to_torch(uw_wp)

        if self._provider is not None:
            env_id = self.cfg.viewport_env_id
            self._provider.set_bytes_data_from_gpu(uw_wp[env_id].ptr, (self.cfg.width, self.cfg.height))

    def set_water_params(self,
                         env_ids: Sequence[int],
                         backscatter_value: tuple | None = None,
                         atten_coeff:       tuple | None = None,
                         backscatter_coeff: tuple | None = None) -> None:
        """env_ids에 해당하는 환경의 수질 파라미터를 갱신."""
        if backscatter_value is not None:
            val = np.array(backscatter_value, dtype=np.float32)
            self._backscatter_value_np[env_ids] = val
            if self._backscatter_value_t is not None:
                self._backscatter_value_t[env_ids] = torch.from_numpy(val).to(self._backscatter_value_t.device)

        if atten_coeff is not None:
            val = np.array(atten_coeff, dtype=np.float32)
            self._atten_coeff_np[env_ids] = val
            if self._atten_coeff_t is not None:
                self._atten_coeff_t[env_ids] = torch.from_numpy(val).to(self._atten_coeff_t.device)

        if backscatter_coeff is not None:
            val = np.array(backscatter_coeff, dtype=np.float32)
            self._backscatter_coeff_np[env_ids] = val
            if self._backscatter_coeff_t is not None:
                self._backscatter_coeff_t[env_ids] = torch.from_numpy(val).to(self._backscatter_coeff_t.device)

    def _make_viewport(self):
        width, height = self.cfg.width, self.cfg.height
        self.window = ui.Window(f"UW Camera Viewport (Env: {self.cfg.viewport_env_id})",
                                width=width, height=height + 40)
        self._provider = ui.ByteImageProvider()

        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.ImageWithProvider(self._provider, width=ui.Percent(100), height=ui.Percent(100),
                                   style={'fill_policy': ui.FillPolicy.PRESERVE_ASPECT_FIT})

    def __del__(self):
        if hasattr(self, 'window') and self.window:
            self._window.destroy()

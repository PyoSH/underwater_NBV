from __future__ import annotations
from typing import TYPE_CHECKING

import warp as wp
import omni.ui as ui

from isaaclab.sensors import Camera
from .UWrenderer_parallel_utils import UW_render_batch
if TYPE_CHECKING:
    from .UW_Camera_cfg import UWCameraCfg

class UWCamera(Camera):
    cfg: UWCameraCfg

    def __init__(self, cfg:UWCameraCfg):
        super().__init__(cfg)
    
    def _initialize_impl(self):
        super()._initialize_impl()
        self._device = wp.get_preferred_device()

        self._backscatter_value = wp.vec3f(*self.cfg.backscatter_value)
        self._atten_coeff       = wp.vec3f(*self.cfg.atten_coeff)
        self._backscatter_coeff = wp.vec3f(*self.cfg.backscatter_coeff)
        
        # self._data.output["uw_rgba"] = None
        self._provider = None
        if self.cfg.enable_viewport:
            self._make_viewport()
    
    def update(self, dt: float, force_recompute: bool = False):
        super().update(dt, force_recompute=force_recompute)

        self._apply_uw_render()

    def _apply_uw_render(self):
        raw_rgba = self.data.output.get("rgba")
        depth = self.data.output.get("distance_to_camera")

        if raw_rgba is None or depth is None:
            return

        N, H, W, _ = raw_rgba.shape

        raw_wp = wp.from_torch(raw_rgba.contiguous(), dtype=wp.uint8)
        depth_wp = wp.from_torch(depth.contiguous(), dtype=wp.float32)
        uw_wp = wp.zeros((N,H,W,4), dtype=wp.uint8, device=self._device)
        
        wp.launch(
            kernel=UW_render_batch,
            dim=(N,H,W),
            inputs=[
                raw_wp,
                depth_wp,
                self._backscatter_value,
                self._atten_coeff,
                self._backscatter_coeff
            ],
            outputs=[uw_wp]
        )

        self.data.output["uw_rgb"] = wp.to_torch(uw_wp)

        if self._provider is not None:
            env_id = self.cfg.viewport_env_id
            self._provider.set_bytes_data_from_gpu(uw_wp[env_id].ptr, (self.cfg.width, self.cfg.height))

    '''
    추후에 env_id 기능 넣을 것.
    '''
    def set_water_params(self,
                         backscatter_value: tuple | None=None,
                         atten_coeff:       tuple | None=None,
                         backscatter_coeff: tuple | None=None) -> None:
        if backscatter_value is not None:
            self._backscatter_value = wp.vec3f(*backscatter_value)
        if atten_coeff is not None:
            self._atten_coeff = wp.vec3f(*atten_coeff)
        if backscatter_coeff is not None:
            self._backscatter_coeff = wp.vec3f(*backscatter_coeff)

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
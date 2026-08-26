from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
import warp as wp

from isaaclab.sensors import Camera

# NOTE: `omni.ui`는 여기서 import하지 않는다 — viewport 창(_make_viewport)에서만
# 쓰이는 선택적 의존인데, 최상단에서 무조건 import하면 `omni.ui`가 없는 headless
# kit(isaaclab.python.headless*.kit)에서 이 모듈을 **쓰지도 않았는데** import
# 단계에서 ModuleNotFoundError로 죽는다. 그 때문에 "조명/이미징 문제로 headless
# 학습이 불가능하다"고 알려져 있었으나, 실제 원인은 이 import 하나였다
# (2026-08-26 확인). headless + 카메라는 isaaclab.python.headless.rendering.kit로
# 정상 지원되므로, 지연 import로 바꿔 headless 학습이 가능해졌다.
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
        # 지연 import — 모듈 최상단 주석 참조(headless kit에는 omni.ui가 없다).
        # viewport를 실제로 켠 경우에만 필요하므로 여기서 가져온다.
        import omni.ui as ui

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

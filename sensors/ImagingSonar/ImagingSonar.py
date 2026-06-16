"""
ImagingSonar — Isaac Lab sensor (single-env, pointcloud annotator).

Pipeline per update()
─────────────────────
  Replicator pointcloud annotator  → pcl (M,3) world, normals (M,4), semantics (M,)
  Replicator CameraParams annotator → viewTransform (4,4)
       │
       ▼  _apply_sonar_pipeline()
  1. compute_intensity   : dim=(M,)
  2. world2local         : dim=(M,)
  3. bin_intensity       : dim=(M,)   → bin_sum/count (R, A)
  4. average (optional)  : dim=(R, A)
  5. noise kernels       : dim=(R, A)
  6. make_sonar_map_*    : dim=(R, A)
  7. make_sonar_image    : dim=(R, A)

Output (N=1 차원 유지 — env.py 변경 불필요):
    sensor.data.output["sonar_map"]    wp.array  (R, A)     vec3 (x, y, intensity)
    sensor.data.output["sonar_image"]  torch.Tensor (1, R, A+1, 4)  uint8 RGBA
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
import omni.replicator.core as rep
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

    def __init__(self, cfg: ImagingSonarCfg) -> None:
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _initialize_impl(self) -> None:
        super()._initialize_impl()

        self._device = wp.get_preferred_device()

        # ── Polar meshgrid ────────────────────────────────────────────
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

        # ── GPU buffers (단일 env, N 차원 없음) ───────────────────────
        self._bin_sum    = wp.zeros((R_bins, A_bins), dtype=wp.float32, device=self._device)
        self._bin_count  = wp.zeros((R_bins, A_bins), dtype=wp.int32,   device=self._device)
        self._binned_int = wp.zeros((R_bins, A_bins), dtype=wp.float32, device=self._device)
        self._sonar_map  = wp.zeros((R_bins, A_bins), dtype=wp.vec3,    device=self._device)
        self._sonar_img  = wp.zeros((R_bins, A_bins + 1, 4), dtype=wp.uint8, device=self._device)
        self._gau_noise  = wp.zeros((R_bins, A_bins), dtype=wp.float32, device=self._device)
        self._ray_noise  = wp.zeros((R_bins, A_bins), dtype=wp.float32, device=self._device)

        self._frame_id: int = 0

        # ── Replicator annotators ─────────────────────────────────────
        rp = self.render_product_paths[0]

        self._pcl_annot = rep.AnnotatorRegistry.get_annotator(
            "pointcloud",
            init_params={"includeUnlabelled": True},
            device=str(self._device),
        )
        self._pcl_annot.attach(rp)

        self._cam_params_annot = rep.AnnotatorRegistry.get_annotator("CameraParams")
        self._cam_params_annot.attach(rp)

        # oceansim semanticSeg_annot와 동일: idToLabels 취득용 (data_types=[]이므로 수동 attach)
        self._sem_annot = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False},
        )
        self._sem_annot.attach(rp)

        # ── Viewport ─────────────────────────────────────────────────
        self._sonar_provider = None
        if self.cfg.enable_viewport:
            self._make_viewport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt: float, force_recompute: bool = False) -> None:
        super().update(dt, force_recompute=force_recompute)
        self._apply_sonar_pipeline()

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _apply_sonar_pipeline(self) -> None:
        _dbg = (self._frame_id < 5)

        # ── 1. Pointcloud annotator 데이터 수집 ───────────────────────
        # Replicator pipeline이 첫 render step 이전엔 AnnotatorCache에 없음 → KeyError
        try:
            pcl_data = self._pcl_annot.get_data()
        except KeyError:
            if _dbg:
                print(f"[Sonar dbg frame={self._frame_id}] annotator cache not ready → skip")
            return
        if pcl_data is None or len(pcl_data.get("data", [])) == 0:
            if _dbg:
                print(f"[Sonar dbg frame={self._frame_id}] pcl empty → skip")
            return

        pcl_wp  = pcl_data["data"]                          # warp array (M, 3) world frame
        nrm_raw = pcl_data["info"]["pointNormals"]          # warp array (M, 4)
        sem_raw = pcl_data["info"]["pointSemantic"]         # warp array (M,)  uint32 RGBA

        nrm_np = nrm_raw.numpy()[:, :3].astype(np.float32)
        nrm_wp = wp.array(nrm_np, ndim=2, dtype=wp.float32, device=self._device)
        sem_np = sem_raw.numpy().astype(np.uint32)

        M = pcl_wp.shape[0]
        if _dbg:
            print(f"[Sonar dbg frame={self._frame_id}] M={M}  sem unique={np.unique(sem_np)[:4]}")

        # ── 2. View transform (CameraParams annotator) ────────────────
        try:
            cam_data = self._cam_params_annot.get_data()
        except KeyError:
            if _dbg:
                print(f"[Sonar dbg frame={self._frame_id}] cam_params cache not ready → skip")
            return
        if cam_data is None:
            if _dbg:
                print(f"[Sonar dbg frame={self._frame_id}] cam_data None → skip")
            return
        view_np  = cam_data["cameraViewTransform"].reshape(4, 4).T
        view_mat = wp.mat44(view_np.flatten().tolist())

        # ── 3. Per-point reflectivity (oceansim make_indexToProp의 5.x 이식) ──
        # oceansim: indexToProp = np.ones(...) → 기본값 1.0, add_update_semantics로 오버라이드
        # 5.x 변경: idToLabels 키가 정수문자열 → RGBA 튜플문자열, semantic ID가 RGBA uint32
        # oceansim semanticSeg_annot 패턴: data_types=[]이므로 _sem_annot에서 직접 취득
        try:
            sem_out = self._sem_annot.get_data()
        except KeyError:
            if _dbg:
                print(f"[Sonar dbg frame={self._frame_id}] sem cache not ready → skip")
            return
        id_to_labels = sem_out.get("info", {}).get("idToLabels", {}) if sem_out else {}
        if _dbg:
            print(f"[Sonar dbg frame={self._frame_id}] id_to_labels={id_to_labels}")
        # oceansim guard: idToLabels가 비어 있으면 annotator 미준비
        if not id_to_labels:
            if _dbg:
                print(f"[Sonar dbg frame={self._frame_id}] id_to_labels empty → skip")
            return
        refl_np = np.ones(M, dtype=np.float32)   # oceansim np.ones 기본값과 동일
        refl_map = self._build_refl_map(id_to_labels)
        if _dbg:
            print(f"[Sonar dbg frame={self._frame_id}] refl_map={refl_map}")
        for uid, refl in refl_map.items():
            refl_np[sem_np == uid] = refl
        refl_wp = wp.array(refl_np, ndim=1, dtype=wp.float32, device=self._device)

        # ── 4. Per-point intensity   dim = (M,) ───────────────────────
        intensity_wp = wp.empty((M,), dtype=wp.float32, device=self._device)
        wp.launch(
            kernel=compute_intensity, dim=M,
            inputs=[pcl_wp, nrm_wp, view_mat, refl_wp, self.cfg.attenuation],
            outputs=[intensity_wp],
            device=self._device,
        )

        # ── 5. World → local → spherical   dim = (M,) ─────────────────
        pcl_local_wp = wp.empty((M,), dtype=wp.vec3, device=self._device)
        pcl_spher_wp = wp.empty((M,), dtype=wp.vec3, device=self._device)
        wp.launch(
            kernel=world2local, dim=M,
            inputs=[view_mat, pcl_wp],
            outputs=[pcl_local_wp, pcl_spher_wp],
            device=self._device,
        )

        # ── 6. Bin   dim = (M,) ───────────────────────────────────────
        self._bin_sum.zero_()
        self._bin_count.zero_()
        self._binned_int.zero_()
        wp.launch(
            kernel=bin_intensity, dim=M,
            inputs=[
                pcl_spher_wp, intensity_wp,
                wp.float32(self.cfg.min_range),
                wp.float32(self.min_azi),
                wp.float32(self.cfg.range_res),
                wp.float32(float(np.deg2rad(self.cfg.angular_res))),
            ],
            outputs=[self._bin_sum, self._bin_count],
            device=self._device,
        )

        # ── 7. Binning method   dim = (R, A) ──────────────────────────
        bin_shape = self._bin_sum.shape
        if self.cfg.binning_method == "mean":
            wp.launch(
                kernel=average, dim=bin_shape,
                inputs=[self._bin_sum, self._bin_count],
                outputs=[self._binned_int],
                device=self._device,
            )
        else:  # "sum"
            self._binned_int = self._bin_sum

        # ── 8. Noise   dim = (R, A) ───────────────────────────────────
        self._gau_noise.zero_()
        self._ray_noise.zero_()
        self._sonar_map.zero_()

        wp.launch(
            kernel=normal_2d, dim=bin_shape,
            inputs=[self._frame_id, 0.0, self.cfg.gau_noise_param],
            outputs=[self._gau_noise],
            device=self._device,
        )
        wp.launch(
            kernel=range_dependent_rayleigh_2d, dim=bin_shape,
            inputs=[
                self._frame_id, self._r, self._azi,
                self.cfg.max_range, self.cfg.ray_noise_param,
                self.cfg.central_peak, self.cfg.central_std,
            ],
            outputs=[self._ray_noise],
            device=self._device,
        )

        # ── 9. Normalise + composite   dim = (R, A) ───────────────────
        offset_f = wp.float32(self.cfg.intensity_offset)
        gain_f   = wp.float32(self.cfg.intensity_gain)

        if self.cfg.normalizing_method == "all":
            maximum = wp.zeros((1,), dtype=wp.float32, device=self._device)
            wp.launch(
                kernel=all_max, dim=bin_shape,
                inputs=[self._binned_int], outputs=[maximum],
                device=self._device,
            )
            wp.launch(
                kernel=make_sonar_map_all, dim=bin_shape,
                inputs=[self._r, self._azi, self._binned_int, maximum,
                        self._gau_noise, self._ray_noise, offset_f, gain_f],
                outputs=[self._sonar_map],
                device=self._device,
            )
        else:  # "range"
            maximum = wp.zeros((self._r.shape[0],), dtype=wp.float32, device=self._device)
            wp.launch(
                kernel=range_max, dim=bin_shape,
                inputs=[self._binned_int], outputs=[maximum],
                device=self._device,
            )
            wp.launch(
                kernel=make_sonar_map_range, dim=bin_shape,
                inputs=[self._r, self._azi, self._binned_int, maximum,
                        self._gau_noise, self._ray_noise, offset_f, gain_f],
                outputs=[self._sonar_map],
                device=self._device,
            )

        # ── 10. Sonar image   dim = (R, A) ────────────────────────────
        self._sonar_img.zero_()
        wp.launch(
            kernel=_make_sonar_image_kernel, dim=bin_shape,
            inputs=[self._sonar_map], outputs=[self._sonar_img],
            device=self._device,
        )

        # ── 11. 출력 저장 (N=1 차원 유지 → env.py 변경 불필요) ──────────
        self._data.output["sonar_map"]   = self._sonar_map
        self._data.output["sonar_image"] = wp.to_torch(self._sonar_img).unsqueeze(0)  # (1, R, A+1, 4)

        # ── 12. Viewport (env 0 only) ──────────────────────────────────
        if self._sonar_provider is not None:
            R_bins, A_bins = self._sonar_map.shape
            self._sonar_provider.set_bytes_data_from_gpu(
                self._sonar_img.ptr, [A_bins, R_bins])

        self._frame_id += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_refl_map(id_to_labels: dict) -> dict:
        """oceansim make_indexToProp_array의 Isaac Sim 5.x 이식.

        4.x: 키 = 정수 문자열 '2', semantic ID = 소정수
        5.x: 키 = RGBA 튜플 문자열 '(r, g, b, a)', semantic ID = RGBA uint32

        Returns:
            {uint32_id: float_reflectivity}  — 'reflectivity' 속성이 있는 항목만
        """
        result = {}
        for key, info in id_to_labels.items():
            if not isinstance(info, dict) or "reflectivity" not in info:
                continue
            key_str = str(key).strip()
            if key_str.startswith("("):                         # 5.x RGBA 포맷
                r, g, b, a = [int(x.strip()) for x in key_str.strip("()").split(",")]
                uid = np.uint32((int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r))
            else:                                               # 4.x 정수 포맷 (하위호환)
                uid = np.uint32(int(key_str))
            result[uid] = float(info["reflectivity"])
        return result

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
        try:
            rp = self.render_product_paths[0]
            if hasattr(self, "_pcl_annot"):
                self._pcl_annot.detach(rp)
            if hasattr(self, "_cam_params_annot"):
                self._cam_params_annot.detach(rp)
            if hasattr(self, "_sem_annot"):
                self._sem_annot.detach(rp)
        except Exception:
            pass
        if hasattr(self, "_viewport_window") and self._viewport_window is not None:
            self._viewport_window.destroy()

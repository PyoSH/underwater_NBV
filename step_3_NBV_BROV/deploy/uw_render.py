"""수중 렌더 — Isaac 의 UWCamera 와 **같은 식**을 ROS/Gazebo 쪽에서 재현한다. 2026-09-10.

출처: `sensors/UWCamera/UWrenderer_parallel_utils.py:34`

    UW_RGB = raw_RGB * exp(-d * atten_coeff) + backscatter_value * 255 * (1 - exp(-d * backscatter_coeff))

Gazebo 는 수중 감쇠/후방산란을 모사하지 않으므로 그대로 두면 정책이 학습 때 본 적 없는
밝고 선명한 이미지를 받는다. rgbd_camera 가 depth 를 주므로 픽셀별로 같은 식을 적용해
이미지 도메인 갭을 닫는다. 채널 순서 주의: Isaac 은 RGB, OpenCV/ROS bgr8 은 BGR.
"""
from __future__ import annotations

import numpy as np

# step_1_NBV/utils_NBV/jerlov_presets.py — (R, G, B) 순서
JERLOV_PRESETS = {
    "IB":  dict(atten_coeff=(0.325835, 0.196346, 0.177762),
                backscatter_coeff=(0.279616, 0.186807, 0.176059),
                backscatter_value=(0.181711, 0.495286, 0.647559)),
    "II":  dict(atten_coeff=(0.386476, 0.262386, 0.257074),
                backscatter_coeff=(0.352879, 0.255191, 0.255578),
                backscatter_value=(0.194231, 0.487382, 0.582686)),
    "III": dict(atten_coeff=(0.512426, 0.395839, 0.411934),
                backscatter_coeff=(0.490660, 0.391060, 0.410257),
                backscatter_value=(0.198886, 0.449882, 0.471129)),
    "1C":  dict(atten_coeff=(0.664638, 0.551640, 0.579858),
                backscatter_coeff=(0.648422, 0.547907, 0.577633),
                backscatter_value=(0.212554, 0.447515, 0.359715)),
}


def uw_render_bgr(bgr: np.ndarray, depth: np.ndarray, preset: str = "IB",
                  far_fill: float = 20.0) -> np.ndarray:
    """BGR uint8 + depth[m] float32 -> 수중 BGR uint8.

    depth 가 유한하지 않은 화소(하늘/무한대)는 `far_fill` 로 채운다 — 감쇠가 포화하므로
    사실상 후방산란 색이 된다. 이는 Isaac 에서 clipping_range 밖이 배경색이 되는 것과 같다.
    """
    p = JERLOV_PRESETS[preset]
    ac = np.asarray(p["atten_coeff"], np.float32)[::-1]          # RGB -> BGR
    bc = np.asarray(p["backscatter_coeff"], np.float32)[::-1]
    bv = np.asarray(p["backscatter_value"], np.float32)[::-1]

    d = np.asarray(depth, np.float32)
    d = np.where(np.isfinite(d) & (d > 0.0), d, far_fill)[..., None]

    src = bgr.astype(np.float32)
    out = src * np.exp(-d * ac) + bv * 255.0 * (1.0 - np.exp(-d * bc))
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def sim_fov_window(fx_real: float, w: int, h: int, hfov_sim_deg: float = 47.2,
                   vfov_sim_deg: float = 36.3):
    """실기 해상도에서 sim 화각에 해당하는 중앙 창 (x0, y0, cw, ch).

    640x480 · fx 465.518 · sim 47.2x36.3 도 -> 407 x 305 px (계획서 §11.1).
    이미지와 depth 가 **같은 창**을 써야 정책 관측과 지도가 같은 장면을 본다.
    """
    import math
    cw = min(int(round(2.0 * fx_real * math.tan(math.radians(hfov_sim_deg) / 2.0))), w)
    ch = min(int(round(2.0 * fx_real * math.tan(math.radians(vfov_sim_deg) / 2.0))), h)
    return (w - cw) // 2, (h - ch) // 2, cw, ch


def crop_to_sim_fov(img: np.ndarray, fx_real: float, fx_sim_hfov_deg: float = 47.2,
                    vfov_sim_deg: float = 36.3) -> np.ndarray:
    """실기 FOV(69.0 x 54.6) 이미지를 sim FOV(47.2 x 36.3) 로 중앙 crop.

    정책은 sim 화각으로 학습됐으므로 화각을 맞춰야 관측 분포가 같아진다.
    640x480, fx 465.518 에서 crop 크기는 407 x 305 px (DEPLOY_3WEEK_PLAN §11.1).
    """
    h, w = img.shape[:2]
    x0, y0, cw, ch = sim_fov_window(fx_real, w, h, fx_sim_hfov_deg, vfov_sim_deg)
    return img[y0:y0 + ch, x0:x0 + cw]

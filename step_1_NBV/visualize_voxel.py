"""
visualize_voxel.py
==================
evaluate_recon.py 에피소드 디렉토리의 voxel numpy 파일을 Open3D로 시각화.

출력:
  binary_voxel.png  — occupied(파랑) / free(회색) 이진 복셀
  quality_voxel.png — 관측 품질 연속값 (jet colormap)

Isaac Sim 없이 독립 실행 (EGL 헤드리스, 디스플레이 불필요):
  python visualize_voxel.py \
      --ep_dir ./recon_output/UW_NBV_3/ep_000_env0 \
      --out_dir ./voxel_vis \
      --voxel_size 0.05
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt


Q_SAT = 0.80

COLOR_OCCUPIED = np.array([0.2, 0.4, 0.8])
COLOR_FREE     = np.array([0.7, 0.7, 0.7])


def _render_offscreen(coords: np.ndarray, colors: np.ndarray,
                      voxel_size: float, out_path: Path,
                      width: int = 800, height: int = 800) -> None:
    """OffscreenRenderer (EGL) 로 헤드리스 렌더링."""
    renderer = rendering.OffscreenRenderer(width, height)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    renderer.scene.add_geometry("geom", pcd, mat)
    renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

    # 카메라: 비스듬한 45° 시점
    bbox   = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    diag   = float(np.linalg.norm(extent))

    front = np.array([1.0, -1.0, 0.19])
    front = front / np.linalg.norm(front)
    eye   = center + front * diag * 1.1
    up    = np.array([0.0, 0.0, 1.0])

    renderer.scene.camera.look_at(
        center.tolist(), eye.tolist(), up.tolist()
    )

    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)
    print(f"  saved → {out_path}")


def visualize_binary(weight_vol: np.ndarray, tsdf_vol: np.ndarray,
                     voxel_size: float, out_path: Path) -> None:
    occupied_mask = (weight_vol > 0) & (tsdf_vol <= 0)

    if not occupied_mask.any():
        print("  [binary] occupied voxel 없음, 스킵")
        return

    coords = np.argwhere(occupied_mask) * voxel_size
    colors = np.tile(COLOR_OCCUPIED, (len(coords), 1))

    _render_offscreen(coords, colors, voxel_size, out_path)


def visualize_quality(weight_vol: np.ndarray, tsdf_vol: np.ndarray,
                      quality_vol: np.ndarray,
                      voxel_size: float, out_path: Path) -> None:
    # binary와 동일한 occupied surface voxel 대상 — ch2 표현 방식만 다름
    mask = (weight_vol > 0) & (tsdf_vol <= 0)
    if not mask.any():
        print("  [quality] occupied voxel 없음, 스킵")
        return

    coords = np.argwhere(mask) * voxel_size
    q_vals = np.clip(quality_vol[mask] / Q_SAT, 0.0, 1.0)
    colors = plt.cm.jet(q_vals)[:, :3]

    _render_offscreen(coords, colors, voxel_size, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="evaluate_recon 에피소드 voxel numpy → Open3D 이미지"
    )
    parser.add_argument("--ep_dir",     type=str, required=True)
    parser.add_argument("--out_dir",    type=str, default="./voxel_vis")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--hires",      action="store_true",
                        help="고해상도 voxel 사용 (*_hires.npy, voxel_size=0.025)")
    args = parser.parse_args()

    ep_dir  = Path(args.ep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_hires" if args.hires else ""
    if args.hires:
        args.voxel_size = 0.025

    weight_vol  = np.load(ep_dir / f"weight_vol{suffix}.npy")
    tsdf_vol    = np.load(ep_dir / f"tsdf_vol{suffix}.npy")
    quality_vol = np.load(ep_dir / f"quality_vol{suffix}.npy")

    print("[binary voxel]")
    visualize_binary(weight_vol, tsdf_vol, args.voxel_size,
                     out_dir / "binary_voxel.png")

    print("[quality voxel]")
    visualize_quality(weight_vol, tsdf_vol, quality_vol, args.voxel_size,
                      out_dir / "quality_voxel.png")


if __name__ == "__main__":
    main()

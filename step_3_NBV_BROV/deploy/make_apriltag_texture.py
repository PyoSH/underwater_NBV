"""AprilTag 16h5 ID 2 텍스처 + UV 명시 평면 메쉬 생성 (Gazebo용). 2026-09-10.

치수 근거 — brov_perception/config/aruco.yaml:
  payload 4 cell + 검은 테두리 1 cell = 한 변 6 cell
  6 cell x 0.070 m = 0.420 m  <- marker_length_m (검은 외곽 테두리)
  quiet zone 1 cell(0.070 m)은 검출에 필수이나 위 길이에 불포함
  => 인쇄/렌더 영역 8 cell x 0.070 = 0.560 m

box 프리미티브에 albedo_map을 걸면 UV 방향이 렌더러 구현에 달려 있고, V가 뒤집히면
태그가 거울상이 되어 **검출 자체가 실패**한다. 그래서 UV를 직접 쓴 평면 OBJ를 낸다.
검출이 안 되면 --flip-v 로 한 번에 뒤집는다 (Phase 2 진단 항목).

사용법:
  python3 deploy/make_apriltag_texture.py --out deploy/models/apriltag_16h5_id2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CELL_M = 0.070          # aruco.yaml: 6 cell x 0.070 = 0.420 m
BORDER_BITS = 1         # 검은 테두리 1 cell
PAYLOAD_CELLS = 4       # 16h5 = 4x4
QUIET_CELLS = 1
PX_PER_CELL = 100

TAG_CELLS = PAYLOAD_CELLS + 2 * BORDER_BITS          # 6
FULL_CELLS = TAG_CELLS + 2 * QUIET_CELLS             # 8
BLACK_EDGE_M = TAG_CELLS * CELL_M                    # 0.420
FULL_EDGE_M = FULL_CELLS * CELL_M                    # 0.560


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--marker-id", type=int, default=2)
    ap.add_argument("--flip-v", action="store_true",
                    help="검출 실패 시 사용 — 텍스처 V축을 뒤집는다")
    args = ap.parse_args()

    tex_dir = args.out / "materials" / "textures"
    mesh_dir = args.out / "meshes"
    tex_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    tag = cv2.aruco.generateImageMarker(
        dictionary, args.marker_id, TAG_CELLS * PX_PER_CELL, BORDER_BITS
    )

    pad = QUIET_CELLS * PX_PER_CELL
    img = np.full((FULL_CELLS * PX_PER_CELL,) * 2, 255, np.uint8)
    img[pad:pad + tag.shape[0], pad:pad + tag.shape[1]] = tag
    if args.flip_v:
        img = img[::-1]

    png = tex_dir / f"apriltag_16h5_id{args.marker_id}.png"
    cv2.imwrite(str(png), img)

    # ── UV 명시 평면 (+Z 법선, 원점 중앙) ──
    h = FULL_EDGE_M / 2.0
    obj = mesh_dir / f"apriltag_16h5_id{args.marker_id}.obj"
    mtl = mesh_dir / f"apriltag_16h5_id{args.marker_id}.mtl"
    obj.write_text(
        f"mtllib {mtl.name}\n"
        f"usemtl tag\n"
        f"v {-h:.6f} {-h:.6f} 0.0\n"
        f"v { h:.6f} {-h:.6f} 0.0\n"
        f"v { h:.6f} { h:.6f} 0.0\n"
        f"v {-h:.6f} { h:.6f} 0.0\n"
        # 이미지 좌상단이 (-x,+y) 가 되도록 = 태그 '위쪽'이 메쉬 +Y
        "vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n"
        "vn 0.0 0.0 1.0\n"
        "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n"
    )
    mtl.write_text(
        "newmtl tag\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n"
        "d 1.0\nillum 1\n"
        f"map_Kd ../materials/textures/{png.name}\n"
    )

    print(f"[apriltag] id={args.marker_id} dict=DICT_APRILTAG_16h5")
    print(f"[apriltag] 검은 테두리 한 변 {BLACK_EDGE_M:.3f} m  "
          f"(= marker_length_m, aruco.yaml 준수)")
    print(f"[apriltag] quiet zone 포함 평면 {FULL_EDGE_M:.3f} m, "
          f"{img.shape[1]}x{img.shape[0]} px")
    print(f"[apriltag] {png}")
    print(f"[apriltag] {obj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

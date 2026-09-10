"""표적 물체의 GT 표면 voxel 마스크 — Gazebo 루프를 Isaac 과 **같은 눈금**으로 채점한다.

`envs/env_utils.py::_voxelize_gt_mesh` 와 동일한 절차를 OBJ 에서 재현한다(USD 불필요):
면마다 균등 barycentric 표본 1점 -> 표본 bbox 중심에서 볼륨 원점 산출 -> voxel 기입.

검증: Isaac 실측 GT 표면 voxel 은 **455~475** 개다(계획서 §11.1). 이 스크립트가 그
범위를 내면 격자 정렬이 맞은 것이다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_obj(path: Path):
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(v, np.float32), np.asarray(f, np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=Path, required=True)
    ap.add_argument("--yaw-deg", type=float, default=45.0)
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--vol-dim", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    verts, faces = load_obj(args.obj)
    a = math.radians(args.yaw_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0],
                    [0, 0, 1]], np.float32)
    verts = verts @ rot.T                       # 물체 원점 기준 world 정렬

    rng = np.random.default_rng(args.seed)
    r1 = rng.random((len(faces), 1)).astype(np.float32)
    r2 = rng.random((len(faces), 1)).astype(np.float32)
    w0 = 1.0 - np.sqrt(r1)
    w1 = np.sqrt(r1) * (1.0 - r2)
    w2 = np.sqrt(r1) * r2
    pts = w0 * verts[faces[:, 0]] + w1 * verts[faces[:, 1]] + w2 * verts[faces[:, 2]]

    nx, ny, nz = args.vol_dim
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2.0
    origin = center - np.array([nx, ny, nz], np.float32) * args.voxel / 2.0

    idx = ((pts - origin) / args.voxel).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array([nx, ny, nz])), axis=1)
    idx = idx[ok]
    surf = np.zeros((nx, ny, nz), bool)
    surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, surf)
    meta = dict(obj=str(args.obj), yaw_deg=args.yaw_deg, voxel=args.voxel,
                vol_dim=list(args.vol_dim),
                vol_origin_object_frame=[float(v) for v in origin],
                surface_voxels=int(surf.sum()),
                sampled_bbox_min=[float(v) for v in lo],
                sampled_bbox_max=[float(v) for v in hi])
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[gt-surface] 표면 voxel {surf.sum()}  (Isaac 실측 455~475 와 비교)")
    print(f"[gt-surface] 볼륨 원점(물체 프레임) {origin.tolist()}")
    print(f"[gt-surface] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""단일 OBJ → USD 변환 (실제 표적 물체용). 2026-09-10.

`convert_gso_to_usd.py`는 GSO zip 묶음을 0.75 m로 **정규화**해 변환한다.
실수조 표적(STEP → FreeCAD 메쉬)은 **실제 크기 그대로** 넣어야 sim 평가가
수조와 같은 기하를 재현하므로, 정규화 없이(scale 1.0) 같은 MeshConverter 경로로
변환하고 mesh_pool이 읽는 manifest(1항목)를 쓴다.

텍스처는 없다(석고 표면). mesh_pool의 텍스처 실재 확인을 통과하지 못하므로
평가 시 `--no_require_texture`를 줄 것. 이 물체가 실제로도 무늬가 거의 없다는
점은 sim2real의 이미지 채널 위험으로 별도 기록한다.

사용법 (isaac-lab-base 컨테이너):
  /isaac-sim/python.sh -u tools/convert_obj_to_usd.py --headless \\
      --obj /workspace/OceanRL_test/robots/data/real_object/plaster_mold.obj \\
      --out /workspace/OceanRL_test/robots/data/real_object/usd --name plaster_mold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="단일 OBJ → USD (실제 크기)")
parser.add_argument("--obj", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--scale", type=float, default=1.0,
                    help="기본 1.0 = 실제 크기. OBJ가 mm 단위면 0.001")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402


def obj_extent(path: Path) -> np.ndarray:
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                pts.append([float(x) for x in line.split()[1:4]])
    a = np.asarray(pts, dtype=np.float64)
    return a.max(0) - a.min(0), a.min(0), a.max(0)


def main() -> int:
    obj, out = Path(args.obj), Path(args.out)
    ext, lo, hi = obj_extent(obj)
    s = args.scale
    usd_path = out / args.name / f"{args.name}.usd"
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    MeshConverter(MeshConverterCfg(
        asset_path=str(obj), usd_dir=str(usd_path.parent), usd_file_name=usd_path.name,
        force_usd_conversion=True, scale=(s, s, s),
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=False),
        rigid_props=None, mass_props=None,
    ))
    ext_m = ext * s
    entry = dict(
        name=args.name, usd=str(usd_path), scale=s,
        orig_extent=[float(v) for v in ext], norm_extent=[float(v) for v in ext_m],
        bbox_min=[float(v * s) for v in lo], bbox_max=[float(v * s) for v in hi],
        aspect_min_over_max=float(ext.min() / ext.max()),
        has_texture=False, n_texture=0, solidity=None,
        source="STEP 20220811-석고틀.stp → FreeCAD 0.19 tessellation (deflection 2 mm)",
    )
    (out / "manifest.json").write_text(json.dumps([entry], indent=2, ensure_ascii=False))
    print(f"[obj2usd] {args.name}: extent {ext_m[0]:.3f} x {ext_m[1]:.3f} x {ext_m[2]:.3f} m "
          f"(bbox z {lo[2]*s:.3f}..{hi[2]*s:.3f}), aspect {entry['aspect_min_over_max']:.2f}")
    print(f"[obj2usd] USD: {usd_path}\n[obj2usd] manifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)

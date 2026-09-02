"""Google Scanned Objects(GSO) zip → 정규화된 USD 변환 (Stage 4).

왜 GSO인가
----------
Stage 4(다중 물체 일반화)에 쓸 대상 메쉬가 필요한데, OceanSim 자산에는
`rock.usd` 하나뿐이다. 수중 분야에는 물체 형상 데이터셋이 없어(장면 단위이거나
소나 점군이거나 2D) 지상 데이터셋을 가져오는 것이 유일한 경로이며, 이는
step_1의 원형인 GenNBV(CVPR 2024)가 Houses3K 학습 → OmniObject3D/Objaverse
교차평가로 이미 확립한 프로토콜이다.

GSO는 실제 스캔 생활용품 ~1,030개로 품질이 균일하고, `meshes/model.obj` +
`materials/textures/texture.png` 구조라 Isaac 변환기와 궁합이 좋다.

스케일 정규화가 왜 필수인가
---------------------------
TSDF 볼륨이 40³ × 5 cm = **2 m 큐브**다. GSO는 실측 크기라 물체마다 제각각이라
(테이프 8 cm, 토스터 30 cm) 그대로 넣으면 작은 물체는 몇 voxel만 차지해
coverage가 무의미해진다. 최대 변 길이를 목표 범위로 맞춰 변환 시점에 굽는다 —
env 쪽이 메쉬별 스케일을 알 필요가 없어진다.

주의: 텍스처 보존
-----------------
IsaacLab에 Objaverse `.glb` 임포트 시 텍스처가 유실된 사례가 보고돼 있다
(Discussion #595). 변환 후 USD에 UsdShade 머티리얼과 텍스처 참조가 실제로
들어갔는지 **검증**하고, 실패한 자산은 목록에서 뺀다. 우리 actor 입력은
그레이스케일이라 색 자체보다 **휘도 변화(표면 무늬)**가 중요한데, 텍스처가
없으면 실루엣만 남아 표면 관측이라는 과제가 성립하지 않는다.

사용법 (isaac-lab-base 컨테이너 안)
-----------------------------------
/isaac-sim/python.sh -u tools/convert_gso_to_usd.py --headless \\
    --src /workspace/google_scanned_obj --count 20 \\
    --out /workspace/OceanRL_test/robots/data/gso_usd
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GSO zip → 정규화 USD")
parser.add_argument("--src", type=str, required=True, help="GSO zip들이 있는 디렉토리")
parser.add_argument("--out", type=str, required=True, help="USD 출력 디렉토리")
parser.add_argument("--count", type=int, default=20, help="이름 정렬 기준 앞에서 N개")
parser.add_argument("--names", type=str, default=None,
                    help="쉼표 구분 zip 이름(확장자 제외). 주면 --count 무시")
parser.add_argument("--target_size", type=float, default=1.2,
                    help="정규화 후 최대 변 길이 [m]. TSDF 볼륨 2 m 큐브 안에 "
                         "회전해도 들어가야 하므로 여유를 둔다")
parser.add_argument("--stage_dir", type=str, default="/tmp/gso_stage")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg


def obj_extent(obj_path: Path) -> np.ndarray:
    """OBJ의 정점 bounding box 크기. 별도 의존성 없이 v 행만 읽는다."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = np.array([float(t) for t in line.split()[1:4]])
                lo = np.minimum(lo, p)
                hi = np.maximum(hi, p)
    return hi - lo


def verify_usd_texture(usd_path: Path) -> tuple[bool, int]:
    """변환된 USD에 텍스처 참조가 살아있는지. (성공여부, 텍스처 수)"""
    from pxr import Sdf, Usd, UsdShade
    stage = Usd.Stage.Open(str(usd_path))
    n_tex = 0
    # 변환기가 메쉬를 Props/instanceable_meshes.usd로 분리하고 instanceable로
    # 표시하는데, 기본 Traverse()는 instance proxy 안으로 안 들어간다 —
    # 그래서 셰이더를 하나도 못 찾는다. 명시적으로 프록시까지 순회한다.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            for inp in shader.GetInputs():
                val = inp.Get()
                if isinstance(val, Sdf.AssetPath) and (val.path or val.resolvedPath):
                    n_tex += 1
    return n_tex > 0, n_tex


def main() -> int:
    src, out = Path(args.src), Path(args.out)
    stage_dir = Path(args.stage_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(src.glob("*.zip"))
    if args.names:
        want = {n.strip() for n in args.names.split(",")}
        zips = [z for z in zips if z.stem in want]
    else:
        zips = zips[: args.count]
    print(f"[gso] 대상 {len(zips)}개 (목표 최대변 {args.target_size} m)")

    manifest = []
    for i, z in enumerate(zips):
        name = z.stem
        work = stage_dir / name
        if work.exists():
            shutil.rmtree(work)
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(work)
        except zipfile.BadZipFile:
            print(f"[gso] {i+1:>2}/{len(zips)} {name}: zip 손상 — 건너뜀 "
                  f"(다운로드 진행 중일 수 있음)")
            continue

        obj = work / "meshes" / "model.obj"
        if not obj.exists():
            print(f"[gso] {i+1:>2}/{len(zips)} {name}: model.obj 없음 — 건너뜀")
            continue

        # GSO 패키징 불일치 보정: model.mtl은 `map_Kd texture.png`라고 쓰는데
        # (= mtl이 있는 meshes/ 기준 상대경로) 실제 파일은
        # materials/textures/texture.png에 있다. Gazebo SDF 로더는 이를
        # 알아서 해결하지만 Isaac 변환기는 못 찾아 텍스처를 통째로 잃는다
        # (2026-09-02 실측: 3/3 모두 텍스처 0개). mtl이 기대하는 자리로
        # 복사해 준다 — mtl을 고쳐 쓰는 것보다 원본을 덜 건드린다.
        for tex in (work / "materials" / "textures").glob("*"):
            if tex.is_file():
                target = obj.parent / tex.name
                if not target.exists():
                    shutil.copy2(tex, target)

        ext = obj_extent(obj)
        max_ext = float(ext.max())
        if not np.isfinite(max_ext) or max_ext <= 1e-6:
            print(f"[gso] {i+1:>2}/{len(zips)} {name}: 정점 파싱 실패 — 건너뜀")
            continue
        s = args.target_size / max_ext

        # **객체마다 하위 디렉토리**를 쓴다. 변환기는 텍스처를 usd_dir/textures/에
        # 복사하는데 GSO는 파일명이 전부 texture.png라, 한 디렉토리에 모으면
        # 나중 객체가 앞 객체의 텍스처를 덮어쓴다(2026-09-02 실측으로 확인).
        usd_path = out / name / f"{name}.usd"
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = MeshConverterCfg(
            asset_path=str(obj),
            usd_dir=str(usd_path.parent),
            usd_file_name=usd_path.name,
            force_usd_conversion=True,
            scale=(s, s, s),
            # 대상 물체는 관측 대상일 뿐 물리 상호작용을 하지 않는다 —
            # 충돌/강체 속성을 넣으면 스폰 비용만 늘고 쓰이지 않는다.
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=None,
            mass_props=None,
        )
        MeshConverter(cfg)

        ok, n_tex = verify_usd_texture(usd_path)
        # 정규화 결과 검산: 스케일 적용 후 실제 크기
        norm_ext = ext * s
        print(f"[gso] {i+1:>2}/{len(zips)} {name[:38]:<38} "
              f"원본 {max_ext:.3f}m → ×{s:.2f} → {norm_ext.max():.2f}m  "
              f"텍스처 {'OK' if ok else '없음!'}({n_tex})")
        manifest.append(dict(
            name=name, usd=str(usd_path), scale=s,
            orig_extent=[float(v) for v in ext],
            norm_extent=[float(v) for v in norm_ext],
            # 최소변/최대변 비 — 1에 가까우면 등방(구/상자류), 작으면 납작하다.
            # 납작한 물체는 어느 시점에서 봐도 비슷해 NBV 과제로는 부적합하다.
            aspect_min_over_max=float(ext.min() / ext.max()),
            has_texture=bool(ok), n_texture=int(n_tex),
        ))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok_list = [m for m in manifest if m["has_texture"]]
    print(f"\n[gso] 변환 성공 {len(manifest)}개 중 텍스처 보존 {len(ok_list)}개")
    print(f"[gso] manifest: {out/'manifest.json'}")
    if manifest:
        flat = [m for m in manifest if m["aspect_min_over_max"] < 0.25]
        if flat:
            print(f"[gso] ⚠ 납작한 형상 {len(flat)}개 — NBV 대상으로 부적합할 수 있다:")
            for m in flat:
                print(f"      {m['name'][:44]:<44} 종횡비 {m['aspect_min_over_max']:.2f}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)

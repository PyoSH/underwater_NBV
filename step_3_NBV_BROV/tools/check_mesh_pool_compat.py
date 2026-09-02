"""메쉬 풀이 step_3 환경과 호환되는지 사전 점검 (Stage 4).

씬을 띄우기 전에 USD 자산만 열어 확인한다. 학습을 돌려놓고 나서 발견하면
비싸기 때문이다.

무엇을 왜 확인하는가
--------------------
1. **`Usd.PrimRange`로 Mesh를 찾을 수 있는가**
   `env_utils.py::_load_mesh()`가 `/World/envs/env_N/Object` 아래를 기본
   `Usd.PrimRange`로 훑어 첫 `UsdGeom.Mesh`를 찾는다. 그런데 Isaac 변환기는
   메쉬를 `Props/instanceable_meshes.usd`로 분리하고 instanceable로 표시하며,
   **기본 순회는 instance proxy 안으로 들어가지 않는다**. 텍스처 검증에서
   이미 같은 함정에 빠졌다(셰이더 0개로 보고됨). 여기서 걸리면 GT 복셀화가
   통째로 실패한다.

2. **삼각형 수**
   `_voxelize_gt_mesh()`가 면마다 barycentric 샘플을 뽑는다. 바위는 163만
   삼각형이었고 그 Python 삼각분할이 처리량을 5.7배 깎았다(벡터화로 해결).
   새 자산이 그보다 훨씬 크면 리셋 비용을 다시 봐야 한다.

3. **정규화 후 크기가 TSDF 볼륨에 들어가는가**
   볼륨은 40³×5 cm = 2 m 큐브다. 변환 시 최대변을 1.2 m로 맞췄지만
   `_randomize_rock_pose()`가 그 위에 0.8~1.5배 스케일과 임의 회전을 더한다.
   최악의 경우 대각선이 볼륨을 넘어 잘린다.

사용법
------
/isaac-sim/python.sh -u tools/check_mesh_pool_compat.py --headless \\
    --manifest ../robots/data/gso_usd/manifest.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="메쉬 풀 호환성 점검")
parser.add_argument("--manifest", type=str, required=True)
parser.add_argument("--filter_flat", action="store_true", default=True)
parser.add_argument("--voxel_size", type=float, default=0.05)
parser.add_argument("--vol_dim", type=int, default=40)
parser.add_argument("--scale_max", type=float, default=1.5,
                    help="_randomize_rock_pose()가 얹는 최대 스케일")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from envs.mesh_pool import load_mesh_pool


def main() -> int:
    from pxr import Usd, UsdGeom

    import json
    pool = load_mesh_pool(args.manifest, filter_flat=args.filter_flat)
    # 변환기의 scale은 정점에 굽지 않고 **Xform 스케일**로 붙는다. 로컬 정점만
    # 읽으면 원본 크기가 나오므로 manifest의 scale을 곱해야 실제 크기가 된다
    # (`_load_mesh()`는 GetLocalToWorldTransform을 쓰므로 스케일이 반영된다).
    scale_of = {e["name"]: e["scale"] for e in json.loads(open(args.manifest).read())}
    vol_m = args.vol_dim * args.voxel_size
    print(f"\nTSDF 볼륨 = {args.vol_dim}³ × {args.voxel_size*100:.0f} cm "
          f"= {vol_m:.1f} m 큐브\n")
    print(f"{'자산':<40}{'기본순회':>9}{'프록시':>8}{'삼각형':>11}{'최대변':>8}{'회전시':>8}")
    print("-" * 86)

    n_bad_traverse, n_oversize = 0, 0
    for usd_path in pool:
        stage = Usd.Stage.Open(usd_path)
        root = stage.GetPseudoRoot()
        plain = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
        proxy = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                 if p.IsA(UsdGeom.Mesh)]

        n_tri, ext = 0, 0.0
        for m in (proxy or plain):
            mesh = UsdGeom.Mesh(m)
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            n_tri += sum(max(c - 2, 0) for c in counts)
            pts = mesh.GetPointsAttr().Get() or []
            if pts:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
                ext = max(ext, max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)))

        # 최악의 경우: 변환 스케일 × 리셋 랜덤 스케일 × 대각선 회전
        conv = scale_of.get(os.path.basename(usd_path)[:-4], 1.0)
        ext = ext * conv
        worst = ext * args.scale_max * math.sqrt(3)
        over = worst > vol_m
        # `_load_mesh()`는 2026-09-02부터 TraverseInstanceProxies를 쓴다.
        # 따라서 결함 판정 기준은 "기본 순회로 못 찾음"이 아니라
        # **"프록시 순회로도 못 찾음"**이다.
        n_bad_traverse += (len(proxy) == 0)
        n_oversize += over
        name = os.path.basename(usd_path)[:38]
        print(f"{name:<40}{len(plain):>9}{len(proxy):>8}{n_tri:>11,}"
              f"{ext:>8.2f}{worst:>7.2f}{'!' if over else ' '}")

    print("-" * 86)
    print(f"\n[판정]")
    if n_bad_traverse:
        print(f"  ✗ 프록시 순회로도 Mesh를 못 찾는 자산 {n_bad_traverse}개 — GT 복셀화 실패.")
    else:
        print(f"  ✓ 모든 자산에서 Mesh를 찾는다 (`_load_mesh()`의 프록시 순회 기준).")
        print(f"     참고: 기본 `Usd.PrimRange`만으로는 여전히 못 찾는다 — 변환기가")
        print(f"     메쉬를 instanceable로 분리하기 때문이며, 그래서 프록시 순회가 필수다.")
    if n_oversize:
        print(f"  ✗ 최대 스케일+회전 시 볼륨({vol_m:.1f} m)을 넘는 자산 {n_oversize}개.")
        print(f"     → 변환 시 `--target_size`를 낮추거나 `_randomize_rock_pose()`의")
        print(f"        스케일 상한을 줄일 것. 넘치면 GT 표면이 잘려 coverage 분모가 틀어진다.")
    else:
        print(f"  ✓ 최악 조건에서도 볼륨 안에 들어간다.")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)

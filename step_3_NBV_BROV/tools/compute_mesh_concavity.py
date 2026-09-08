"""GSO 메쉬의 오목도를 계산해 manifest에 기록한다 (Stage 4 물체 선별 ③).

왜 필요한가
-----------
NBV는 "지금 안 보이는 면이 있다"는 전제 위에서만 성립한다. 그런데 GSO는
대부분 볼록에 가까운 생활용품이라, 물체를 프레임에 담을 수 있는 거리에서는
한두 시점이면 표면이 거의 드러난다. 실측(2026-09-07): 25결정 random이 0.789,
40결정 상한이 0.850 — **여유가 0.06밖에 없다.** 정책이 배울 여지가 없으면
보상 지형이 평평해지고, PPO는 아무 경계로나 간다(Stage 2는 psi 하한 고착,
run03은 psi 상한 고착). 두 번 모두 "구석 찾기"였다.

납작 필터(종횡비)는 "앞뒤 두 장이면 끝나는 물체"를 걸렀지만, **볼록한 물체**는
그대로 통과한다. 이 도구는 그 축을 잰다.

지표
----
- **solidity** = 메쉬 부피 / 볼록껍질 부피.  1에 가까우면 볼록(=가려짐 없음),
  낮을수록 오목·구멍·돌출이 많다(=시점 선택이 의미 있다).
  부피는 발산정리로 구한다: V = (1/6)|Σ (v0 × v1)·v2|. 워터타이트가 아니면
  값이 망가지므로 `hull_area_ratio`를 함께 남겨 교차 확인한다.
- **hull_area_ratio** = 볼록껍질 표면적 / 메쉬 표면적.  낮을수록 표면이 복잡.

두 지표를 모두 남기고 필터 임계값은 `mesh_pool.py`에서 정한다 — 변환은 싸고
선별은 실험 조건이라는 기존 원칙과 같다.

실행 (pxr은 kit 확장에 있어 경로를 직접 준다):
    P=/workspace/isaac-sim/extscache/omni.usd.libs-*
    PYTHONPATH=$P LD_LIBRARY_PATH=$P/bin:$LD_LIBRARY_PATH \
      /workspace/isaac-sim/python.sh -u tools/compute_mesh_concavity.py \
      --manifest <manifest.json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import ConvexHull


def load_mesh(usd_path: Path):
    """USD에서 정점·삼각형을 읽는다.

    `Usd.TraverseInstanceProxies()`가 필수다 — Isaac 메쉬 변환기는 지오메트리를
    `Props/instanceable_meshes.usd`로 분리하고 원본을 instanceable로 표시하는데,
    기본 순회는 instance proxy 안으로 들어가지 않아 Mesh를 0개 찾는다.
    """
    stage = Usd.Stage.Open(str(usd_path))
    mesh = None
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            break
    if mesh is None:
        return None, None
    verts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    cnt = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    if cnt.size == 0 or verts.size == 0:
        return None, None
    if bool((cnt == 3).all()):
        tris = idx.reshape(-1, 3)
    else:                                   # 혼합 다각형 → 부채꼴 삼각화
        per = np.maximum(cnt - 2, 0)
        start = np.zeros(cnt.size, dtype=np.int64)
        np.cumsum(cnt[:-1], out=start[1:])
        tstart = np.zeros(cnt.size, dtype=np.int64)
        np.cumsum(per[:-1], out=tstart[1:])
        fid = np.repeat(np.arange(cnt.size), per)
        j = np.arange(int(per.sum())) - tstart[fid] + 1
        b = start[fid]
        tris = np.stack([idx[b], idx[b + j], idx[b + j + 1]], axis=-1)
    return verts, tris


def rebuild_entries(root: Path) -> list[dict]:
    """USD 디렉토리에서 manifest를 복원한다.

    변환기(`convert_gso_to_usd.py`)가 기록하던 필드를 USD에서 역산한다:
    정규화 배율은 `/model/geometry`의 xformOp:scale에 남아 있고, 원본 크기는
    변환 전 정점 extent이므로 `정규화 extent / 배율`로 되돌릴 수 있다.

    디렉토리 이름 순(=변환 당시 알파벳순)으로 정렬해 원본과 같은 순서를
    유지한다 — `mesh_pool.py`의 train/holdout 분할이 이 순서에 의존한다.
    """
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        usd = d / f"{d.name}.usd"
        if not usd.exists():
            continue
        try:
            verts, _ = load_mesh(usd)
            if verts is None:
                continue
            stage = Usd.Stage.Open(str(usd))
            mesh = next(pr for pr in Usd.PrimRange(stage.GetPseudoRoot(),
                                                   Usd.TraverseInstanceProxies())
                        if pr.IsA(UsdGeom.Mesh))
            m = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(mesh)).reshape(4, 4).T
            scale = float(np.linalg.norm(m[:3, 0]))
            raw = verts.max(axis=0) - verts.min(axis=0)          # 변환 전 크기
            norm = raw * scale
            tex = sorted((d / "textures").glob("*")) if (d / "textures").is_dir() else []
        except Exception:                                        # noqa: BLE001
            continue
        out.append(dict(
            name=d.name, usd=str(usd), scale=scale,
            orig_extent=[float(v) for v in raw],
            norm_extent=[float(v) for v in norm],
            aspect_min_over_max=float(raw.min() / raw.max()),
            has_texture=bool(tex), n_texture=len(tex),
        ))
    return out


def metrics(verts, tris) -> dict:
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    # 발산정리 — 워터타이트 가정. 부호는 winding에 좌우되므로 절댓값.
    vol = abs(np.einsum("ij,ij->i", np.cross(v0, v1), v2).sum()) / 6.0
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
    hull = ConvexHull(verts)
    return dict(
        solidity=float(vol / hull.volume) if hull.volume > 0 else float("nan"),
        hull_area_ratio=float(hull.area / area) if area > 0 else float("nan"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="처리할 개수 제한(시험용). **manifest는 자르지 않는다** — "
                         "2026-09-07에 잘린 리스트를 그대로 덮어써 manifest를 "
                         "1030→20으로 파손한 적이 있다")
    ap.add_argument("--rebuild", action="store_true",
                    help="USD 디렉토리에서 manifest를 **새로 만든다**. 변환 당시의 "
                         "필드(scale/extent/텍스처)를 USD에서 역산해 복원한다")
    args = ap.parse_args()

    mp = Path(args.manifest)
    if args.rebuild:
        entries = rebuild_entries(mp.parent)
        print(f"[concavity] USD 디렉토리에서 {len(entries)}개 재구성")
    else:
        entries = json.loads(mp.read_text())

    # 처리 대상만 자른다 — `entries` 자체를 자르면 저장 시 manifest가 파손된다.
    targets = entries[: args.limit] if args.limit else entries

    t0, ok, fail = time.time(), 0, []
    for i, e in enumerate(targets):
        usd = mp.parent / e["name"] / f"{e['name']}.usd"
        try:
            verts, tris = load_mesh(usd)
            if verts is None:
                raise RuntimeError("메쉬 없음")
            e.update(metrics(verts, tris))
            ok += 1
        except Exception as exc:                       # noqa: BLE001
            e["solidity"] = float("nan")
            e["hull_area_ratio"] = float("nan")
            fail.append((e["name"], str(exc)[:60]))
        if (i + 1) % 100 == 0:
            print(f"[concavity] {i+1}/{len(targets)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    mp.write_text(json.dumps(entries, indent=2))
    sol = np.array([e["solidity"] for e in entries], dtype=float)
    sol = sol[np.isfinite(sol)]
    print(f"\n[concavity] 성공 {ok} / 실패 {len(fail)}  ({time.time()-t0:.0f}s)")
    if len(sol):
        q = lambda p: float(np.quantile(sol, p))
        print(f"[concavity] solidity  최소 {sol.min():.3f} / p10 {q(.10):.3f} / "
              f"중앙 {q(.50):.3f} / p90 {q(.90):.3f} / 최대 {sol.max():.3f}")
        for th in (0.5, 0.6, 0.7, 0.8, 0.9):
            print(f"   solidity < {th:.1f} (오목) 인 물체: {int((sol < th).sum()):4d}개")
    for n, why in fail[:5]:
        print(f"[concavity]   실패: {n[:40]:<40} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

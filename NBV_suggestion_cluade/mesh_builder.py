"""
mesh_builder.py
---------------
USD Stage의 geometry를 추출하여 NBUV용 coarse mesh를 생성한다.

NBUV 논문 Section 7:
  "We exploit a rough prior 3D model, since underwater such a model
   is routinely obtained using active sonar."

이 구현에서는:
  - Isaac Sim USD geometry를 직접 추출 (소나 스캔 결과로 간주)
  - Open3D voxel downsampling으로 의도적으로 coarsening
  - 결과를 삼각형 패치 배열로 변환

출력 데이터 구조:
  patches: dict
    'centers':  (N, 3) float64  - 패치 중심 좌표 (미터)
    'normals':  (N, 3) float64  - 패치 법선 벡터 (단위벡터)
    'areas':    (N,)   float64  - 패치 면적 (m^2)
    'N':        int             - 패치 총 수
"""

import numpy as np


# ── Isaac Sim 환경 여부 확인 ───────────────────────────────────────────────────
try:
    import omni.usd
    from pxr import UsdGeom, Gf
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False

# ── Open3D 환경 여부 확인 ─────────────────────────────────────────────────────
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────

def build_patches_from_usd(prim_path: str, voxel_size: float = 0.05) -> dict:
    """
    Isaac Sim USD Stage에서 geometry를 추출하여 패치 배열을 반환한다.

    Args:
        prim_path:  USD 씬 내 대상 prim 경로 (예: "/World/Environment/shipwreck")
        voxel_size: coarsening용 voxel 크기 (미터)

    Returns:
        patches dict (상단 모듈 주석 참조)
    """
    if not ISAAC_SIM_AVAILABLE:
        raise RuntimeError("Isaac Sim 환경이 아닙니다. build_patches_from_numpy()를 사용하세요.")

    vertices, triangles = _extract_usd_geometry(prim_path)
    return _build_patches(vertices, triangles, voxel_size)


def build_patches_from_numpy(vertices: np.ndarray,
                              triangles: np.ndarray,
                              voxel_size: float = 0.05) -> dict:
    """
    NumPy 배열로 제공된 geometry에서 패치 배열을 반환한다.
    Isaac Sim 없이도 동작하는 독립 실행 / 테스트용 인터페이스.

    Args:
        vertices:  (V, 3) float64 - 정점 좌표
        triangles: (T, 3) int     - 삼각형 인덱스
        voxel_size: coarsening용 voxel 크기

    Returns:
        patches dict
    """
    return _build_patches(vertices, triangles, voxel_size)


def build_patches_from_pointcloud(points: np.ndarray,
                                  voxel_size: float = 0.05,
                                  poisson_depth: int = 6) -> dict:
    """
    소나 포인트클라우드에서 Poisson reconstruction → 패치 배열을 반환한다.
    Open3D가 필요하다.

    Args:
        points:        (P, 3) float64 - 소나 포인트 좌표
        voxel_size:    coarsening voxel 크기
        poisson_depth: Poisson reconstruction 깊이 (낮을수록 coarse)

    Returns:
        patches dict
    """
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("Open3D가 설치되지 않았습니다: pip install open3d")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 3, max_nn=30))

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth)

    vertices  = np.asarray(mesh.vertices,  dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    return _build_patches(vertices, triangles, voxel_size)


# ─────────────────────────────────────────────────────────────────────────────
# 내부 구현
# ─────────────────────────────────────────────────────────────────────────────

def _extract_usd_geometry(prim_path: str):
    """USD Stage에서 정점/삼각형 배열을 추출한다."""
    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise ValueError(f"유효하지 않은 prim 경로: {prim_path}")

    mesh_prim = UsdGeom.Mesh(prim)
    points_attr = mesh_prim.GetPointsAttr().Get()
    indices_attr = mesh_prim.GetFaceVertexIndicesAttr().Get()

    vertices  = np.array([[p[0], p[1], p[2]] for p in points_attr], dtype=np.float64)
    triangles = np.array(indices_attr, dtype=np.int64).reshape(-1, 3)
    return vertices, triangles


def _build_patches(vertices: np.ndarray,
                   triangles: np.ndarray,
                   voxel_size: float) -> dict:
    """
    geometry → (선택적 coarsening) → 패치 배열 생성.

    coarsening 전략:
      Open3D가 있으면: voxel downsampling → Poisson reconstruction
      없으면:          삼각형을 그대로 사용 (no coarsening)
    """
    if OPEN3D_AVAILABLE and voxel_size > 0:
        vertices, triangles = _coarsen_mesh(vertices, triangles, voxel_size)

    centers, normals, areas = _compute_patch_properties(vertices, triangles)

    # 면적이 0인 degenerate 삼각형 제거
    valid = areas > 1e-10
    centers = centers[valid]
    normals = normals[valid]
    areas   = areas[valid]

    return {
        'centers': centers,   # (N, 3)
        'normals': normals,   # (N, 3)
        'areas':   areas,     # (N,)
        'N':       len(areas),
    }


def _coarsen_mesh(vertices: np.ndarray,
                  triangles: np.ndarray,
                  voxel_size: float):
    """
    Open3D를 사용하여 mesh를 coarsening한다.
    NBUV 논문 Section 7의 "coarse prior" 재현.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # voxel downsampling으로 정점 수 감소
    mesh_simplified = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

    mesh_simplified.remove_degenerate_triangles()
    mesh_simplified.remove_duplicated_vertices()

    v_out = np.asarray(mesh_simplified.vertices,  dtype=np.float64)
    t_out = np.asarray(mesh_simplified.triangles, dtype=np.int64)
    return v_out, t_out


def _compute_patch_properties(vertices: np.ndarray,
                               triangles: np.ndarray):
    """
    삼각형 배열에서 중심, 법선, 면적을 한 번에 계산한다.

    Returns:
        centers: (T, 3)
        normals: (T, 3) 단위벡터
        areas:   (T,)
    """
    v0 = vertices[triangles[:, 0]]  # (T, 3)
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # 중심: 세 정점의 평균
    centers = (v0 + v1 + v2) / 3.0  # (T, 3)

    # 법선: 외적 (크기 = 2 × 삼각형 면적)
    edge1 = v1 - v0  # (T, 3)
    edge2 = v2 - v0
    cross  = np.cross(edge1, edge2)  # (T, 3)

    # 면적 = |cross| / 2
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)  # (T, 1)
    areas = cross_norm[:, 0] / 2.0  # (T,)

    # 법선 단위벡터화 (면적 0인 경우 0벡터 유지)
    safe_norm = np.where(cross_norm > 1e-12, cross_norm, 1.0)
    normals   = cross / safe_norm  # (T, 3)

    return centers, normals, areas


# ─────────────────────────────────────────────────────────────────────────────
# 테스트용 geometry 생성기
# ─────────────────────────────────────────────────────────────────────────────

def make_test_plane(nx: int = 10, ny: int = 10,
                    width: float = 1.0, height: float = 1.0) -> dict:
    """
    수평 평면 mesh를 생성한다. 단위 테스트 및 데모용.

    Returns:
        patches dict
    """
    xs = np.linspace(-width / 2,  width / 2,  nx + 1)
    ys = np.linspace(-height / 2, height / 2, ny + 1)
    xx, yy = np.meshgrid(xs, ys)

    vertices = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.zeros(len(xx.ravel()))
    ])  # (V, 3)

    triangles = []
    for j in range(ny):
        for i in range(nx):
            tl = j * (nx + 1) + i
            tr = tl + 1
            bl = tl + (nx + 1)
            br = bl + 1
            triangles.append([tl, bl, tr])
            triangles.append([tr, bl, br])
    triangles = np.array(triangles, dtype=np.int64)

    return build_patches_from_numpy(vertices, triangles, voxel_size=0.0)


if __name__ == "__main__":
    # 빠른 동작 확인
    patches = make_test_plane(nx=5, ny=5)
    print(f"패치 수: {patches['N']}")
    print(f"중심 shape: {patches['centers'].shape}")
    print(f"법선 shape: {patches['normals'].shape}")
    print(f"면적 shape: {patches['areas'].shape}")
    print(f"평균 면적: {patches['areas'].mean():.4f} m^2")

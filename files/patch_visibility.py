"""
patch_visibility.py
-------------------
카메라와 조명의 현재 pose를 기준으로 각 패치의 가시성과 거리를 계산한다.

NBUV 논문 표기:
  l_SC[s]: 카메라(C)에서 패치 s까지의 거리
  l_LS[s]: 조명(L)에서 패치 s까지의 거리

가시성 판별 기준 (모두 만족해야 visible):
  1. 법선 조건: 패치 법선이 카메라를 향해야 함 (cos_angle > 0)
  2. FOV 조건:  패치가 카메라 시야각 안에 있어야 함
  3. 조명 법선: 패치 법선이 조명을 향해야 함 (cos_angle > 0)

출력:
  visible_mask: (N,) bool   - 가시 패치 마스크
  l_SC:         (N,) float64 - 카메라-패치 거리 (m)
  l_LS:         (N,) float64 - 조명-패치 거리 (m)
  cos_cam:      (N,) float64 - 카메라 방향 cos (조명 모델에서 사용)
  cos_light:    (N,) float64 - 조명 방향 cos
"""

import numpy as np


def compute_visibility(patches: dict,
                       cam_pos: np.ndarray,
                       light_pos: np.ndarray,
                       fov_deg: float = 60.0) -> dict:
    """
    패치 배열에 대해 가시성과 거리를 계산한다.

    Args:
        patches:    mesh_builder가 반환한 patches dict
        cam_pos:    (3,) 카메라 위치 (미터)
        light_pos:  (3,) 조명 위치 (미터)
        fov_deg:    카메라 수평 FOV (도)

    Returns:
        dict:
          'visible_mask': (N,) bool
          'l_SC':         (N,) float64
          'l_LS':         (N,) float64
          'cos_cam':      (N,) float64
          'cos_light':    (N,) float64
    """
    centers = patches['centers']  # (N, 3)
    normals = patches['normals']  # (N, 3)

    # ── 카메라 → 패치 방향 및 거리 ────────────────────────────────────────────
    vec_cam  = centers - cam_pos[np.newaxis, :]    # (N, 3)
    l_SC     = np.linalg.norm(vec_cam, axis=1)     # (N,)
    dir_cam  = _safe_normalize(vec_cam, l_SC)      # (N, 3) 단위벡터

    # ── 조명 → 패치 방향 및 거리 ─────────────────────────────────────────────
    vec_light = centers - light_pos[np.newaxis, :] # (N, 3)
    l_LS      = np.linalg.norm(vec_light, axis=1)  # (N,)
    dir_light = _safe_normalize(vec_light, l_LS)   # (N, 3)

    # ── 가시성 판별 ──────────────────────────────────────────────────────────

    # 법선 방향 규약:
    #   _compute_patch_properties에서 외적 방향으로 법선이 결정됨.
    #   평면(z=0)의 경우 법선이 [0,0,-1] (아래) 또는 [0,0,+1] (위)일 수 있음.
    #   카메라가 z>0에서 아래를 볼 때, 법선과 카메라→패치 벡터의 내적이 양수이면
    #   카메라가 앞면을 보고 있음 (법선이 카메라 반대쪽을 가리킴).

    # 1) 법선 조건: n · (cam→patch) > 0 이면 카메라가 앞면을 봄
    #    (법선이 카메라 반대 방향을 가리키는 경우 포함)
    cos_cam = np.einsum('ni,ni->n', normals, dir_cam)   # (N,)
    # 내적의 절댓값이 크면 잘 보이는 것. 방향 부호로 앞/뒷면 구분.
    # 카메라→패치 방향(dir_cam)과 법선의 내적이 양수 = 같은 방향 = 앞면에서 봄
    front_cam = cos_cam > 0.0

    # 2) FOV 조건
    in_fov = _check_fov(dir_cam, fov_deg)   # (N,) bool

    # 3) 조명 법선 조건: 조명→패치 방향과 법선의 내적이 양수이면 앞면 조사
    cos_light   = np.einsum('ni,ni->n', normals, dir_light)  # (N,)
    front_light = cos_light > 0.0

    visible_mask = front_cam & in_fov & front_light

    # cos 값은 조명 계산에서 입사각으로 사용 → 절댓값 반환
    cos_cam_abs   = np.abs(cos_cam)
    cos_light_abs = np.abs(cos_light)

    return {
        'visible_mask': visible_mask,   # (N,) bool
        'l_SC':         l_SC,           # (N,)
        'l_LS':         l_LS,           # (N,)
        'cos_cam':      cos_cam_abs,    # (N,)
        'cos_light':    cos_light_abs,  # (N,)
    }


def compute_visibility_for_candidate(patches: dict,
                                     cam_pos: np.ndarray,
                                     light_pos: np.ndarray,
                                     fov_deg: float = 60.0) -> dict:
    """
    compute_visibility와 동일하지만 후보 pose 탐색 시 명시적 이름으로 호출.
    nbuv_optimizer에서 각 후보 pose에 대해 반복 호출한다.
    """
    return compute_visibility(patches, cam_pos, light_pos, fov_deg)


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _safe_normalize(vecs: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """
    벡터 배열을 안전하게 정규화한다. 길이가 0인 경우 0벡터를 유지.

    Args:
        vecs:  (N, 3)
        norms: (N,) — vecs의 L2 norm

    Returns:
        (N, 3) 단위벡터
    """
    safe = np.where(norms > 1e-12, norms, 1.0)  # (N,)
    return vecs / safe[:, np.newaxis]


def _check_fov(dir_cam: np.ndarray, fov_deg: float) -> np.ndarray:
    """
    카메라 FOV 체크.

    단순화 가정:
      카메라는 타겟을 향해 아래를 바라보고 있다 (-Z 방향).
      dir_cam은 카메라→패치 단위벡터이다.

    실제 Isaac Sim 구현 시에는 카메라 회전 행렬을 사용해야 한다.

    Args:
        dir_cam: (N, 3) 카메라→패치 단위벡터
        fov_deg: 시야각 (도)

    Returns:
        (N,) bool
    """
    half_fov_rad = np.deg2rad(fov_deg / 2.0)
    cos_half_fov = np.cos(half_fov_rad)

    # 카메라 광축: -Z (카메라가 아래 타겟을 향함)
    optical_axis = np.array([0.0, 0.0, -1.0])

    # dir_cam과 광축 사이의 cos
    cos_angle = dir_cam @ optical_axis   # (N,)

    # cos_angle > cos(half_fov) 이면 시야각 내부
    return cos_angle > cos_half_fov


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mesh_builder import make_test_plane

    patches = make_test_plane(nx=4, ny=4)
    cam_pos   = np.array([0.0,  0.0, 1.0])   # 위에서 내려다 봄
    light_pos = np.array([0.2,  0.0, 0.8])   # 약간 옆에서 조사

    vis = compute_visibility(patches, cam_pos, light_pos, fov_deg=60.0)

    n_vis = vis['visible_mask'].sum()
    print(f"전체 패치: {patches['N']}")
    print(f"가시 패치: {n_vis} ({100*n_vis/patches['N']:.1f}%)")
    print(f"l_SC 평균: {vis['l_SC'][vis['visible_mask']].mean():.3f} m")
    print(f"l_LS 평균: {vis['l_LS'][vis['visible_mask']].mean():.3f} m")

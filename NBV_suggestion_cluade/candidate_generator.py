"""
candidate_generator.py
-----------------------
NBUV exhaustive search를 위한 후보 (카메라, 조명) pose 집합을 생성한다.

생성 전략:
  1. Cartesian 반구 샘플링: 타겟 중심 위의 반구에서 N_cam개 카메라 후보
  2. Baseline 필터링: 각 카메라 후보로부터 baseline 범위 내 N_light개 조명 후보
  3. IK 필터링: Isaac Sim 환경이면 UR5e 도달 가능 여부 검사
     Isaac Sim 없으면: 단순 거리 제약으로 근사

NBUV 논문 실험:
  - 카메라-조명 baseline: 2cm ~ 34cm
  - 타겟까지 거리: ~수십 cm 스케일

출력:
  candidates: list of dict
    'cam_pos':   (3,) 카메라 위치
    'light_pos': (3,) 조명 위치
"""

import numpy as np


# ── Isaac Sim IK 환경 확인 ────────────────────────────────────────────────────
try:
    from omni.isaac.motion_generation import LulaKinematicsSolver
    ISAAC_IK_AVAILABLE = True
except ImportError:
    ISAAC_IK_AVAILABLE = False


def generate_candidates(target_center: np.ndarray,
                         constraints: dict,
                         use_ik: bool = False) -> list:
    """
    후보 (카메라, 조명) pose 집합을 생성하고 필터링한다.

    Args:
        target_center: (3,) 관심 타겟의 중심 좌표
        constraints:   workspace_constraints.json에서 로드된 dict
        use_ik:        True이면 Isaac Sim IK 필터링 적용

    Returns:
        list of dict: [{'cam_pos': (3,), 'light_pos': (3,)}, ...]
    """
    n_cam   = constraints['n_cam_candidates']
    n_light = constraints['n_light_candidates']
    r_min   = constraints['cam_hemisphere_radius_min']
    r_max   = constraints['cam_hemisphere_radius_max']
    bl_min  = constraints['baseline_min']
    bl_max  = constraints['baseline_max']
    reach   = constraints['ur5e_reach_max']

    # ── 1단계: 카메라 후보 반구 샘플링 ──────────────────────────────────────
    cam_candidates = _sample_hemisphere(target_center, r_min, r_max, n_cam)

    # ── 2단계: 각 카메라 후보에 대해 조명 후보 생성 + 필터링 ──────────────
    candidates = []
    for cam_pos in cam_candidates:

        # IK 필터링 (카메라 팔)
        if use_ik and ISAAC_IK_AVAILABLE:
            if not _check_ik_reachable(cam_pos, arm='left'):
                continue
        else:
            # 단순 거리 제약으로 근사 (Isaac Sim 없을 때)
            if np.linalg.norm(cam_pos) > reach:
                continue

        # 조명 후보 생성: 카메라 주변 baseline 범위 내
        light_candidates = _sample_baseline_sphere(cam_pos, bl_min, bl_max, n_light)

        for light_pos in light_candidates:

            # IK 필터링 (조명 팔)
            if use_ik and ISAAC_IK_AVAILABLE:
                if not _check_ik_reachable(light_pos, arm='right'):
                    continue
            else:
                if np.linalg.norm(light_pos) > reach:
                    continue

            # Baseline 재확인 (정확한 거리)
            baseline = np.linalg.norm(cam_pos - light_pos)
            if not (bl_min <= baseline <= bl_max):
                continue

            candidates.append({
                'cam_pos':   cam_pos,
                'light_pos': light_pos,
            })

    return candidates


def _sample_hemisphere(center: np.ndarray,
                        r_min: float,
                        r_max: float,
                        n: int) -> np.ndarray:
    """
    타겟 중심 위의 반구(z > 0)에서 n개의 점을 균일하게 샘플링한다.

    피보나치 격자를 사용하여 반구 위에 균일하게 분포시킨다.

    Returns:
        (n, 3) float64
    """
    # 피보나치 반구 샘플링
    golden = (1 + np.sqrt(5)) / 2
    i      = np.arange(n, dtype=np.float64)

    theta = np.arccos(1.0 - i / n)         # 극각: [0, π/2] (상반구)
    phi   = 2 * np.pi * i / golden          # 방위각

    # 구면 좌표 → 단위 벡터
    dirs = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])  # (n, 3)

    # 반경을 r_min ~ r_max 사이에서 균일 샘플링
    radii  = np.random.uniform(r_min, r_max, n)
    points = center[np.newaxis, :] + dirs * radii[:, np.newaxis]

    return points   # (n, 3)


def _sample_baseline_sphere(cam_pos: np.ndarray,
                              bl_min: float,
                              bl_max: float,
                              n: int) -> np.ndarray:
    """
    카메라 위치로부터 baseline 범위 내 구면 위의 n개 점을 샘플링한다.

    Returns:
        (n, 3) float64
    """
    # 균일 구면 샘플링 (rejection sampling)
    dirs   = _random_unit_vectors(n)
    radii  = np.random.uniform(bl_min, bl_max, n)
    points = cam_pos[np.newaxis, :] + dirs * radii[:, np.newaxis]
    return points


def _random_unit_vectors(n: int) -> np.ndarray:
    """
    n개의 균일 랜덤 단위벡터를 반환한다.

    가우시안 샘플링 → 정규화 방식으로 구면 균일 분포를 생성.

    Returns:
        (n, 3) float64
    """
    vecs  = np.random.randn(n, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return vecs / norms


def _check_ik_reachable(pos: np.ndarray, arm: str) -> bool:
    """
    Isaac Sim LulaKinematicsSolver로 UR5e IK 도달 가능 여부를 확인한다.
    (Isaac Sim 환경에서만 동작)

    Args:
        pos: (3,) 목표 end-effector 위치
        arm: 'left' (카메라) 또는 'right' (조명)

    Returns:
        bool
    """
    # TODO: Isaac Sim 연동 시 실제 IK 구현
    # 현재는 항상 True 반환 (placeholder)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, os

    config_path = os.path.join(os.path.dirname(__file__),
                               '..', 'config', 'workspace_constraints.json')
    with open(config_path) as f:
        raw = json.load(f)
    constraints = {k: v for k, v in raw.items() if not k.startswith('_')}

    target = np.array([0.0, 0.0, 0.0])

    candidates = generate_candidates(target, constraints, use_ik=False)

    print(f"생성된 후보 수: {len(candidates)}")

    if candidates:
        baselines = [np.linalg.norm(c['cam_pos'] - c['light_pos'])
                     for c in candidates]
        print(f"Baseline 범위: [{min(baselines):.3f}, {max(baselines):.3f}] m")

        cam_dists = [np.linalg.norm(c['cam_pos'] - target)
                     for c in candidates]
        print(f"카메라 거리:   [{min(cam_dists):.3f}, {max(cam_dists):.3f}] m")

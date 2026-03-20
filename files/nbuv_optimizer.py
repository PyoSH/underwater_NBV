"""
nbuv_optimizer.py
-----------------
NBUV Algorithm 1: 다음 최적 뷰를 탐색한다.

알고리즘:
  입력: 후보 pose 집합 V, 현재 추정 상태 (Q^ML, ρ̂^ML)
  
  for each (cam_pose, light_pose) in V:
    1. 가시성 계산 → visible_mask, l_SC, l_LS
    2. 조명 계산   → E_s, B_s
    3. 예상 q_next 계산 (Eq.19 전방 시뮬레이션)
    4. 정보이득 계산 (Eq.22~25)
  
  출력: argmax I_{t+1}(O) → best (cam_pose, light_pose)

최적화:
  후보 수 M이 적을 때 (≤200): 배열 브로드캐스팅으로 배치 처리
  M이 클 때: 순차 처리 (메모리 절약)
"""

import numpy as np

from models.patch_visibility import compute_visibility
from models.nbuv_lighting    import compute_lighting
from models.nbuv_info_gain   import (compute_expected_q,
                                      batch_compute_info_gains)


BATCH_THRESHOLD = 300   # 배치 처리 vs 순차 처리 기준 후보 수


def find_next_best_view(candidates: list,
                         patches: dict,
                         estimator,
                         params: dict,
                         fov_deg: float = 60.0) -> dict:
    """
    후보 pose 집합에서 최대 정보이득을 주는 pose를 찾는다 (Algorithm 1).

    Args:
        candidates:  candidate_generator.generate_candidates()의 반환값
                     [{'cam_pos': (3,), 'light_pos': (3,)}, ...]
        patches:     mesh_builder가 반환한 patches dict
        estimator:   NBUVEstimator 인스턴스 (현재 Q_ML, rho_ML 보유)
        params:      water_params (beta, sigma_RN 등)
        fov_deg:     카메라 FOV (도)

    Returns:
        dict:
          'cam_pos':     (3,) 최적 카메라 위치
          'light_pos':   (3,) 최적 조명 위치
          'info_gain':   float 예상 정보이득
          'best_index':  int 후보 집합 내 인덱스
          'all_gains':   (M,) 전체 후보의 정보이득 (분석용)
    """
    M = len(candidates)

    if M == 0:
        raise ValueError("후보 pose가 없습니다. candidate_generator를 확인하세요.")

    estimator_state = {
        'rho_ML': estimator.rho_ML,
        'Q_ML':   estimator.Q_ML,
    }

    if M <= BATCH_THRESHOLD:
        gains = _batch_search(candidates, patches, estimator_state,
                               params, fov_deg)
    else:
        gains = _sequential_search(candidates, patches, estimator_state,
                                    params, fov_deg)

    best_idx  = int(np.argmax(gains))
    best      = candidates[best_idx]

    return {
        'cam_pos':    best['cam_pos'],
        'light_pos':  best['light_pos'],
        'info_gain':  float(gains[best_idx]),
        'best_index': best_idx,
        'all_gains':  gains,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 내부 탐색 전략
# ─────────────────────────────────────────────────────────────────────────────

def _batch_search(candidates: list,
                   patches: dict,
                   estimator_state: dict,
                   params: dict,
                   fov_deg: float) -> np.ndarray:
    """
    M개 후보를 배열 연산으로 배치 처리한다.

    Returns:
        (M,) 후보별 정보이득
    """
    M = len(candidates)
    N = patches['N']

    q_batch   = np.zeros((M, N), dtype=np.float64)
    vis_batch = np.zeros((M, N), dtype=bool)

    for i, cand in enumerate(candidates):
        vis      = compute_visibility(patches, cand['cam_pos'],
                                       cand['light_pos'], fov_deg)
        lighting = compute_lighting(vis, params)

        q_next = compute_expected_q(lighting, estimator_state,
                                     sigma_RN=params['sigma_RN'])

        q_batch[i]   = q_next
        vis_batch[i] = vis['visible_mask']

    gains = batch_compute_info_gains(q_batch, estimator_state['Q_ML'], vis_batch)
    return gains


def _sequential_search(candidates: list,
                        patches: dict,
                        estimator_state: dict,
                        params: dict,
                        fov_deg: float) -> np.ndarray:
    """
    M개 후보를 순차 처리한다. 메모리 효율적.

    Returns:
        (M,) 후보별 정보이득
    """
    from models.nbuv_info_gain import compute_info_gain

    M     = len(candidates)
    gains = np.zeros(M, dtype=np.float64)

    for i, cand in enumerate(candidates):
        vis      = compute_visibility(patches, cand['cam_pos'],
                                       cand['light_pos'], fov_deg)
        lighting = compute_lighting(vis, params)

        q_next = compute_expected_q(lighting, estimator_state,
                                     sigma_RN=params['sigma_RN'])

        gains[i] = compute_info_gain(q_next,
                                      estimator_state['Q_ML'],
                                      vis['visible_mask'])

    return gains


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from models.mesh_builder       import make_test_plane
    from models.nbuv_estimator     import NBUVEstimator
    from planner.candidate_generator import generate_candidates

    patches = make_test_plane(nx=5, ny=5)
    N       = patches['N']

    config_path = os.path.join(os.path.dirname(__file__),
                               '..', 'config', 'water_params.json')
    with open(config_path) as f:
        raw = json.load(f)
    params = {k: v for k, v in raw.items() if not k.startswith('_')}
    params['sigma_RN'] = params.get('sigma_RN', 0.01)

    ws_config_path = os.path.join(os.path.dirname(__file__),
                                  '..', 'config', 'workspace_constraints.json')
    with open(ws_config_path) as f:
        raw_ws = json.load(f)
    constraints = {k: v for k, v in raw_ws.items() if not k.startswith('_')}

    estimator  = NBUVEstimator(N=N)
    target     = np.array([0.0, 0.0, 0.0])
    candidates = generate_candidates(target, constraints)

    print(f"후보 수: {len(candidates)}")

    result = find_next_best_view(candidates, patches, estimator, params)

    print(f"최적 후보 index:  {result['best_index']}")
    print(f"예상 정보이득:    {result['info_gain']:.4f}")
    print(f"최적 cam_pos:     {result['cam_pos']}")
    print(f"최적 light_pos:   {result['light_pos']}")
    print(f"전체 gain 범위:   [{result['all_gains'].min():.4f}, {result['all_gains'].max():.4f}]")

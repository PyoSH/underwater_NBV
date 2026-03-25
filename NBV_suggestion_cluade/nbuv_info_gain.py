"""
nbuv_info_gain.py
-----------------
NBUV 논문 Eq.22~25: 다음 뷰 후보의 예상 정보이득을 계산한다.

이 모듈이 NBUV의 핵심: "어느 pose로 이동하면 가장 많은 정보를 얻는가?"

수식:
  Eq.22: I_{t+1}(O) = Σ_s I_{t+1}(ρ̂^ML_s)
           전체 장면 정보이득 = 패치별 정보이득의 합

  Eq.23: I_{t+1}(ρ̂^ML_s) = H_t(ρ̂^ML_s) - H_{t+1}(ρ̂^ML_s)
           패치 s의 엔트로피 감소량

  Eq.24/25:
    I_{t+1}(ρ̂^ML_s) = (1/2) · ln(1 + q_s(t+1) / Q^ML_s(t))
           가우시안 근사 하에서의 닫힌형 정보이득
           q_s(t+1): 다음 관측의 예상 신뢰도
           Q^ML_s(t): 현재 누적 신뢰도

알고리즘 1 관련:
  후보 pose V에 대해 I_{t+1}(O)를 계산하고 argmax를 찾는다.
  이 모듈은 단일 후보의 정보이득을 계산하며,
  nbuv_optimizer가 전체 후보에 대해 반복 호출한다.
"""

import numpy as np


def compute_info_gain(q_next: np.ndarray,
                      Q_ML: np.ndarray,
                      visible_mask: np.ndarray) -> float:
    """
    단일 후보 pose에 대한 전체 정보이득을 계산한다 (Eq.22~25).

    Args:
        q_next:       (N,) 다음 관측의 예상 신뢰도 q_s(t+1)
        Q_ML:         (N,) 현재 누적 신뢰도 Q^ML_s(t)
        visible_mask: (N,) bool 해당 후보에서 가시인 패치

    Returns:
        float: 전체 정보이득 I_{t+1}(O)
    """
    # 비가시 패치: 정보이득 = 0
    if visible_mask.sum() == 0:
        return 0.0

    q_v   = q_next[visible_mask]    # 가시 패치의 예상 신뢰도
    Q_v   = Q_ML[visible_mask]      # 가시 패치의 현재 누적 신뢰도

    # Eq.24/25: I = (1/2) · ln(1 + q_next / Q_ML)
    ratio = q_v / np.maximum(Q_v, 1e-12)
    patch_gains = 0.5 * np.log1p(ratio)   # log1p(x) = ln(1+x), 수치 안정

    return float(patch_gains.sum())


def compute_expected_q(lighting_candidate: dict,
                       estimator_state: dict,
                       sigma_RN: float) -> np.ndarray:
    """
    후보 pose에서의 예상 신뢰도 q_s(t+1)를 계산한다 (Eq.19 전방 시뮬레이션).

    현재 rho_ML 추정값을 사용하여 다음 관측의 신뢰도를 예측한다.

    Args:
        lighting_candidate: 후보 pose에서의 compute_lighting() 결과
          - 'E_s': (N,)
          - 'B_s': (N,)
        estimator_state: NBUVEstimator 현재 상태
          - 'rho_ML': (N,) 현재 ML albedo 추정
          - 'Q_ML':   (N,) 현재 누적 신뢰도
        sigma_RN: 판독 잡음 표준편차

    Returns:
        (N,) float64 예상 q_s(t+1)
    """
    E_s     = lighting_candidate['E_s']    # (N,)
    B_s     = lighting_candidate['B_s']    # (N,)
    rho_ML  = estimator_state['rho_ML']    # (N,)

    # Eq.19: q_s = E²_s / (ρ̂^ML · E_s + B_s + σ²_RN)
    denom = rho_ML * E_s + B_s + sigma_RN ** 2
    denom = np.where(denom > 1e-12, denom, 1e-12)

    q_next = (E_s ** 2) / denom
    q_next = np.maximum(q_next, 0.0)

    return q_next


def compute_patch_info_gains(q_next: np.ndarray,
                              Q_ML: np.ndarray,
                              visible_mask: np.ndarray) -> np.ndarray:
    """
    패치별 정보이득을 배열로 반환한다 (시각화/분석용).

    Returns:
        (N,) float64 패치별 정보이득 (비가시 패치 = 0)
    """
    gains = np.zeros(len(Q_ML), dtype=np.float64)

    if visible_mask.sum() == 0:
        return gains

    q_v = q_next[visible_mask]
    Q_v = Q_ML[visible_mask]

    ratio = q_v / np.maximum(Q_v, 1e-12)
    gains[visible_mask] = 0.5 * np.log1p(ratio)

    return gains


def batch_compute_info_gains(q_next_batch: np.ndarray,
                              Q_ML: np.ndarray,
                              visible_masks: np.ndarray) -> np.ndarray:
    """
    M개의 후보 pose에 대해 정보이득을 한 번에 계산한다.
    nbuv_optimizer에서 효율적인 exhaustive search를 위해 사용.

    Args:
        q_next_batch:  (M, N) 후보별 예상 신뢰도
        Q_ML:          (N,) 현재 누적 신뢰도 (모든 후보에 동일)
        visible_masks: (M, N) bool 후보별 가시 마스크

    Returns:
        (M,) float64 후보별 총 정보이득
    """
    M, N = q_next_batch.shape

    # Eq.24/25: ratio = q_next / Q_ML (브로드캐스팅)
    Q_ML_safe = np.maximum(Q_ML, 1e-12)   # (N,)
    ratio     = q_next_batch / Q_ML_safe[np.newaxis, :]   # (M, N)

    # 패치별 정보이득
    patch_gains = 0.5 * np.log1p(ratio)   # (M, N)

    # 비가시 패치 마스킹
    patch_gains = patch_gains * visible_masks.astype(np.float64)   # (M, N)

    # 패치 방향으로 합산 → 후보별 총 정보이득
    total_gains = patch_gains.sum(axis=1)   # (M,)

    return total_gains


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    N = 50
    M = 10   # 후보 수

    # 더미 상태
    Q_ML     = np.random.uniform(0.5, 5.0, N)
    rho_ML   = np.random.uniform(0.2, 0.8, N)

    # 더미 후보 데이터
    q_batch  = np.random.uniform(0.0, 2.0, (M, N))
    vis_masks = np.random.rand(M, N) > 0.3   # ~70% 가시

    # 배치 정보이득 계산
    gains = batch_compute_info_gains(q_batch, Q_ML, vis_masks)
    best  = np.argmax(gains)

    print(f"후보 수:           {M}")
    print(f"패치 수:           {N}")
    print(f"정보이득 범위:     [{gains.min():.3f}, {gains.max():.3f}]")
    print(f"최적 후보 index:   {best}  (gain={gains[best]:.3f})")

    # 단일 후보 계산과 배치 결과 일치 확인
    single = compute_info_gain(q_batch[best], Q_ML, vis_masks[best])
    print(f"단일 계산 일치:    {np.isclose(single, gains[best])}")

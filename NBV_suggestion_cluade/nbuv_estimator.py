"""
nbuv_estimator.py
-----------------
NBUV 논문 Eq.12~20: 최대우도 albedo 추정 및 누적 정보량 관리.

핵심 개념:
  패치 s마다 albedo ρ_s를 반복 관측을 통해 최대우도(ML) 추정한다.
  각 관측의 신뢰도(q_s)를 누적하여 추정 정확도를 높인다.

수식:
  Eq.12: ρ̂_s = (I_s - B_s) / E_s
           단일 관측에서의 albedo 원시 추정

  Eq.17: ρ̂^ML_s = Σ_t(ρ̂_s(t) · q_s(t)) / Σ_t(q_s(t))
           가중 평균 ML 추정량

  Eq.19: q_s(t) = E²_s(t) / (ρ̂^ML_s · E_s(t) + B_s(t) + σ²_RN)
           관측 t의 신뢰도 가중치

  Eq.20: Q^ML_s(t) = Σ_{t'≤t} q_s(t')
           누적 신뢰도 (Fisher information)

초기값:
  Q^ML_s(0) = 1/σ²_max   (미관측 prior: 최대 불확실성)
  ρ̂^ML_s(0) = 0.5        (균일 prior)
"""

import numpy as np


class NBUVEstimator:
    """
    N개의 표면 패치에 대한 albedo ML 추정 상태를 관리한다.

    매 스텝마다 update()를 호출하면
    Q_ML, rho_ML 배열이 갱신된다.
    """

    def __init__(self, N: int, sigma_max: float = 0.5, sigma_RN: float = 0.01):
        """
        Args:
            N:         패치 총 수
            sigma_max: 미관측 패치의 최대 불확실성 (prior σ)
            sigma_RN:  판독 잡음 표준편차
        """
        self.N        = N
        self.sigma_RN = sigma_RN

        # ── 상태 초기화 ───────────────────────────────────────────────────
        # Q^ML[s] = 1/σ²_max  (Eq.20, t=0 초기값)
        self.Q_ML   = np.full(N, 1.0 / (sigma_max ** 2), dtype=np.float64)

        # ρ̂^ML[s] = 0.5  (균일 prior)
        self.rho_ML = np.full(N, 0.5, dtype=np.float64)

        # 가중 albedo 누적합: Σ(ρ̂ · q)
        self._rho_q_sum = self.rho_ML.copy() * self.Q_ML.copy()

        # 관측 횟수 추적
        self.obs_count = np.zeros(N, dtype=np.int32)

        # 스텝 카운터
        self.t = 0

    def update(self, lighting: dict,
               vis: dict,
               rgb_obs: np.ndarray = None) -> dict:
        """
        한 스텝의 관측으로 albedo 추정을 갱신한다.

        Args:
            lighting:  nbuv_lighting.compute_lighting()의 반환값
              - 'E_s':      (N,) 유효 조도
              - 'B_s':      (N,) 후방산란
              - 'sigma2_s': (N,) 측정 분산
            vis:       patch_visibility.compute_visibility()의 반환값
              - 'visible_mask': (N,) bool
            rgb_obs:   (N,) 또는 None. 패치별 관측 복사 휘도.
                       None이면 현재 rho_ML로 시뮬레이션 I_s를 생성.

        Returns:
            dict:
              'rho_ML':  (N,) 갱신된 ML albedo 추정
              'Q_ML':    (N,) 갱신된 누적 Fisher information
              'q_s':     (N,) 이번 스텝 관측 신뢰도
        """
        mask = vis['visible_mask']   # (N,)
        E_s  = lighting['E_s']       # (N,)
        B_s  = lighting['B_s']       # (N,)

        q_s = np.zeros(self.N, dtype=np.float64)

        if mask.sum() == 0:
            self.t += 1
            return self._make_result(q_s)

        # ── 가시 패치만 처리 ───────────────────────────────────────────────
        E_v   = E_s[mask]
        B_v   = B_s[mask]
        rho_v = self.rho_ML[mask]

        # 관측 I_s 준비
        if rgb_obs is not None:
            I_v = rgb_obs[mask]
        else:
            # rgb_obs가 없으면 현재 추정값으로 합성 (Eq.5: I_s = ρ_s·E_s + B_s + n)
            noise = np.random.normal(0, self.sigma_RN, E_v.shape)
            I_v   = rho_v * E_v + B_v + noise

        # ── Eq.12: ρ̂_s = (I_s - B_s) / E_s ─────────────────────────────
        E_safe  = np.where(E_v > 1e-12, E_v, 1e-12)
        rho_hat = (I_v - B_v) / E_safe
        rho_hat = np.clip(rho_hat, 0.0, 1.0)   # albedo ∈ [0, 1]

        # ── Eq.19: q_s = E²_s / (ρ̂^ML · E_s + B_s + σ²_RN) ─────────────
        denom = rho_v * E_v + B_v + self.sigma_RN ** 2
        denom = np.where(denom > 1e-12, denom, 1e-12)
        q_v   = (E_v ** 2) / denom
        q_v   = np.maximum(q_v, 0.0)

        # ── Eq.20: Q^ML_s(t) += q_s(t) ───────────────────────────────────
        self.Q_ML[mask] += q_v

        # ── Eq.17: ρ̂^ML = Σ(ρ̂·q) / Q^ML ────────────────────────────────
        self._rho_q_sum[mask] += rho_hat * q_v
        self.rho_ML[mask] = self._rho_q_sum[mask] / self.Q_ML[mask]
        self.rho_ML = np.clip(self.rho_ML, 0.0, 1.0)

        # 신뢰도 배열 전체에 삽입
        q_s[mask] = q_v

        # 관측 횟수 증가
        self.obs_count[mask] += 1
        self.t += 1

        return self._make_result(q_s)

    def _make_result(self, q_s: np.ndarray) -> dict:
        return {
            'rho_ML': self.rho_ML.copy(),   # (N,)
            'Q_ML':   self.Q_ML.copy(),     # (N,)
            'q_s':    q_s,                  # (N,)
        }

    def get_uncertainty(self) -> np.ndarray:
        """
        패치별 현재 추정 불확실성을 반환한다.
        σ²_est = 1 / Q^ML

        Returns:
            (N,) float64 — 낮을수록 잘 관측된 패치
        """
        return 1.0 / np.maximum(self.Q_ML, 1e-12)

    def get_observed_fraction(self) -> float:
        """최소 1회 이상 관측된 패치의 비율을 반환한다."""
        return (self.obs_count > 0).sum() / self.N

    def reset(self, sigma_max: float = 0.5):
        """상태를 초기값으로 리셋한다."""
        self.Q_ML      = np.full(self.N, 1.0 / (sigma_max ** 2), dtype=np.float64)
        self.rho_ML    = np.full(self.N, 0.5, dtype=np.float64)
        self._rho_q_sum = self.rho_ML.copy() * self.Q_ML.copy()
        self.obs_count  = np.zeros(self.N, dtype=np.int32)
        self.t          = 0


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, json, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from models.mesh_builder import make_test_plane
    from models.patch_visibility import compute_visibility
    from models.nbuv_lighting import compute_lighting

    patches = make_test_plane(nx=5, ny=5)
    N = patches['N']

    cam_pos   = np.array([0.0, 0.0, 1.0])
    light_pos = np.array([0.1, 0.0, 0.8])

    config_path = os.path.join(os.path.dirname(__file__),
                               '..', 'config', 'water_params.json')
    with open(config_path) as f:
        raw = json.load(f)
    params = {k: v for k, v in raw.items() if not k.startswith('_')}

    estimator = NBUVEstimator(N=N, sigma_max=params.get('sigma_max', 0.5),
                               sigma_RN=params['sigma_RN'])

    # 5스텝 시뮬레이션
    for step in range(5):
        vis      = compute_visibility(patches, cam_pos, light_pos)
        lighting = compute_lighting(vis, params)
        result   = estimator.update(lighting, vis)

    print(f"스텝 수:           {estimator.t}")
    print(f"관측 패치 비율:    {estimator.get_observed_fraction()*100:.1f}%")
    print(f"rho_ML 평균:       {result['rho_ML'].mean():.4f}")
    print(f"Q_ML 평균:         {result['Q_ML'].mean():.4f}")
    print(f"불확실성 평균:     {estimator.get_uncertainty().mean():.4f}")

"""
nbuv_lighting.py
----------------
NBUV 논문 Eq.2~7의 수중 조명 모델을 구현한다.

수식 참조:
  Eq.2:  D_s ∝ C₀ · exp(-β · l_LS) / l_LS²
           직접 조명 복사 조도 (Distance falloff + 감쇠)

  Eq.3:  E_s = D_s · (n̂_s · L̂_s)
           패치 s의 유효 조도 (입사각 반영)

  Eq.4:  B_s = B∞ · (1 - exp(-β · l_SC))
           패치까지의 후방산란 (배경 산란광)

  Eq.5:  I_s = ρ_s · E_s + B_s + n_I
           카메라가 관측하는 복사 휘도

  Eq.7:  SNR_s ≈ √(ρ·E_s) / √(ρ·E_s + B_s + σ²_RN)
           신호 대 잡음비 (측정 불확실성 계산의 기반)

모든 계산은 가시 패치(visible_mask=True)에만 적용되며,
비가시 패치는 0 또는 NaN으로 설정된다.
"""

import numpy as np


def compute_lighting(vis: dict,
                     params: dict) -> dict:
    """
    가시성 정보와 수중 파라미터로부터 조명 물리량을 계산한다.

    Args:
        vis: patch_visibility.compute_visibility()의 반환값
          - 'visible_mask': (N,) bool
          - 'l_SC':         (N,) 카메라-패치 거리 (m)
          - 'l_LS':         (N,) 조명-패치 거리 (m)
          - 'cos_light':    (N,) 입사각 cos

        params: water_params.json에서 로드된 dict
          - 'beta':    수중 감쇠 계수
          - 'C0':      조명 광원 강도
          - 'B_inf':   원거리 후방산란 (B∞)
          - 'sigma_RN': 판독 잡음 표준편차

    Returns:
        dict:
          'D_s':   (N,) 직접 조명 복사 조도 (Eq.2)
          'E_s':   (N,) 유효 조도 (Eq.3)
          'B_s':   (N,) 후방산란 (Eq.4)
          'SNR_s': (N,) 신호대잡음비 (Eq.7)
          'sigma2_s': (N,) 측정 분산 (SNR 역수 기반)
    """
    mask      = vis['visible_mask']   # (N,) bool
    l_SC      = vis['l_SC']           # (N,)
    l_LS      = vis['l_LS']           # (N,)
    cos_light = vis['cos_light']      # (N,)

    beta     = params['beta']
    C0       = params['C0']
    B_inf    = params['B_inf']
    sigma_RN = params['sigma_RN']

    N = len(mask)

    # 결과 배열 초기화 (비가시 패치 = 0)
    D_s     = np.zeros(N, dtype=np.float64)
    E_s     = np.zeros(N, dtype=np.float64)
    B_s     = np.zeros(N, dtype=np.float64)
    SNR_s   = np.zeros(N, dtype=np.float64)
    sigma2_s = np.full(N, np.inf, dtype=np.float64)  # 비가시: 무한 분산

    if mask.sum() == 0:
        return {'D_s': D_s, 'E_s': E_s, 'B_s': B_s,
                'SNR_s': SNR_s, 'sigma2_s': sigma2_s}

    # 가시 패치에 대해서만 계산
    l_sc_v  = l_SC[mask]
    l_ls_v  = l_LS[mask]
    cos_v   = cos_light[mask]

    # ── Eq.2: D_s ∝ C₀ · exp(-β · l_LS) / l_LS² ─────────────────────────
    l_ls_safe = np.maximum(l_ls_v, 1e-6)   # 분모 0 방지
    D_s_v = C0 * np.exp(-beta * l_ls_safe) / (l_ls_safe ** 2)

    # ── Eq.3: E_s = D_s · cos(입사각) ────────────────────────────────────
    E_s_v = D_s_v * cos_v
    E_s_v = np.maximum(E_s_v, 0.0)   # 음수 방지

    # ── Eq.4: B_s = B∞ · (1 - exp(-β · l_SC)) ───────────────────────────
    B_s_v = B_inf * (1.0 - np.exp(-beta * l_sc_v))

    # ── Eq.7: SNR_s ≈ √(ρ·E_s) / √(ρ·E_s + B_s + σ²_RN) ───────────────
    # ρ (albedo)는 현재 추정값 없이 SNR 계산 → ρ=1 기준 (상대적 SNR)
    rho_ref = 1.0
    numerator   = np.sqrt(rho_ref * E_s_v)
    denominator = np.sqrt(rho_ref * E_s_v + B_s_v + sigma_RN ** 2)
    SNR_s_v     = np.where(denominator > 1e-12,
                            numerator / denominator,
                            0.0)

    # ── 측정 분산: σ²_s = 1/SNR²  (정보량 계산에서 사용) ─────────────────
    # NBUV Eq.19: q_s = SNR²_s / ρ  →  여기서는 SNR² 저장
    snr2_v    = SNR_s_v ** 2
    sigma2_v  = np.where(snr2_v > 1e-12, 1.0 / snr2_v, np.inf)

    # 결과 배열에 삽입
    D_s[mask]     = D_s_v
    E_s[mask]     = E_s_v
    B_s[mask]     = B_s_v
    SNR_s[mask]   = SNR_s_v
    sigma2_s[mask] = sigma2_v

    return {
        'D_s':      D_s,       # (N,)
        'E_s':      E_s,       # (N,)
        'B_s':      B_s,       # (N,)
        'SNR_s':    SNR_s,     # (N,)
        'sigma2_s': sigma2_s,  # (N,)
    }


def compute_irradiance_map(vis: dict,
                            params: dict) -> np.ndarray:
    """
    E_s만 빠르게 반환하는 편의 함수.
    candidate_generator에서 빠른 스크리닝에 사용.

    Returns:
        (N,) float64
    """
    result = compute_lighting(vis, params)
    return result['E_s']


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, json, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from models.mesh_builder import make_test_plane
    from models.patch_visibility import compute_visibility

    patches = make_test_plane(nx=5, ny=5)
    cam_pos   = np.array([0.0, 0.0, 1.0])
    light_pos = np.array([0.1, 0.0, 0.8])

    vis = compute_visibility(patches, cam_pos, light_pos)

    config_path = os.path.join(os.path.dirname(__file__),
                               '..', 'config', 'water_params.json')
    with open(config_path) as f:
        raw = json.load(f)
    params = {k: v for k, v in raw.items() if not k.startswith('_')}

    lighting = compute_lighting(vis, params)

    n_vis = vis['visible_mask'].sum()
    print(f"가시 패치: {n_vis} / {patches['N']}")
    print(f"D_s  (가시): mean={lighting['D_s'][vis['visible_mask']].mean():.4f}")
    print(f"E_s  (가시): mean={lighting['E_s'][vis['visible_mask']].mean():.4f}")
    print(f"B_s  (가시): mean={lighting['B_s'][vis['visible_mask']].mean():.4f}")
    print(f"SNR_s(가시): mean={lighting['SNR_s'][vis['visible_mask']].mean():.4f}")

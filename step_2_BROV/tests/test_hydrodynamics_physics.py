"""Fossen 유체동역학의 **물리 법칙** 회귀 시험.

계수값이나 부호 규약을 자기 자신과 비교하면 오류를 영원히 못 잡는다. 여기서는
규약과 무관한 물리적 성질만 검사한다 — 2026-08-28에 Coriolis 힘 항의 부호
오류를 잡아낸 것이 이 방식이다.
"""

import math

import pytest
import torch

pytest.importorskip("isaacsim", reason="IsaacLab 컨테이너에서만 실행")
try:
    import isaaclab.utils.math as mu
except ModuleNotFoundError:
    from isaacsim import SimulationApp
    _APP = SimulationApp({"headless": True})
    import isaaclab.utils.math as mu

from robots.dynamics.fossen import Hydrodynamics


def _h(**kw):
    return Hydrodynamics(num_envs=1, dt=0.01, device="cpu", **kw)


def test_coriolis_does_no_work():
    """C_A(ν)는 스큐대칭이어야 한다 — ν·(C_A ν) = 0.

    아니면 Coriolis가 에너지를 만들거나 없앤다. 2026-08-28까지 힘 항의 부호가
    반대라 이 값이 30.07이었다.
    """
    h = _h()
    g = torch.Generator().manual_seed(0)
    worst = max(
        abs(float((v * h._coriolis(v)).sum()))
        for v in (torch.randn(1, 6, generator=g) * 0.5 for _ in range(300))
    )
    assert worst < 1e-5, f"ν·(C_A ν) = {worst:.3e} — Coriolis가 일을 한다"


def test_coriolis_pushes_the_vehicle_outward_in_a_turn():
    """전진 + 선회에서 부가질량 반작용은 선회 **바깥**을 향해야 한다.

    부가질량 운동량 p = A₁₁ν₁ 를 회전시키는 힘이 ω × p 이고, 반작용으로 물이
    기체를 미는 힘은 −(ω × p) — 원심 반작용이다. 스큐대칭만으로는 전체 부호가
    정해지지 않으므로(±C 둘 다 스큐) 이 시험이 따로 필요하다.
    """
    h = _h()
    u, r = 0.5, 0.5
    v = torch.zeros(1, 6)
    v[0, 0], v[0, 5] = u, r                     # SNAME/NED: x전방, +r = 우선회
    applied = -h._coriolis(v)[0, :3]            # 모듈은 −C_A·ν 를 적용한다
    # +y가 우현(선회 안쪽) → 바깥으로 밀리려면 y성분이 음수여야 한다
    assert float(applied[1]) < 0.0, f"선회 안쪽으로 당긴다: {applied.tolist()}"
    expected = h._Ma[0, 0, 0] * u * r
    assert abs(abs(float(applied[1])) - float(expected)) < 1e-4


def test_damping_dissipates_energy():
    h = _h()
    g = torch.Generator().manual_seed(1)
    worst = min(
        float((v * h._damping(v)).sum())
        for v in (torch.randn(1, 6, generator=g) * 0.5 for _ in range(300))
    )
    assert worst >= -1e-9, f"감쇠가 에너지를 만든다: {worst}"


@pytest.mark.parametrize("axis,name", [(0, "roll"), (1, "pitch")])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_restoring_moment_rights_the_vehicle(axis, name, sign):
    """CoB가 CoM 위에 있으므로 기울이면 되돌리는 모멘트가 나와야 한다."""
    h = _h()
    e = [torch.zeros(1)] * 3
    e[axis] = torch.tensor([sign * 0.2])
    q = mu.quat_from_euler_xyz(*e)
    t_zup = h._buoyancy(q)[0, 3:] * torch.tensor([1.0, -1.0, -1.0])
    assert float(t_zup[axis]) * (-sign) > 0.0, f"{name}: 기울임을 키운다"


def test_added_mass_survives_at_high_frequency():
    """유효질량이 전 대역에서 M_RB + M_A 여야 한다.

    2026-08-28까지 가속도에 저역필터(α=0.3, ~5.7 Hz)를 걸어 명시적 적분을
    안정화했는데, 그 필터가 added mass를 주파수 의존으로 만들었다 — 10 Hz에서
    유효질량이 18.74 kg(M_RB=14.635에 근접, added mass가 사라진 값)이었다.
    지금은 ν̇ 를 풀어서 구하므로 필터가 없다.
    """
    M, DT = 14.635, 0.01
    h = _h()
    target = M + h._Ma[0, 2, 2].item()

    def effective_mass(f0):
        n, v, hist = int(20 / f0 / DT), 0.0, []
        for k in range(n):
            F = torch.zeros(1, 6)
            F[0, 2] = 25.0 * math.sin(2 * math.pi * f0 * k * DT)
            vel = torch.zeros(1, 3)
            vel[0, 2] = v
            fh, _ = h.compute(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), vel,
                              torch.zeros(1, 3), F)
            v += DT * (float(F[0, 2]) + float(fh[0, 2])) / M
            hist.append(v)
        seg = hist[-int(5 / f0 / DT):]
        amp = (max(seg) - min(seg)) / 2
        return 25.0 / (amp * 2 * math.pi * f0)

    for f0 in (5.0, 10.0):
        m_eff = effective_mass(f0)
        assert abs(m_eff - target) / target < 0.10, \
            f"{f0} Hz 유효질량 {m_eff:.2f} vs {target:.2f} kg"


def test_net_buoyancy_is_slightly_positive_with_the_yaml_parameters():
    """정본 파라미터에서 기체는 **약한 양성부력**이어야 한다.

    ROV는 관례적으로 동력 상실 시 부상하도록 만든다. volume은 그 목표
    (mass + 0.03 kgf)로 역산된 값이라, 밀도를 바꾸면 부호가 뒤집힌다 —
    클래스 fallback(_WATER_DENSITY=997)과 짝지으면 -0.137 N이 된다.
    실제 경로는 YAML의 1000.0을 넘긴다.
    """
    h = _h(volume=0.014665, water_density=1000.0)
    net = float(h._buoy_mag[0]) - 14.635 * 9.81
    assert 0.0 < net < 1.0, f"순부력 {net:+.4f} N"

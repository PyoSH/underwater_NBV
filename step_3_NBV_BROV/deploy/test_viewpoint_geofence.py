"""viewpoint_geofence 단위시험. Phase 1 통과 기준의 일부.

기대값은 전부 이 모델에서 계산해 **측정으로 고정한** 값이다 (DEPLOY_3WEEK_PLAN §12.0).
바뀌면 기하 가정이 바뀐 것이므로 문서와 함께 갱신할 것.
"""
import math

import pytest

from viewpoint_geofence import (
    PoolBox,
    clearance_to_object,
    is_valid,
    project,
    psi_max_surface,
    psi_min_floor,
    psi_min_object,
    psi_min_safe,
    view_position,
)

PHI_MIN, PHI_MAX = math.radians(10), math.radians(80)
PSI_MIN, PSI_MAX = 1.4, 2.0
DEGREES = list(range(10, 81))


def test_clearance_is_zero_inside_and_grows_outside():
    assert clearance_to_object(0.0, 0.5) == 0.0            # 물체 내부
    assert clearance_to_object(0.0, 1.2) == pytest.approx(1.2 - 0.933)
    assert clearance_to_object(1.5, 0.5) == pytest.approx(1.5 - 0.801)
    # 상단 모서리 바깥은 대각 거리 — OR 근사보다 관대한 것이 정확한 기하다
    d = clearance_to_object(0.801 + 0.3, 0.933 + 0.4)
    assert d == pytest.approx(math.hypot(0.3, 0.4))


def test_clearance_monotone_along_ray():
    for deg in DEGREES:
        phi = math.radians(deg)
        prev = -1.0
        for psi in [0.1 * k for k in range(1, 41)]:
            d = clearance_to_object(psi * math.sin(phi), psi * math.cos(phi))
            assert d >= prev - 1e-12
            prev = d


def test_psi_min_safe_peak_fits_inside_psi_max():
    """최악값 1.608 m (phi=40deg 부근) < psi_max 2.0 -> 사영은 항상 해가 있다."""
    grid = {d: psi_min_safe(math.radians(d)) for d in DEGREES}
    peak_deg = max(grid, key=grid.get)
    assert peak_deg == 41
    assert grid[peak_deg] == pytest.approx(1.609, abs=2e-3)
    assert grid[peak_deg] < PSI_MAX


def test_band_where_psi_min_1p4_is_unsafe():
    """psi_min=1.4 위반은 phi 21~57도(물체)와 79~80도(바닥) 두 구간."""
    bad = [d for d in DEGREES if psi_min_safe(math.radians(d)) > PSI_MIN]
    runs, start, prev = [], bad[0], bad[0]
    for d in bad[1:]:
        if d != prev + 1:
            runs.append((start, prev))
            start = d
        prev = d
    runs.append((start, prev))
    assert runs == [(21, 57), (79, 80)]


def test_high_phi_is_floor_limited_not_object_limited():
    """phi=80deg(pitch -10deg)에서는 바닥이 구속이다 — psi >= 1.525."""
    phi = math.radians(80)
    assert psi_min_floor(phi) > psi_min_object(phi)
    assert psi_min_safe(phi) == pytest.approx(1.525, abs=2e-3)


def test_surface_never_binds_in_this_pool():
    """6x10x3 수조에서 수면 여유는 시점 상자를 구속하지 않는다."""
    assert min(psi_max_surface(math.radians(d)) for d in DEGREES) > PSI_MAX


def test_walls_never_bind_in_this_pool():
    pool = PoolBox()
    for deg in DEGREES:
        x, y, _ = view_position(0.0, math.radians(deg), PSI_MAX)
        assert math.hypot(x, y) <= min(pool.x_abs_max, pool.y_abs_max)


def test_inside_envelope_is_rejected_and_projected_out():
    phi = math.radians(40)
    assert not is_valid(0.0, phi, PSI_MIN)[0]
    _, _, psi, reason = project(0.0, phi, PSI_MIN, PSI_MIN, PSI_MAX)
    assert reason == "object_envelope"
    assert psi == pytest.approx(psi_min_safe(phi), abs=1e-3)
    assert is_valid(0.0, phi, psi)[0]


def test_floor_violation_names_floor():
    phi = math.radians(80)
    _, _, psi, reason = project(0.0, phi, PSI_MIN, PSI_MIN, PSI_MAX)
    assert reason == "pool_floor"
    assert is_valid(0.0, phi, psi)[0]


def test_projection_preserves_direction():
    th, ph = 1.234, math.radians(40)
    th2, ph2, _, _ = project(th, ph, PSI_MIN, PSI_MIN, PSI_MAX)
    assert th2 == th
    assert ph2 == ph


def test_valid_view_is_untouched():
    th, ph, ps = 0.5, math.radians(65), 1.8
    assert is_valid(th, ph, ps)[0]
    _, _, ps2, reason = project(th, ph, ps, PSI_MIN, PSI_MAX)
    assert reason == ""
    assert ps2 == pytest.approx(ps)


def test_projection_always_yields_a_valid_view():
    """전 (theta, phi) x 여러 psi 에서 사영 결과가 반드시 유효해야 한다."""
    for deg in DEGREES:
        phi = math.radians(deg)
        for t in range(0, 360, 15):
            theta = math.radians(t)
            for psi in (PSI_MIN, 1.6, PSI_MAX):
                _, _, psi_new, _ = project(theta, phi, psi, PSI_MIN, PSI_MAX)
                ok, why = is_valid(theta, phi, psi_new)
                assert ok, f"phi={deg} theta={t} psi={psi}->{psi_new:.3f}: {why}"

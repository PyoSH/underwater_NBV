"""a50_model 단위시험. 데이터시트 값이 실제로 반영됐는지 고정한다."""
import math

import numpy as np
import pytest

from a50_model import (BEAM_ANGLE_DEG, MIN_ALTITUDE_M, A50Config, A50State,
                       beam_directions, bottom_lock, dr_step, figure_of_merit,
                       measured_body_velocity)


def R_pitch(deg):
    """FLU 에서 기수 하향 pitch. look_at 은 phi 에 대해 pitch = -(90-phi)."""
    p = math.radians(deg); c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def test_beam_geometry_matches_datasheet():
    """수평 자세: 4빔 전부 수직에서 22.5 deg."""
    for d in beam_directions(np.eye(3)):
        assert math.degrees(math.acos(-d[2])) == pytest.approx(BEAM_ANGLE_DEG, abs=1e-6)


def test_one_beam_crosses_horizon_but_three_survive():
    """정정(2026-09-10): 빔 하나가 수평을 넘어도 **3빔이 남아 해가 나온다**.

    22.5 deg 콘이므로 pitch > 67.5 deg 에서 최상단 빔이 수평을 넘는다. 그러나
    4-beam Janus 는 3빔이면 3-D 속도 해가 나오므로 lock 은 유지된다.
    "phi<=20 에서 바닥 반사가 없다" 는 초기 서술은 틀렸다.
    """
    cfg = A50Config(max_beam_from_vertical_deg=None)
    for pitch, beams in ((60, 4), (70, 3), (80, 3)):
        ok, n, _ = bottom_lock(np.array([0., 0., 1.5]), R_pitch(pitch), 0.0, cfg)
        assert n == beams, f"pitch={pitch} -> beams={n}"
        assert ok is True


def test_grazing_threshold_is_the_unmeasured_knob():
    """스치는 각 한계를 주면 그때 비로소 무효가 된다 — 그 값은 사양에 없다."""
    strict = A50Config(max_beam_from_vertical_deg=60.0)
    ok60, n60, _ = bottom_lock(np.array([0., 0., 1.5]), R_pitch(60), 0.0, strict)
    ok30, n30, _ = bottom_lock(np.array([0., 0., 1.5]), R_pitch(30), 0.0, strict)
    assert ok30 is True and n30 == 4
    assert ok60 is False and n60 < 3


def test_min_altitude_from_datasheet():
    cfg = A50Config()
    assert bottom_lock(np.array([0., 0., MIN_ALTITUDE_M - 0.001]), np.eye(3), 0.0, cfg)[0] is False
    assert bottom_lock(np.array([0., 0., MIN_ALTITUDE_M + 0.001]), np.eye(3), 0.0, cfg)[0] is True


def test_lever_arm_dominates_during_rotation():
    """|r|=0.285 m -> |w|=0.2 rad/s 에서 겉보기 속도가 순항의 20 % 를 넘는다."""
    cfg = A50Config()
    rng = np.random.default_rng(0)
    zero_noise = A50Config(velocity_noise_std=0.0)
    omega = np.array([0., 0.2, 0.])
    v = measured_body_velocity(np.zeros(3), omega, zero_noise, rng)
    # |w x r| 이지 |w||r| 이 아니다 — w 와 r 가 나란한 성분은 기여하지 않는다
    assert np.linalg.norm(v) == pytest.approx(
        np.linalg.norm(np.cross(omega, np.asarray(cfg.lever_arm_m))), rel=1e-9)
    assert np.linalg.norm(v) > 0.2 * 0.25          # 순항 0.25 m/s 의 20 % 초과


def test_yaw_drift_rate_matches_docs():
    """0.2 deg/min 이 5 분 뒤 1.0 deg 가 된다."""
    cfg = A50Config(yaw_drift_deg_per_min=0.2, velocity_noise_std=0.0)
    st = A50State(); st.reset(0.0, np.random.default_rng(0), cfg)
    for k in range(1, 301):
        dr_step(st, float(k), np.zeros(3), 0.0, np.eye(3), True, cfg)
    assert math.degrees(st.yaw_bias_rad) == pytest.approx(1.0, abs=1e-6)


def test_scale_error_within_long_term_accuracy():
    """장기 정확도 ±1.01 % 안의 스케일 실현값만 나온다."""
    cfg = A50Config(long_term_accuracy=0.0101)
    rng = np.random.default_rng(7)
    for _ in range(200):
        st = A50State(); st.reset(0.0, rng, cfg)
        assert abs(st.scale_err) <= cfg.long_term_accuracy + 1e-12


def test_straight_run_error_is_scale_plus_drift():
    """직진 40 m: 오차가 스케일항과 yaw 드리프트항의 합으로 설명된다."""
    cfg = A50Config(long_term_accuracy=0.0, yaw_drift_deg_per_min=0.3,
                    velocity_noise_std=0.0, lever_arm_m=(0., 0., 0.))
    st = A50State(); st.reset(0.0, np.random.default_rng(0), cfg)
    v = np.array([0.25, 0., 0.]); dt = 0.1
    T = 160.0                                     # 40 m
    t = 0.0
    while t < T:
        t += dt
        dr_step(st, t, v, 0.0, np.eye(3), True, cfg)
    # 스케일 0 이므로 진행거리는 맞고, 교차오차만 남는다
    assert st.p_dr[0] == pytest.approx(40.0, rel=1e-3)
    expected = math.radians(0.3 * (T / 60.0)) * 40.0 / 2.0
    assert st.p_dr[1] == pytest.approx(expected, rel=0.05)


def test_invalid_stops_integration_and_grows_fom():
    cfg = A50Config(velocity_noise_std=0.0)
    st = A50State(); st.reset(0.0, np.random.default_rng(0), cfg)
    v = np.array([0.25, 0., 0.])
    for k in range(1, 11):
        dr_step(st, k * 0.1, v, 0.0, np.eye(3), True, cfg)
    p_before = st.p_dr.copy()
    for k in range(11, 111):
        dr_step(st, k * 0.1, v, 0.0, np.eye(3), False, cfg)
    assert np.allclose(st.p_dr, p_before)                    # 얼지만 조용하지 않다
    f_ok = figure_of_merit(st, 1.0, True, 1.0, cfg)
    f_gap = figure_of_merit(st, 11.0, False, 1.0, cfg)
    assert f_gap > f_ok * 3

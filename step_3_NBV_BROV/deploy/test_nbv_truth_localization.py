"""^pool T_odom 고정 변환이 brov_base 의 실제 프레임 계약과 일치하는지 (컨테이너에서 실행)."""
import copy
import math

import numpy as np
import pytest
import torch

from brov_base.gazebo_truth import GazeboTruthBuffer
from brov_base.odometry import ned_frd_to_odom_flu  # noqa: F401  (존재 확인)

from nbv_truth_localization import PoolFromOdom, quat_angle_deg, quat_mul_xyzw, rot_from_xyzw


def _odom_via_brov_base(p_gz, q_gz_xyzw):
    """gz Odometry -> GazeboTruthBuffer(NED/FRD) -> ned_frd_to_odom_flu 로 얻은 odom pose."""
    from nav_msgs.msg import Odometry
    m = Odometry()
    m.header.frame_id, m.child_frame_id = "odom", "base_link"
    m.header.stamp.sec = 10
    m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = p_gz
    (m.pose.pose.orientation.x, m.pose.pose.orientation.y,
     m.pose.pose.orientation.z, m.pose.pose.orientation.w) = q_gz_xyzw
    buf = GazeboTruthBuffer()
    # 버퍼는 각속도 proxy 를 위해 두 표본이 필요하다 — 같은 pose 를 50 ms 뒤에 한 번 더
    buf.update(m)
    m2 = copy.deepcopy(m)
    m2.header.stamp.nanosec = 50_000_000
    buf.update(m2)
    s = buf.snapshot()
    assert s is not None, "GazeboTruthBuffer.snapshot() is None after two samples"
    conv = ned_frd_to_odom_flu(s["pos_ned"], s["att_quat_ned"], s["vel_ned"], s["body_rates_ned"])
    return conv.position_odom.numpy(), tuple(float(v) for v in conv.orientation_xyzw)


@pytest.mark.parametrize("p_gz", [(1.0, 2.0, -2.0), (-3.2, 0.4, -0.2), (0.0, 0.0, -2.7)])
def test_position_round_trip_matches_brov_base_contract(p_gz):
    xf = PoolFromOdom(-2.7)
    p_odom, _ = _odom_via_brov_base(p_gz, (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(p_odom, PoolFromOdom.odom_from_gz(p_gz), atol=1e-6)
    assert np.allclose(xf.position(p_odom), np.array(p_gz) + [0, 0, 2.7], atol=1e-5)


@pytest.mark.parametrize("rpy_deg", [(0, 0, 0), (0, -80, 0), (10, -30, 120), (-15, 60, -90)])
def test_orientation_round_trip(rpy_deg):
    r, p, y = (math.radians(v) for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r/2), math.sin(r/2), math.cos(p/2), math.sin(p/2), math.cos(y/2), math.sin(y/2)
    q_gz = (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)
    _, q_odom = _odom_via_brov_base((0.5, 0.5, -1.0), q_gz)
    q_pool = PoolFromOdom(-2.7).orientation(q_odom)
    assert quat_angle_deg(q_pool, q_gz) < 0.05   # brov_base 변환은 float32


def test_rotation_helpers_consistent():
    q = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    R = rot_from_xyzw(q)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)
    assert quat_mul_xyzw(q, (0, 0, 0, 1)) == pytest.approx(q)

"""Gazebo truth 기반 pool localization (SIMULATION ONLY, Phase 4 ②, 2026-09-10).

실기에서는 `brov_localization/pool_alignment_node` 가 AprilTag 관측과 정지 상태의 odometry
를 짝지어 `pool -> odom` 을 **한 번** 풀고 동결한다. SITL 에서 그 자리를 이 노드가 맡는다:
pool 프레임이 gz world 와 정의상 같은 축(원점만 물체 바닥면, gz (0,0,-2.7))이므로
정렬은 풀 것이 없고 **해석적으로 고정**이다.

프레임 규약 (brov_base 계약 그대로):
  gz world  : ENU, 수면 z=0
  NED       : (n,e,d) = (y_gz, x_gz, -z_gz)                 (gazebo_truth.py)
  odom      : diag(1,-1,-1) * NED = (y_gz, -x_gz, z_gz)      (ned_frd_to_odom_flu)
  pool      : gz + (0,0,2.7)                                  (물체 바닥면 = 원점)
  => ^pool T_odom : R = R_z(+90 deg), t = (0, 0, 2.7)
     p_pool = R p_odom + t   (검산: p_odom=(y,-x,z) -> R p_odom = (x,y,z) = p_gz)

자기검증: obs_node 가 발행하는 odom pose(truth 운동학, `allow_gazebo_truth_pool_missions`)
를 위 변환으로 pool 로 보낸 값과 **gz truth 를 직접 pool 로 보낸 값**을 매 표본 비교한다.
어긋나면(규약이 바뀌었거나 obs_node 가 EKF 로 되돌아갔거나) INVALID 로 떨어져
mission manager / obs_node 의 게이트가 닫힌다 — 조용히 틀린 프레임으로 가지 않는다.

발행 (pool_alignment_node 와 같은 토픽·메시지):
  /brov/localization/status                        LocalizationStatus (latched, 5 Hz)
  /brov/localization/valid                         Bool
  /brov/localization/odometry_pool                 Odometry  (pool -> base_link)
  /brov/localization/odometry_pool_with_alignment  AlignedOdometry
구독:
  /brov/odometry/local_with_session   OdometrySession (obs_node)
  /brov/sim/gazebo_odometry_raw       Odometry (gz truth, ENU)   — 자기검증용
"""
from __future__ import annotations

import math
import time
import uuid

import numpy as np
import rclpy
from brov_interfaces.msg import AlignedOdometry, LocalizationStatus, OdometrySession
from geometry_msgs.msg import Transform
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from std_msgs.msg import Bool


def quat_mul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def rot_from_xyzw(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_angle_deg(a, b) -> float:
    d = abs(sum(float(u) * float(v) for u, v in zip(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


class PoolFromOdom:
    """^pool T_odom = (R_z(+90deg), (0,0,-pool_origin_gz_z)). ROS-독립, 시험 가능."""

    def __init__(self, pool_origin_gz_z: float = -2.7):
        s = math.sqrt(0.5)
        self.q_xyzw = (0.0, 0.0, s, s)
        self.t = np.array([0.0, 0.0, -pool_origin_gz_z])
        self.R = rot_from_xyzw(self.q_xyzw)
        self.pool_origin_gz_z = pool_origin_gz_z

    def position(self, p_odom) -> np.ndarray:
        return self.R @ np.asarray(p_odom, dtype=float) + self.t

    def orientation(self, q_odom_base_xyzw):
        return quat_mul_xyzw(self.q_xyzw, q_odom_base_xyzw)

    def pool_from_gz(self, p_gz) -> np.ndarray:
        return np.asarray(p_gz, dtype=float) - np.array([0.0, 0.0, self.pool_origin_gz_z])

    @staticmethod
    def odom_from_gz(p_gz) -> np.ndarray:
        """규약 재현: odom = diag(1,-1,-1) * NED(ENU). 시험과 자기검증에서만 쓴다."""
        x, y, z = (float(v) for v in p_gz)
        return np.array([y, -x, z])


class TruthLocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__("nbv_truth_localization")
        p = self.declare_parameter
        p("pool_frame", "pool")
        p("odom_frame", "odom")
        p("base_frame", "base_link")
        p("pool_origin_gz_z", -2.7)
        p("truth_topic", "/brov/sim/gazebo_odometry_raw")
        p("odom_session_topic", "/brov/odometry/local_with_session")
        p("max_position_mismatch_m", 0.05)
        p("max_rotation_mismatch_deg", 3.0)
        p("status_period_s", 0.2)
        p("odom_stale_s", 1.0)
        g = lambda n: self.get_parameter(n).value
        self._pool = str(g("pool_frame"))
        self._odom = str(g("odom_frame"))
        self._base = str(g("base_frame"))
        self._xf = PoolFromOdom(float(g("pool_origin_gz_z")))
        self._tol_p = float(g("max_position_mismatch_m"))
        self._tol_r = float(g("max_rotation_mismatch_deg"))
        self._stale = float(g("odom_stale_s"))

        self._session = ""
        self._alignment_id = ""
        self._epoch = 0
        self._state = LocalizationStatus.UNINITIALIZED
        self._reason = "waiting for local odometry session"
        self._last_odom_t = None
        self._last_truth = None      # (t_mono, p_gz, q_gz)
        self._mismatch = None
        self._samples = 0

        latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub_status = self.create_publisher(LocalizationStatus,
                                                 "/brov/localization/status", latched)
        self._pub_valid = self.create_publisher(Bool, "/brov/localization/valid", 10)
        self._pub_pool = self.create_publisher(Odometry,
                                               "/brov/localization/odometry_pool", 10)
        self._pub_aligned = self.create_publisher(
            AlignedOdometry, "/brov/localization/odometry_pool_with_alignment", 10)
        self.create_subscription(OdometrySession, str(g("odom_session_topic")),
                                 self._on_session, 20)
        self.create_subscription(Odometry, str(g("truth_topic")), self._on_truth,
                                 qos_profile_sensor_data)
        self.create_timer(float(g("status_period_s")), self._publish_status)
        self.get_logger().warning(
            "SIMULATION ONLY: pool localization is Gazebo truth with a fixed "
            f"^pool T_odom (R_z(+90 deg), t={self._xf.t.tolist()})")

    # ── 입력 ──
    def _on_truth(self, m: Odometry) -> None:
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        self._last_truth = (time.monotonic(), np.array([p.x, p.y, p.z]),
                            (q.x, q.y, q.z, q.w))

    def _on_session(self, m: OdometrySession) -> None:
        sid = m.odometry_session_id.strip()
        if not sid:
            self._invalidate("empty odometry session id")
            return
        if sid != self._session:
            # 실기 노드와 같은 규약: 세션이 바뀌면 정렬 identity 도 새로 낸다.
            self._session = sid
            self._alignment_id = str(uuid.uuid4())
            self._epoch += 1
            self._mismatch = None
            self._samples = 0
            self.get_logger().info(
                f"odometry session {sid}: alignment {self._alignment_id[:8]} epoch {self._epoch}")
        o = m.odometry
        if o.header.frame_id.strip() != self._odom:
            self._invalidate(f"odometry frame {o.header.frame_id!r} != {self._odom!r}")
            return
        p_odom = np.array([o.pose.pose.position.x, o.pose.pose.position.y,
                           o.pose.pose.position.z])
        q_odom = (o.pose.pose.orientation.x, o.pose.pose.orientation.y,
                  o.pose.pose.orientation.z, o.pose.pose.orientation.w)
        p_pool = self._xf.position(p_odom)
        q_pool = self._xf.orientation(q_odom)

        # 자기검증: gz truth 를 직접 pool 로 보낸 값과 대조 (표본 시각차 <= 수십 ms)
        if self._last_truth is not None and time.monotonic() - self._last_truth[0] < 0.3:
            _, p_gz, q_gz = self._last_truth
            dp = float(np.linalg.norm(p_pool - self._xf.pool_from_gz(p_gz)))
            dr = quat_angle_deg(q_pool, q_gz)
            self._mismatch = (dp, dr)
            if dp > self._tol_p or dr > self._tol_r:
                self._invalidate(
                    f"truth/odom frame mismatch: {dp * 100:.1f} cm, {dr:.1f} deg "
                    "(is obs_node navigating on Gazebo truth?)")
                return
        self._samples += 1
        self._last_odom_t = time.monotonic()
        if self._state != LocalizationStatus.INITIALIZED:
            self._state = LocalizationStatus.INITIALIZED
            self._reason = "fixed truth alignment"
            self._publish_status()

        out = Odometry()
        out.header.stamp = o.header.stamp
        out.header.frame_id = self._pool
        out.child_frame_id = self._base
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = (
            float(p_pool[0]), float(p_pool[1]), float(p_pool[2]))
        (out.pose.pose.orientation.x, out.pose.pose.orientation.y,
         out.pose.pose.orientation.z, out.pose.pose.orientation.w) = (float(v) for v in q_pool)
        out.twist = o.twist            # body FLU twist 는 프레임 무관
        out.pose.covariance = o.pose.covariance
        out.twist.covariance = o.twist.covariance
        self._pub_pool.publish(out)
        al = AlignedOdometry()
        al.odometry = out
        al.localization_epoch = self._epoch
        al.odometry_session_id = self._session
        al.alignment_id = self._alignment_id
        self._pub_aligned.publish(al)

    def _invalidate(self, reason: str) -> None:
        if self._state != LocalizationStatus.INVALID or self._reason != reason:
            self.get_logger().error(f"localization INVALID: {reason}")
        self._state = LocalizationStatus.INVALID
        self._reason = reason
        self._alignment_id = ""
        self._epoch += 1
        self._publish_status()

    def _publish_status(self) -> None:
        fresh = self._last_odom_t is not None and time.monotonic() - self._last_odom_t < self._stale
        st = LocalizationStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.header.frame_id = self._pool
        st.state = int(self._state)
        st.epoch = int(self._epoch)
        st.odometry_session_id = self._session
        st.alignment_id = self._alignment_id
        tr: Transform = st.pool_to_odom
        tr.translation.x, tr.translation.y, tr.translation.z = (float(v) for v in self._xf.t)
        tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = self._xf.q_xyzw
        st.output_valid = bool(self._state == LocalizationStatus.INITIALIZED
                               and self._alignment_id and self._session and fresh)
        st.sample_count = int(self._samples)
        st.reason = self._reason if fresh or self._state != LocalizationStatus.INITIALIZED \
            else f"{self._reason}; local odometry is stale"
        self._pub_status.publish(st)
        self._pub_valid.publish(Bool(data=st.output_valid))


def main() -> None:
    rclpy.init()
    node = TruthLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inject a Water Linked-like DVL stream into ArduSub SITL.

The real BlueOS Water Linked extension feeds ArduSub with
``VISION_POSITION_DELTA``: a body-frame displacement accumulated over the
DVL sample interval.  This diagnostic node reproduces that contract from the
Gazebo ground-truth odometry without replacing ArduSub's IMU/AHRS or EKF.

Input
-----
``/brov/sim/gazebo_odometry_raw`` (``nav_msgs/Odometry``)
    Gazebo world pose and base_link/body FLU linear velocity.

Output
------
MAVLink ``VISION_POSITION_DELTA``
    Body FRD displacement at the configured DVL packet rate and delay.
MAVLink ``DISTANCE_SENSOR`` (optional)
    Downward flat-bottom altitude matching the Water Linked extension's
    default rangefinder output.
ROS diagnostics under ``/brov/stage2/dvl_*``
    The exact truth, noise, delayed measurement, timing, confidence, and
    validity used for every injected packet.

The node is intentionally simulator-only and fail-closed.  It requires an
explicit ``--confirm-sitl`` flag, accepts only an ``udpin:`` endpoint, rejects
bad frames/non-monotonic simulation stamps, and never arms or actuates the
vehicle.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import os
import time
from typing import Deque, Optional

# VISION_POSITION_DELTA is an ArduPilotMega MAVLink-2 extension message.  This
# must be set before importing pymavlink; the default v1 dialect omits it.
os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, String


@dataclass(frozen=True)
class DvlAcquisition:
    sequence: int
    source_time_ns: int
    dt_s: float
    position_enu: np.ndarray
    altitude_m: float
    velocity_body_frd_true: np.ndarray
    velocity_body_frd_measured: np.ndarray
    position_delta_body_frd: np.ndarray
    noise_body_frd: np.ndarray


def flu_to_frd(vector: np.ndarray) -> np.ndarray:
    """Convert an xyz vector from ROS body FLU to ArduPilot body FRD."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("body vector must be finite and have shape (3,)")
    return np.array([value[0], -value[1], -value[2]], dtype=np.float64)


def confidence_from_fom(fom_mps: float) -> float:
    """Match the Water Linked BlueOS extension's FOM-to-confidence map."""

    if not math.isfinite(fom_mps) or fom_mps < 0.0:
        raise ValueError("FOM must be finite and non-negative")
    return 100.0 * (1.0 - min(0.4, fom_mps) / 0.4)


def stamp_ns(message: Odometry) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", default="udpin:0.0.0.0:14555")
    parser.add_argument(
        "--topic", default="/brov/sim/gazebo_odometry_raw"
    )
    parser.add_argument("--rate-hz", type=float, default=15.0)
    parser.add_argument(
        "--far-rate-hz",
        type=float,
        default=0.0,
        help=(
            "Optional Water Linked auto-range far-mode rate. Zero disables "
            "range-dependent switching."
        ),
    )
    parser.add_argument(
        "--range-transition-m",
        type=float,
        default=3.0,
        help="Flat-bottom altitude above which --far-rate-hz is used.",
    )
    parser.add_argument("--delay-s", type=float, default=0.10)
    parser.add_argument("--velocity-noise-std", type=float, default=0.003)
    parser.add_argument("--fom-mps", type=float, default=0.003)
    parser.add_argument(
        "--rangefinder",
        action="store_true",
        help=(
            "Emit Water Linked-compatible downward DISTANCE_SENSOR messages "
            "from the flat Gazebo seabed altitude."
        ),
    )
    parser.add_argument(
        "--seabed-world-z",
        type=float,
        default=-10.0,
        help="Flat Gazebo seabed world-Z used to derive DVL altitude [m].",
    )
    parser.add_argument("--rangefinder-max-m", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--expected-frame", default="odom")
    parser.add_argument("--expected-child-frame", default="base_link")
    parser.add_argument(
        "--bottom-lock-min-world-z",
        type=float,
        default=-9.5,
        help="Reject samples below this Gazebo ENU world z [m].",
    )
    parser.add_argument(
        "--bottom-lock-max-world-z",
        type=float,
        default=-0.5,
        help="Reject samples above this Gazebo ENU world z [m].",
    )
    parser.add_argument(
        "--duration-s", type=float, default=0.0, help="0 means run until stopped."
    )
    parser.add_argument(
        "--confirm-sitl",
        action="store_true",
        help="Required acknowledgement that this endpoint is Gazebo SITL.",
    )
    return parser


class DvlInjector(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("brov_stage2_dvl_injector")
        self.args = args
        self.near_period_ns = int(round(1_000_000_000 / args.rate_hz))
        self.far_period_ns = (
            int(round(1_000_000_000 / args.far_rate_hz))
            if args.far_rate_hz > 0.0
            else self.near_period_ns
        )
        slowest_rate_hz = (
            min(args.rate_hz, args.far_rate_hz)
            if args.far_rate_hz > 0.0
            else args.rate_hz
        )
        self.max_acquisition_dt_s = 1.25 / slowest_rate_hz
        self.delay_ns = int(round(args.delay_s * 1_000_000_000))
        self.rng = np.random.default_rng(args.seed)
        self.confidence = confidence_from_fom(args.fom_mps)
        self.pending: Deque[DvlAcquisition] = deque()
        self.next_acquisition_ns: Optional[int] = None
        self.previous_acquisition_ns: Optional[int] = None
        self.last_raw_stamp_ns: Optional[int] = None
        self.sequence = 0
        self.sent_count = 0
        self.rangefinder_sent_count = 0
        self.invalid_count = 0
        self.start_monotonic = time.monotonic()

        self.master = mavutil.mavlink_connection(
            args.connection,
            source_system=255,
            source_component=191,
            autoreconnect=False,
        )
        heartbeat = self.master.wait_heartbeat(timeout=10.0)
        if heartbeat is None:
            raise RuntimeError("ArduSub heartbeat unavailable on DVL endpoint")
        if not hasattr(self.master.mav, "vision_position_delta_send"):
            raise RuntimeError(
                "pymavlink did not load MAVLink 2 ArduPilotMega dialect"
            )

        qos = QoSProfile(depth=50)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.subscription = self.create_subscription(
            Odometry, args.topic, self._on_odometry, qos
        )
        self.sample_pub = self.create_publisher(
            Float64MultiArray, "/brov/stage2/dvl_sample", 50
        )
        self.valid_pub = self.create_publisher(
            Bool, "/brov/stage2/dvl_valid", 10
        )
        schema_qos = QoSProfile(depth=1)
        schema_qos.reliability = ReliabilityPolicy.RELIABLE
        schema_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.schema_pub = self.create_publisher(
            String, "/brov/stage2/dvl_schema", schema_qos
        )
        self.status_pub = self.create_publisher(
            String, "/brov/stage2/dvl_status", 10
        )
        self.schema_pub.publish(
            String(
                data=(
                    "sequence,source_time_s,emit_source_time_s,dt_s,"
                    "configured_delay_s,world_z,true_v_frd_x,true_v_frd_y,"
                    "true_v_frd_z,measured_v_frd_x,measured_v_frd_y,"
                    "measured_v_frd_z,delta_frd_x,delta_frd_y,delta_frd_z,"
                    "noise_frd_x,noise_frd_y,noise_frd_z,fom_mps,confidence,"
                    "wall_receive_time_s,altitude_m,rangefinder_sent"
                )
            )
        )
        self.get_logger().info(
            "DVL SITL injector ready: "
            f"topic={args.topic}, rate={args.rate_hz:.3f} Hz, "
            f"far_rate={args.far_rate_hz:.3f} Hz above "
            f"{args.range_transition_m:.3f} m, "
            f"delay={args.delay_s:.3f} s, noise_std={args.velocity_noise_std:.4f} m/s, "
            f"confidence={self.confidence:.3f}, rangefinder={args.rangefinder}, "
            f"connection={args.connection}"
        )

    def _invalidate(self, reason: str, clear: bool = False) -> None:
        self.invalid_count += 1
        self.valid_pub.publish(Bool(data=False))
        self.status_pub.publish(String(data=f"INVALID: {reason}"))
        if clear:
            self.pending.clear()
            self.next_acquisition_ns = None
            self.previous_acquisition_ns = None

    def _validate_message(self, message: Odometry, source_ns: int) -> bool:
        if message.header.frame_id != self.args.expected_frame:
            self._invalidate(
                f"frame_id={message.header.frame_id!r}, expected "
                f"{self.args.expected_frame!r}"
            )
            return False
        if message.child_frame_id != self.args.expected_child_frame:
            self._invalidate(
                f"child_frame_id={message.child_frame_id!r}, expected "
                f"{self.args.expected_child_frame!r}"
            )
            return False
        if source_ns <= 0:
            self._invalidate("zero/negative simulation source stamp", clear=True)
            return False
        if self.last_raw_stamp_ns is not None and source_ns <= self.last_raw_stamp_ns:
            self._invalidate("non-monotonic simulation source stamp", clear=True)
            self.last_raw_stamp_ns = source_ns
            return False
        values = np.array(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
                message.pose.pose.orientation.w,
                message.twist.twist.linear.x,
                message.twist.twist.linear.y,
                message.twist.twist.linear.z,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            self._invalidate("NaN/Inf in Gazebo odometry", clear=True)
            return False
        q_norm = float(np.linalg.norm(values[3:7]))
        if not 0.99 <= q_norm <= 1.01:
            self._invalidate(f"invalid quaternion norm {q_norm:.6f}", clear=True)
            return False
        world_z = float(message.pose.pose.position.z)
        if not (
            self.args.bottom_lock_min_world_z
            <= world_z
            <= self.args.bottom_lock_max_world_z
        ):
            self._invalidate(
                f"bottom lock unavailable at Gazebo world z={world_z:.3f} m",
                clear=True,
            )
            return False
        return True

    def _acquire(self, message: Odometry, source_ns: int) -> None:
        if self.previous_acquisition_ns is None:
            self.previous_acquisition_ns = source_ns
            return
        dt_s = (source_ns - self.previous_acquisition_ns) * 1e-9
        self.previous_acquisition_ns = source_ns
        if not 0.04 <= dt_s <= self.max_acquisition_dt_s:
            self._invalidate(
                f"DVL acquisition dt={dt_s:.6f} s outside "
                f"[0.04,{self.max_acquisition_dt_s:.3f}]"
            )
            return
        velocity_flu = np.array(
            [
                message.twist.twist.linear.x,
                message.twist.twist.linear.y,
                message.twist.twist.linear.z,
            ],
            dtype=np.float64,
        )
        true_frd = flu_to_frd(velocity_flu)
        noise = self.rng.normal(
            loc=0.0, scale=self.args.velocity_noise_std, size=3
        )
        measured_frd = true_frd + noise
        altitude_m = float(message.pose.pose.position.z) - float(
            self.args.seabed_world_z
        )
        if not math.isfinite(altitude_m) or altitude_m <= 0.0:
            self._invalidate(
                f"invalid flat-bottom altitude {altitude_m:.3f} m", clear=True
            )
            return
        self.sequence += 1
        self.pending.append(
            DvlAcquisition(
                sequence=self.sequence,
                source_time_ns=source_ns,
                dt_s=dt_s,
                position_enu=np.array(
                    [
                        message.pose.pose.position.x,
                        message.pose.pose.position.y,
                        message.pose.pose.position.z,
                    ],
                    dtype=np.float64,
                ),
                altitude_m=altitude_m,
                velocity_body_frd_true=true_frd,
                velocity_body_frd_measured=measured_frd,
                position_delta_body_frd=measured_frd * dt_s,
                noise_body_frd=noise,
            )
        )

    def _send_ready(self, current_source_ns: int) -> None:
        while self.pending and (
            current_source_ns - self.pending[0].source_time_ns >= self.delay_ns
        ):
            sample = self.pending.popleft()
            self.master.mav.vision_position_delta_send(
                int(sample.source_time_ns // 1000),
                int(round(sample.dt_s * 1_000_000)),
                [0.0, 0.0, 0.0],
                sample.position_delta_body_frd.tolist(),
                float(self.confidence),
            )
            rangefinder_sent = False
            if self.args.rangefinder and sample.altitude_m > 0.05:
                altitude_cm = int(round(sample.altitude_m * 100.0))
                max_cm = int(round(self.args.rangefinder_max_m * 100.0))
                altitude_cm = max(5, min(max_cm, altitude_cm))
                self.master.mav.distance_sensor_send(
                    int((sample.source_time_ns // 1_000_000) & 0xFFFFFFFF),
                    0,
                    max_cm,
                    altitude_cm,
                    mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,
                    0,
                    mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,
                    0,
                )
                self.rangefinder_sent_count += 1
                rangefinder_sent = True
            self.sent_count += 1
            emit_source_s = current_source_ns * 1e-9
            world_z = float(sample.position_enu[2])
            row = [
                float(sample.sequence),
                sample.source_time_ns * 1e-9,
                emit_source_s,
                sample.dt_s,
                self.args.delay_s,
                world_z,
                *sample.velocity_body_frd_true.tolist(),
                *sample.velocity_body_frd_measured.tolist(),
                *sample.position_delta_body_frd.tolist(),
                *sample.noise_body_frd.tolist(),
                self.args.fom_mps,
                self.confidence,
                time.time_ns() * 1e-9,
                sample.altitude_m,
                1.0 if rangefinder_sent else 0.0,
            ]
            self.sample_pub.publish(Float64MultiArray(data=row))
            self.valid_pub.publish(Bool(data=True))
            self.status_pub.publish(
                String(data=f"VALID sequence={sample.sequence}")
            )

    def _on_odometry(self, message: Odometry) -> None:
        source_ns = stamp_ns(message)
        if not self._validate_message(message, source_ns):
            self.last_raw_stamp_ns = source_ns
            return
        self.last_raw_stamp_ns = source_ns
        if self.next_acquisition_ns is None:
            self.next_acquisition_ns = source_ns
        if source_ns >= self.next_acquisition_ns:
            altitude_m = float(message.pose.pose.position.z) - float(
                self.args.seabed_world_z
            )
            period_ns = (
                self.far_period_ns
                if self.args.far_rate_hz > 0.0
                and altitude_m > self.args.range_transition_m
                else self.near_period_ns
            )
            self._acquire(message, source_ns)
            while self.next_acquisition_ns <= source_ns:
                self.next_acquisition_ns += period_ns
        self._send_ready(source_ns)

    def duration_expired(self) -> bool:
        return self.args.duration_s > 0.0 and (
            time.monotonic() - self.start_monotonic >= self.args.duration_s
        )

    def close(self) -> None:
        self.get_logger().info(
            "DVL injector stopping: "
            f"sent={self.sent_count}, rangefinder_sent={self.rangefinder_sent_count}, "
            f"invalid={self.invalid_count}"
        )
        self.master.close()


def main() -> None:
    args = _parser().parse_args()
    if not args.confirm_sitl:
        raise SystemExit("refusing DVL injection: pass --confirm-sitl")
    if not args.connection.startswith("udpin:"):
        raise SystemExit("DVL SITL injector only accepts a dedicated udpin endpoint")
    if not 1.0 <= args.rate_hz <= 50.0:
        raise SystemExit("--rate-hz must be in [1, 50]")
    if args.far_rate_hz != 0.0 and not 1.0 <= args.far_rate_hz <= 20.0:
        raise SystemExit("--far-rate-hz must be zero or in [1, 20]")
    if not 0.05 <= args.range_transition_m <= args.rangefinder_max_m:
        raise SystemExit("--range-transition-m must be within rangefinder limits")
    if not 0.0 <= args.delay_s <= 1.0:
        raise SystemExit("--delay-s must be in [0, 1]")
    if not 0.0 <= args.velocity_noise_std <= 0.2:
        raise SystemExit("--velocity-noise-std must be in [0, 0.2]")
    if args.bottom_lock_min_world_z >= args.bottom_lock_max_world_z:
        raise SystemExit("bottom-lock min world z must be less than max")
    if not math.isfinite(args.seabed_world_z):
        raise SystemExit("--seabed-world-z must be finite")
    if not 0.05 <= args.rangefinder_max_m <= 100.0:
        raise SystemExit("--rangefinder-max-m must be in [0.05, 100]")

    rclpy.init()
    node: Optional[DvlInjector] = None
    try:
        node = DvlInjector(args)
        while rclpy.ok() and not node.duration_expired():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

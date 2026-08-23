#!/usr/bin/env python3
"""Wait for a deterministic rising-vehicle Gazebo world-Z start barrier."""

from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class StartBarrier(Node):
    def __init__(self, target_world_z: float, timeout_s: float) -> None:
        super().__init__("brov_stage2_gt_start_barrier")
        self.target_world_z = target_world_z
        self.deadline = time.monotonic() + timeout_s
        self.result: dict | None = None
        self.failure = ""
        self.first_sample = True
        self.create_subscription(
            Odometry,
            "/brov/sim/gazebo_odometry_raw",
            self._sample,
            50,
        )

    def _sample(self, message: Odometry) -> None:
        if message.header.frame_id != "odom" or message.child_frame_id != "base_link":
            self.failure = "unexpected Gazebo odometry frames"
            return
        p = message.pose.pose.position
        v = message.twist.twist.linear
        values = [p.x, p.y, p.z, v.x, v.y, v.z]
        if not all(math.isfinite(float(item)) for item in values):
            self.failure = "non-finite Gazebo state"
            return
        if self.first_sample and float(p.z) > self.target_world_z + 0.02:
            self.failure = (
                f"start barrier already overshot: z={p.z:.6f}, "
                f"target={self.target_world_z:.6f}"
            )
            return
        self.first_sample = False
        if float(p.z) < self.target_world_z:
            return
        speed = math.sqrt(float(v.x) ** 2 + float(v.y) ** 2 + float(v.z) ** 2)
        if speed > 0.20:
            self.failure = f"GT speed {speed:.6f} m/s exceeds 0.20 at barrier"
            return
        self.result = {
            "source_time_s": float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9,
            "position_enu_m": [float(p.x), float(p.y), float(p.z)],
            "velocity_body_flu_mps": [float(v.x), float(v.y), float(v.z)],
            "speed_mps": speed,
            "target_world_z_m": self.target_world_z,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-world-z", type=float, default=-6.0)
    parser.add_argument("--timeout-s", type=float, default=7.0)
    args = parser.parse_args()
    if not math.isfinite(args.target_world_z) or args.timeout_s <= 0.0:
        raise SystemExit("invalid start-barrier arguments")

    rclpy.init()
    node = StartBarrier(args.target_world_z, args.timeout_s)
    try:
        while rclpy.ok() and node.result is None and not node.failure:
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.monotonic() >= node.deadline:
                node.failure = "timeout before reaching GT start barrier"
        payload = {
            "success": node.result is not None and not node.failure,
            "state": node.result,
            "failure": node.failure or None,
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        if not payload["success"]:
            raise SystemExit(1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

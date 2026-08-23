#!/usr/bin/env python3
"""Sample Gazebo GT position/attitude for a fixed hold window after START and
report hover drift from the first active sample. This is the first-ever
Gazebo run of this policy through this integration path, so it does not
assert a pass/fail threshold on tracking quality -- only on the run staying
finite and not runaway (drift beyond --abort-drift-m aborts early so the
caller can stop/disarm promptly)."""

from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class HoverMonitor(Node):
    def __init__(self, hold_s: float, abort_drift_m: float) -> None:
        super().__init__("brov_stage2_cmg_hover_monitor")
        self.hold_s = hold_s
        self.abort_drift_m = abort_drift_m
        self.deadline: float | None = None
        self.origin = None
        self.samples: list[dict] = []
        self.failure = ""
        self.done = False
        self.create_subscription(
            Odometry, "/brov/sim/gazebo_odometry_raw", self._sample, 50
        )

    def _sample(self, message: Odometry) -> None:
        if self.done:
            return
        p = message.pose.pose.position
        values = [p.x, p.y, p.z]
        if not all(math.isfinite(float(v)) for v in values):
            self.failure = "non-finite Gazebo position"
            self.done = True
            return
        now = time.monotonic()
        if self.origin is None:
            self.origin = (float(p.x), float(p.y), float(p.z))
            self.deadline = now + self.hold_s
        drift = math.sqrt(
            (float(p.x) - self.origin[0]) ** 2
            + (float(p.y) - self.origin[1]) ** 2
            + (float(p.z) - self.origin[2]) ** 2
        )
        self.samples.append(
            {"t_s": now, "position_enu_m": [float(p.x), float(p.y), float(p.z)], "drift_m": drift}
        )
        if drift > self.abort_drift_m:
            self.failure = f"drift {drift:.3f} m exceeded abort threshold {self.abort_drift_m:.3f} m"
            self.done = True
            return
        if self.deadline is not None and now >= self.deadline:
            self.done = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-s", type=float, default=40.0)
    parser.add_argument("--abort-drift-m", type=float, default=3.0)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args = parser.parse_args()

    rclpy.init()
    node = HoverMonitor(args.hold_s, args.abort_drift_m)
    started = time.monotonic()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.monotonic() - started >= args.timeout_s:
                node.failure = node.failure or "timeout before hold window completed"
                break
        drifts = [s["drift_m"] for s in node.samples]
        payload = {
            "success": not node.failure,
            "failure": node.failure or None,
            "sample_count": len(node.samples),
            "origin_enu_m": list(node.origin) if node.origin else None,
            "final_enu_m": node.samples[-1]["position_enu_m"] if node.samples else None,
            "max_drift_m": max(drifts) if drifts else None,
            "mean_drift_m": (sum(drifts) / len(drifts)) if drifts else None,
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

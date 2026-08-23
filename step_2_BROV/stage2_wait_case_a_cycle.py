#!/usr/bin/env python3
"""Wait for one debounced Case-A takeoff/outbound/return cycle."""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32


class CycleSupervisor(Node):
    """Recognize the looped takeoff_then_align index sequence 0->1->2->1."""

    def __init__(self, timeout_s: float, debounce_samples: int) -> None:
        super().__init__("brov_stage2_case_a_cycle_supervisor")
        self.deadline = time.monotonic() + timeout_s
        self.debounce_samples = debounce_samples
        self.active_seen = False
        self.control_active = False
        self.active_since: float | None = None
        self.last_dvl_true: float | None = None
        self.complete = False
        self.failed = ""
        self.expected = [0, 1, 2, 1]
        self.sequence: list[int] = []
        self.edges_monotonic_s: list[float] = []
        self.candidate: int | None = None
        self.candidate_count = 0

        self.create_subscription(Bool, "/brov/control_active", self._active, 20)
        self.create_subscription(Bool, "/brov/stage2/dvl_valid", self._dvl, 20)
        self.create_subscription(Int32, "/brov/waypoint_idx", self._index, 50)

    def _active(self, message: Bool) -> None:
        self.control_active = bool(message.data)
        if self.control_active:
            if not self.active_seen:
                self.active_since = time.monotonic()
            self.active_seen = True
        elif self.active_seen and not self.complete:
            self.failed = "control became inactive before one complete cycle"

    def _dvl(self, message: Bool) -> None:
        if bool(message.data):
            self.last_dvl_true = time.monotonic()
        elif self.active_seen and self.control_active:
            self.failed = "DVL became invalid during active control"

    def _index(self, message: Int32) -> None:
        if not self.control_active or self.complete or self.failed:
            return
        value = int(message.data)
        if value == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = value
            self.candidate_count = 1
        if self.candidate_count < self.debounce_samples:
            return
        if self.sequence and value == self.sequence[-1]:
            return
        expected = self.expected[len(self.sequence)] if len(self.sequence) < 4 else None
        if value != expected:
            self.failed = (
                f"unexpected debounced waypoint index {value}; "
                f"expected {expected}, observed {self.sequence}"
            )
            return
        self.sequence.append(value)
        self.edges_monotonic_s.append(time.monotonic())
        if self.sequence == self.expected:
            self.complete = True

    def check_timeout(self) -> None:
        now = time.monotonic()
        if self.control_active and not self.failed:
            reference = self.last_dvl_true
            if reference is None:
                reference = self.active_since
            if reference is not None and now - reference > 0.30:
                self.failed = "DVL heartbeat age exceeded 0.30 s"
        if now >= self.deadline and not self.complete:
            self.failed = f"timeout; observed waypoint sequence {self.sequence}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--debounce-samples", type=int, default=3)
    args = parser.parse_args()
    if args.timeout_s <= 0.0 or args.debounce_samples < 1:
        raise SystemExit("timeout must be positive and debounce must be >= 1")

    rclpy.init()
    node = CycleSupervisor(args.timeout_s, args.debounce_samples)
    try:
        while rclpy.ok() and not node.complete and not node.failed:
            rclpy.spin_once(node, timeout_sec=0.05)
            node.check_timeout()
        payload = {
            "success": node.complete and not node.failed,
            "sequence": node.sequence,
            "edge_monotonic_s": node.edges_monotonic_s,
            "failure": node.failed or None,
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

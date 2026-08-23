#!/usr/bin/env python3
"""Stage-2 open-loop SITL pulse diagnostic.

This deliberately bypasses the RL policy and LOS guidance.  It drives four
small, symmetric horizontal-axis pulses through the same ArduSub
RCPassThru/RC_CHANNELS_OVERRIDE transport used by ``brov_ros2`` while
publishing the commanded phase and the received LOCAL_POSITION_NED snapshot
for rosbag correlation.

The script is fail-closed and is intended only for the Edo Gazebo SITL on a
dedicated MAVProxy UDP output.  It never selects a flight mode by itself: the
operator must put ArduSub in MANUAL (custom_mode=19) first.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import torch

from brov_base.mavlink_interface import RealRobotInterface


MANUAL_CUSTOM_MODE = 19
AXES = {
    # T1..T4 follow the BROV SNAME allocation used by the deployed policy.
    # Vertical T5..T8 remain neutral so this test isolates horizontal motion.
    "surge_pos": torch.tensor([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "surge_neg": torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    "sway_pos": torch.tensor([1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    "sway_neg": torch.tensor([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", default="udpin:0.0.0.0:14553")
    parser.add_argument("--amplitude", type=float, default=0.06)
    parser.add_argument("--pulse-s", type=float, default=1.5)
    parser.add_argument("--settle-s", type=float, default=2.5)
    parser.add_argument("--initial-neutral-s", type=float, default=3.0)
    parser.add_argument("--final-neutral-s", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=25.0)
    parser.add_argument(
        "--vertical-trim",
        type=float,
        default=0.0,
        help=(
            "Fixed logical T5..T8 command used in every phase to counter "
            "positive buoyancy; no feedback is applied."
        ),
    )
    parser.add_argument("--max-speed-mps", type=float, default=1.2)
    parser.add_argument("--max-body-rate-rps", type=float, default=3.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--confirm-sitl",
        action="store_true",
        help="Required acknowledgement that this endpoint is Gazebo SITL.",
    )
    return parser


class PulseDiagnostics(Node):
    def __init__(self) -> None:
        super().__init__("brov_stage2_axis_pulse")
        self.phase_pub = self.create_publisher(String, "/brov/stage2/phase", 10)
        self.command_pub = self.create_publisher(
            Float64MultiArray, "/brov/stage2/pulse_pwm", 10
        )
        self.snapshot_pub = self.create_publisher(
            Float64MultiArray, "/brov/stage2/mavlink_snapshot", 10
        )

    def publish_sample(
        self,
        phase: str,
        phase_code: int,
        phase_elapsed_s: float,
        pwm: torch.Tensor,
        snap: dict,
    ) -> list[float]:
        self.phase_pub.publish(String(data=phase))
        self.command_pub.publish(
            Float64MultiArray(
                data=[float(phase_code), float(phase_elapsed_s), *pwm.tolist()]
            )
        )
        ekf_variance = snap.get("ekf_vel_variance")
        ekf_flags = snap.get("ekf_flags")
        row = [
            time.time_ns() * 1e-9,
            time.monotonic(),
            float(phase_code),
            float(phase_elapsed_s),
            float(snap.get("pos_time_boot_ms") or -1),
            *snap["pos_ned"].tolist(),
            *snap["vel_ned"].tolist(),
            *snap["att_quat_ned"].tolist(),
            *snap["body_rates_ned"].tolist(),
            float("nan") if ekf_variance is None else float(ekf_variance),
            -1.0 if ekf_flags is None else float(ekf_flags),
            float(snap["att_age_s"]),
            float(snap["pos_age_s"]),
            *pwm.tolist(),
        ]
        self.snapshot_pub.publish(Float64MultiArray(data=row))
        return row


CSV_FIELDS = [
    "wall_time_s",
    "monotonic_s",
    "phase_code",
    "phase_elapsed_s",
    "pos_time_boot_ms",
    "pos_n",
    "pos_e",
    "pos_d",
    "vel_n",
    "vel_e",
    "vel_d",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
    "rate_p",
    "rate_q",
    "rate_r",
    "ekf_velocity_variance",
    "ekf_flags",
    "att_age_s",
    "pos_age_s",
    *[f"pwm_t{i}" for i in range(1, 9)],
]


def _wait_for_state(interface: RealRobotInterface, timeout_s: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snap = interface.snapshot()
        control = interface.control_snapshot()
        if snap is not None and control.get("heartbeat_system") is not None:
            return snap
        time.sleep(0.05)
    raise RuntimeError("fresh MAVLink attitude/local-position state unavailable")


def _validate_live_state(
    interface: RealRobotInterface,
    snap: dict,
    max_speed_mps: float,
    max_body_rate_rps: float,
) -> None:
    control = interface.control_snapshot()
    if control.get("custom_mode") != MANUAL_CUSTOM_MODE:
        raise RuntimeError(
            f"custom_mode changed to {control.get('custom_mode')}; MANUAL=19 required"
        )
    if control.get("armed") is not True:
        raise RuntimeError("vehicle is no longer armed")
    if snap["att_age_s"] >= 0.20 or snap["pos_age_s"] >= 0.20:
        raise RuntimeError(
            f"stale telemetry: att={snap['att_age_s']:.3f}s, "
            f"position={snap['pos_age_s']:.3f}s"
        )
    if not all(
        torch.isfinite(value).all()
        for value in (
            snap["pos_ned"],
            snap["vel_ned"],
            snap["att_quat_ned"],
            snap["body_rates_ned"],
        )
    ):
        raise RuntimeError("NaN/Inf in MAVLink navigation state")
    speed = float(torch.linalg.vector_norm(snap["vel_ned"]))
    body_rate = float(torch.linalg.vector_norm(snap["body_rates_ned"]))
    if speed > max_speed_mps:
        raise RuntimeError(f"speed abort: {speed:.3f} > {max_speed_mps:.3f} m/s")
    if body_rate > max_body_rate_rps:
        raise RuntimeError(
            f"body-rate abort: {body_rate:.3f} > {max_body_rate_rps:.3f} rad/s"
        )


def _run_phase(
    node: PulseDiagnostics,
    interface: RealRobotInterface,
    writer: csv.writer,
    phase: str,
    phase_code: int,
    pwm: torch.Tensor,
    duration_s: float,
    rate_hz: float,
    max_speed_mps: float,
    max_body_rate_rps: float,
) -> None:
    period = 1.0 / rate_hz
    start = time.monotonic()
    next_tick = start
    while True:
        now = time.monotonic()
        elapsed = now - start
        if elapsed >= duration_s:
            return
        interface.send_pwm(pwm)
        snap = interface.snapshot()
        if snap is None:
            raise RuntimeError("MAVLink state disappeared during pulse")
        _validate_live_state(
            interface, snap, max_speed_mps, max_body_rate_rps
        )
        writer.writerow(
            node.publish_sample(phase, phase_code, elapsed, pwm, snap)
        )
        rclpy.spin_once(node, timeout_sec=0.0)
        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))


def main() -> None:
    args = _parser().parse_args()
    if not args.confirm_sitl:
        raise SystemExit("refusing output: pass --confirm-sitl for Gazebo SITL")
    if not args.connection.startswith("udpin:"):
        raise SystemExit("Stage-2 pulse only accepts a dedicated udpin SITL endpoint")
    if not 0.0 < args.amplitude <= 0.20:
        raise SystemExit("--amplitude must be in (0, 0.20]")
    if not math.isfinite(args.vertical_trim) or abs(args.vertical_trim) > 0.05:
        raise SystemExit("--vertical-trim must be finite and within [-0.05, 0.05]")
    if args.rate_hz < 20.0 or args.rate_hz > 50.0:
        raise SystemExit("--rate-hz must be in [20, 50]")
    durations = (
        args.pulse_s,
        args.settle_s,
        args.initial_neutral_s,
        args.final_neutral_s,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in durations):
        raise SystemExit("all phase durations must be finite and positive")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = PulseDiagnostics()
    interface = RealRobotInterface(
        args.connection, thruster_reversal_sign=[1.0] * 8
    )
    passthrough_enabled = False
    armed_by_script = False
    try:
        interface.connect()
        _wait_for_state(interface)
        control = interface.control_snapshot()
        print(
            "[stage2] heartbeat "
            f"system={control['heartbeat_system']} component={control['heartbeat_component']} "
            f"mode={control['custom_mode']} armed={control['armed']}"
        )
        if control.get("heartbeat_system") != 1 or control.get("heartbeat_component") != 1:
            raise RuntimeError("unexpected MAVLink autopilot identity")
        if control.get("custom_mode") != MANUAL_CUSTOM_MODE:
            raise RuntimeError(
                f"MANUAL custom_mode=19 required, got {control.get('custom_mode')}"
            )
        if control.get("armed"):
            raise RuntimeError("refusing to reconfigure passthrough while already armed")

        interface.enable_passthrough()
        passthrough_enabled = True
        if not interface.arm():
            raise RuntimeError("SITL arm failed")
        armed_by_script = True
        print("[stage2] ARMED; beginning symmetric horizontal pulses")

        with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_FIELDS)
            neutral = torch.zeros(8)
            neutral[4:] = args.vertical_trim
            phases: list[tuple[str, int, torch.Tensor, float]] = [
                ("initial_neutral", 0, neutral, args.initial_neutral_s),
                ("surge_pos", 1, AXES["surge_pos"] * args.amplitude, args.pulse_s),
                ("settle_after_surge_pos", 0, neutral, args.settle_s),
                ("surge_neg", 2, AXES["surge_neg"] * args.amplitude, args.pulse_s),
                ("settle_after_surge_neg", 0, neutral, args.settle_s),
                ("sway_pos", 3, AXES["sway_pos"] * args.amplitude, args.pulse_s),
                ("settle_after_sway_pos", 0, neutral, args.settle_s),
                ("sway_neg", 4, AXES["sway_neg"] * args.amplitude, args.pulse_s),
                ("final_neutral", 0, neutral, args.final_neutral_s),
            ]
            for phase, code, command, duration in phases:
                print(
                    f"[stage2] {phase}: {duration:.2f}s, "
                    f"pwm={command.tolist()}"
                )
                _run_phase(
                    node,
                    interface,
                    writer,
                    phase,
                    code,
                    command,
                    duration,
                    args.rate_hz,
                    args.max_speed_mps,
                    args.max_body_rate_rps,
                )
                stream.flush()
        print(f"[stage2] pulse sequence complete: {args.output_csv}")
    except KeyboardInterrupt:
        print("\n[stage2] operator abort")
    finally:
        try:
            if interface._master is not None and passthrough_enabled:
                stop_deadline = time.monotonic() + 0.6
                while time.monotonic() < stop_deadline:
                    interface.neutral_stop()
                    time.sleep(0.04)
                if armed_by_script:
                    interface.disarm()
                    time.sleep(0.5)
        finally:
            try:
                interface.close(send_stop=passthrough_enabled)
            finally:
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
        print("[stage2] neutral/disarm/release/parameter-restore complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ArduSub RCPassThru PWM/스러스터 매핑 단계별 안전 진단기.

전기 경로(output)와 실제 운동(direction)을 분리한다. direction 시험은 반드시
MANUAL mode, 명시적 arm 및 --confirm-spin 승인이 있어야 실행된다.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deploy import math_utils as mu
from deploy.real_robot_interface import RealRobotInterface
from deploy.vendor.params import load_brov2_yaml, thruster_pos_dir_ned
from deploy.vendor.thruster import build_allocation_matrix

_NEUTRAL_US = 1500
_PWM_RANGE_US = 400
_MANUAL_CUSTOM_MODE = 19


def _parse_channels(value: str) -> list[int]:
    channels = sorted({int(item.strip()) for item in value.split(",")})
    if not channels or any(ch < 1 or ch > 8 for ch in channels):
        raise argparse.ArgumentTypeError("channels는 1~8의 쉼표 구분 목록이어야 함")
    return channels


def _command(channel: int | None = None, delta_us: int = 0) -> torch.Tensor:
    pwm = torch.zeros(8)
    if channel is not None:
        pwm[channel - 1] = delta_us / _PWM_RANGE_US
    return pwm


def _send_for(interface: RealRobotInterface, pwm: torch.Tensor, duration: float, rate_hz: float) -> None:
    period = 1.0 / rate_hz
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        interface.send_pwm(pwm)
        time.sleep(period)


def _neutral(interface: RealRobotInterface, duration: float = 0.4, rate_hz: float = 20.0) -> None:
    _send_for(interface, torch.zeros(8), duration, rate_hz)


def _wait_control(interface: RealRobotInterface, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = interface.control_snapshot()
        if state["heartbeat_system"] is not None:
            return state
        time.sleep(0.05)
    raise RuntimeError("vehicle heartbeat 상태를 수신하지 못함")


def _wait_servo(interface: RealRobotInterface, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = interface.control_snapshot()
        if state["servo_output_us"] is not None and state["servo_age_s"] < 0.25:
            return state
        time.sleep(0.05)
    raise RuntimeError("SERVO_OUTPUT_RAW를 수신하지 못했거나 stale 상태")


def _print_reversal_parameters(interface: RealRobotInterface, channels: list[int]) -> None:
    print("[diag] ArduSub reversal/function parameters")
    for ch in channels:
        values = {}
        for name in (f"MOT_{ch}_DIRECTION", f"SERVO{ch}_REVERSED",
                     f"RC{ch}_REVERSED", f"SERVO{ch}_FUNCTION"):
            values[name] = interface.get_parameter(name, timeout=2.0)
        rendered = ", ".join(f"{key}={value if value is not None else 'N/A'}" for key, value in values.items())
        print(f"  T{ch}: {rendered}")


def _verify_servo(interface: RealRobotInterface, expected: list[int], tolerance_us: int) -> tuple[bool, list[int]]:
    state = _wait_servo(interface)
    actual = state["servo_output_us"].tolist()
    ok = all(abs(a - e) <= tolerance_us for a, e in zip(actual, expected))
    return ok, actual


def _neutral_test(interface: RealRobotInterface, tolerance_us: int, rate_hz: float) -> None:
    _neutral(interface, rate_hz=rate_hz)
    ok, actual = _verify_servo(interface, [_NEUTRAL_US] * 8, tolerance_us)
    print(f"[neutral] actual={actual} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise RuntimeError("neutral SERVO_OUTPUT_RAW 불일치")


def _output_test(
    interface: RealRobotInterface, channels: list[int], delta_us: int,
    tolerance_us: int, rate_hz: float,
) -> None:
    print("[output] RC override → SERVO_OUTPUT_RAW 양방향 시험")
    for ch in channels:
        for sign in (-1, 1):
            delta = sign * delta_us
            _send_for(interface, _command(ch, delta), 0.35, rate_hz)
            expected = [_NEUTRAL_US] * 8
            expected[ch - 1] += delta
            ok, actual = _verify_servo(interface, expected, tolerance_us)
            mirrored = _NEUTRAL_US - delta
            if abs(actual[ch - 1] - mirrored) <= tolerance_us:
                verdict = "REVERSED"
            elif ok:
                verdict = "DIRECT"
            else:
                verdict = "MISMATCH"
            print(f"  T{ch} command={expected[ch-1]} actual={actual[ch-1]} -> {verdict}; all={actual}")
            _neutral(interface, rate_hz=rate_hz)


def _fresh_motion_snapshot(interface: RealRobotInterface) -> dict:
    snap = interface.snapshot()
    if snap is None:
        raise RuntimeError("ATTITUDE_QUATERNION/LOCAL_POSITION_NED 미수신")
    if snap["att_age_s"] >= 0.2 or snap["pos_age_s"] >= 0.2:
        raise RuntimeError(
            f"stale telemetry: att={snap['att_age_s']:.3f}s pos={snap['pos_age_s']:.3f}s"
        )
    tensors = (snap["att_quat_ned"], snap["body_rates_ned"], snap["vel_ned"])
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("motion telemetry에 NaN/Inf 존재")
    return snap


def _collect_motion(
    interface: RealRobotInterface, pwm: torch.Tensor, duration: float, rate_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    times, samples = [], []
    start = time.monotonic()
    period = 1.0 / rate_hz
    while time.monotonic() - start < duration:
        interface.send_pwm(pwm)
        snap = _fresh_motion_snapshot(interface)
        vel_body = mu.quat_apply(mu.quat_conjugate(snap["att_quat_ned"]), snap["vel_ned"])
        samples.append(torch.cat([vel_body, snap["body_rates_ned"]]))
        times.append(time.monotonic() - start)
        time.sleep(period)
    return torch.tensor(times), torch.stack(samples)


def _linear_fit(times: torch.Tensor, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """각 상태의 시간 기울기와 기울기 표준오차를 반환한다."""
    centered = times - times.mean()
    denom = (centered * centered).sum().clamp_min(1e-9)
    mean = samples.mean(dim=0)
    slope = (centered[:, None] * (samples - mean)).sum(dim=0) / denom
    fitted = mean + centered[:, None] * slope
    dof = max(1, len(times) - 2)
    residual_var = ((samples - fitted) ** 2).sum(dim=0) / dof
    slope_stderr = torch.sqrt(residual_var / denom)
    return slope, slope_stderr


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / denom) if denom > 1e-9 else float("nan")


def _classify(cos_pos: float, cos_neg: float) -> str:
    if not math.isfinite(cos_pos) or not math.isfinite(cos_neg):
        return "INCONCLUSIVE"
    if cos_pos > 0.7 and cos_neg < -0.7:
        return "NORMAL"
    if cos_pos < -0.7 and cos_neg > 0.7:
        return "REVERSED"
    return "INCONCLUSIVE"


def _snr(signal: torch.Tensor, stderr: torch.Tensor) -> float:
    noise = float(stderr.norm())
    return float(signal.norm()) / noise if noise > 1e-9 else float("inf")


def _direction_test(
    interface: RealRobotInterface, B: torch.Tensor, channels: list[int],
    delta_us: int, burst_s: float, settle_s: float, rate_hz: float,
) -> None:
    inferred = [1] * 8
    print("[direction] body-frame acceleration/angular-acceleration 양방향 시험")
    for ch in channels:
        _neutral(interface, settle_s, rate_hz)
        base_t, base_x = _collect_motion(interface, torch.zeros(8), burst_s, rate_hz)
        baseline, baseline_err = _linear_fit(base_t, base_x)
        net, net_err = {}, {}
        for sign in (-1, 1):
            _neutral(interface, settle_s, rate_hz)
            t, x = _collect_motion(interface, _command(ch, sign * delta_us), burst_s, rate_hz)
            command_slope, command_err = _linear_fit(t, x)
            net[sign] = command_slope - baseline
            net_err[sign] = torch.sqrt(command_err**2 + baseline_err**2)
        predicted = B[:, ch - 1]
        force_cos_pos = _cosine(net[1][:3], predicted[:3])
        force_cos_neg = _cosine(net[-1][:3], predicted[:3])
        torque_cos_pos = _cosine(net[1][3:], predicted[3:])
        torque_cos_neg = _cosine(net[-1][3:], predicted[3:])
        force_snr = min(_snr(net[sign][:3], net_err[sign][:3]) for sign in (-1, 1))
        torque_snr = min(_snr(net[sign][3:], net_err[sign][3:]) for sign in (-1, 1))
        force_result = _classify(force_cos_pos, force_cos_neg)
        torque_result = _classify(torque_cos_pos, torque_cos_neg)
        if force_snr < 3.0:
            force_result = "INCONCLUSIVE"
        if torque_snr < 3.0:
            torque_result = "INCONCLUSIVE"
        votes = [result for result in (force_result, torque_result) if result != "INCONCLUSIVE"]
        verdict = votes[0] if votes and all(v == votes[0] for v in votes) else "INCONCLUSIVE"
        if verdict == "REVERSED":
            inferred[ch - 1] = -1
        print(f"\n  [T{ch}] predicted={predicted.tolist()}")
        print(f"    net(-)={net[-1].tolist()}")
        print(f"    net(+)={net[1].tolist()}")
        print(f"    force cos(-/+)={force_cos_neg:+.3f}/{force_cos_pos:+.3f} "
              f"SNR={force_snr:.2f} -> {force_result}")
        print(f"    torque cos(-/+)={torque_cos_neg:+.3f}/{torque_cos_pos:+.3f} "
              f"SNR={torque_snr:.2f} -> {torque_result}")
        print(f"    verdict={verdict}")
    _neutral(interface, rate_hz=rate_hz)
    print(f"[direction] inferred reversal_sign={inferred} (진단 결과만 출력, 자동 적용하지 않음)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ArduSub PWM/스러스터 매핑 단계별 진단")
    parser.add_argument("--connection", default="udpin:0.0.0.0:14550")
    parser.add_argument("--test", choices=("neutral", "output", "direction", "all"), default="neutral")
    parser.add_argument("--channels", type=_parse_channels, default=list(range(1, 9)))
    parser.add_argument("--pwm-delta", type=int, default=40, help="1500us 기준 시험 편차")
    parser.add_argument("--burst-s", type=float, default=0.8)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--servo-tolerance-us", type=int, default=2)
    parser.add_argument("--expected-system", type=int, default=1)
    parser.add_argument("--expected-component", type=int, default=1)
    parser.add_argument("--required-mode", type=int, default=_MANUAL_CUSTOM_MODE,
                        help="ArduSub custom_mode 숫자; MANUAL 기본값 19")
    parser.add_argument("--arm", action="store_true", help="실제 arm (반드시 --confirm-spin 필요)")
    parser.add_argument("--confirm-spin", action="store_true", help="추진기 회전 위험을 확인했음을 명시")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 1 <= args.pwm_delta <= 400:
        raise SystemExit("--pwm-delta는 1~400us 범위여야 함")
    if args.rate_hz < 10.0:
        raise SystemExit("watchdog 대비 --rate-hz는 10Hz 이상이어야 함")
    motion_test = args.test in ("direction", "all")
    if motion_test and not args.arm:
        raise SystemExit("direction/all 시험은 --arm --confirm-spin을 모두 명시해야 함")
    if args.arm and not args.confirm_spin:
        raise SystemExit("--arm 사용 시 --confirm-spin도 명시해야 함")

    params = load_brov2_yaml()
    pos, direction = thruster_pos_dir_ned(params)
    B = build_allocation_matrix(torch.tensor(pos), torch.tensor(direction))
    # 이 진단기는 논리 보정 전 raw RC channel을 측정하므로 production reversal mask를 끈다.
    interface = RealRobotInterface(args.connection, thruster_reversal_sign=[1.0] * 8)
    passthrough_enabled = False
    armed_by_script = False
    try:
        interface.connect()
        state = _wait_control(interface)
        print(f"[diag] heartbeat src={state['heartbeat_system']}:{state['heartbeat_component']} "
              f"target={interface._master.target_system}:{interface._master.target_component} "
              f"mode={state['custom_mode']} armed={state['armed']}")
        if state["heartbeat_system"] != args.expected_system:
            raise RuntimeError(f"예상 system {args.expected_system}, 실제 {state['heartbeat_system']}; PWM 차단")
        if args.expected_component and state["heartbeat_component"] != args.expected_component:
            raise RuntimeError(
                f"예상 component {args.expected_component}, 실제 {state['heartbeat_component']}; PWM 차단"
            )
        if state["custom_mode"] != args.required_mode:
            raise RuntimeError(f"요구 mode={args.required_mode}, 현재 mode={state['custom_mode']}; PWM 차단")

        _print_reversal_parameters(interface, args.channels)
        passthrough_enabled = True
        interface.enable_passthrough()
        print("[diag] RCPassThru 설정 및 read-back 검증 완료")

        if args.arm:
            print("[diag] arm 시도...")
            if not interface.arm():
                raise RuntimeError("arm 실패")
            armed_by_script = True
            print("[diag] ARMED — 실제 추진기 회전 가능")

        _neutral_test(interface, args.servo_tolerance_us, args.rate_hz)
        if args.test in ("output", "all"):
            _output_test(interface, args.channels, args.pwm_delta,
                         args.servo_tolerance_us, args.rate_hz)
        if motion_test:
            _direction_test(interface, B, args.channels, args.pwm_delta,
                            args.burst_s, args.settle_s, args.rate_hz)
    except KeyboardInterrupt:
        print("\n[diag] 사용자 중단")
    except Exception as exc:
        print(f"[diag] ERROR: {exc}")
        raise
    finally:
        try:
            if interface._master is not None and passthrough_enabled:
                _neutral(interface)
                if armed_by_script:
                    interface.disarm()
                    time.sleep(0.5)
        finally:
            # close(): override release + (설정했다면) SERVO FUNCTION 원복/read-back
            interface.close(send_stop=passthrough_enabled)
            print(f"[diag] 종료 (passthrough_changed={passthrough_enabled})")


if __name__ == "__main__":
    main()

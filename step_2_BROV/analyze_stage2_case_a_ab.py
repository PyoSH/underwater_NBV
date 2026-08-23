#!/usr/bin/env python3
"""Analyze a paired full Case-A Gazebo-truth/DVL-EKF regression.

The first bag must use ``feedback_source=gazebo_truth`` and the second must use
``feedback_source=mavlink_ekf`` with the no-GPS DVL injector running.  A valid
cycle has the waypoint-index run-length encoding ``[0, 1, 2, 1]``:

* index 0: takeoff, ending at the first 0 -> 1 edge;
* index 1: outbound, ending at the 1 -> 2 turn edge;
* index 2: return, ending at the 2 -> 1 arrival edge;
* the final index 1 is the next-outbound sentinel, not part of return.

The script reuses the synchronization, quaternion, tracking, consistency and
JSON helpers from :mod:`analyze_brov_stage1_ab`, but does not reuse its
single-leg ``idx == 1`` phase mask because that would merge repeated Case-A
indices.  Bags are read-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from std_msgs.msg import (
    Bool,
    Float32MultiArray,
    Float64MultiArray,
    Int32,
    Int32MultiArray,
    String,
)

import analyze_brov_stage1_ab as stage1


EXPECTED_RLE = [0, 1, 2, 1]
AXES = stage1.AXES
THRUSTERS = tuple(f"T{i}" for i in range(1, 9))
WRENCH_SCALE = np.asarray([85.0, 85.0, 120.0, 26.0, 14.0, 22.0])

ARRAY32_TOPICS = set(stage1.ARRAY32_TOPICS) | {
    "/brov/debug/pos_ned",
    "/brov/debug/vel_ned",
    "/brov/debug/att_quat_ned",
    "/brov/target_waypoint",
}
ARRAY64_TOPICS = set(stage1.ARRAY64_TOPICS) | {
    "/brov/stage2/dvl_sample",
}
BOOL_TOPICS = set(stage1.BOOL_TOPICS) | {
    "/brov/stage2/dvl_valid",
}
INT_TOPICS = set(stage1.INT_TOPICS)
INT_ARRAY_TOPICS = {"/brov/debug/servo_output_us"}
STRING_TOPICS = set(stage1.STRING_TOPICS) | {
    "/brov/stage2/dvl_schema",
    "/brov/stage2/dvl_status",
}
ALL_TOPICS = (
    ARRAY32_TOPICS
    | ARRAY64_TOPICS
    | BOOL_TOPICS
    | INT_TOPICS
    | INT_ARRAY_TOPICS
    | STRING_TOPICS
)

CORE_REQUIRED = {
    "/brov/control_active",
    "/brov/waypoint_idx",
    "/brov/observation",
    "/brov/action",
    "/brov/policy/action_raw",
    "/brov/policy/wrench_requested",
    "/brov/policy/wrench_after_thruster_limit",
    "/brov/policy/thruster_force_requested",
    "/brov/policy/thruster_force_limited",
    "/brov/policy/thruster_pwm_requested",
    "/brov/policy/thruster_pwm_preview",
    "/brov/thruster_pwm",
    "/brov/debug/servo_output_us",
    "/brov/debug/feedback_source",
    "/brov/debug/feedback_timing",
    "/brov/debug/feedback_pos_ned",
    "/brov/debug/feedback_vel_ned",
    "/brov/debug/feedback_att_quat_ned",
    "/brov/debug/pos_ned",
    "/brov/debug/vel_ned",
    "/brov/debug/att_quat_ned",
    "/brov/debug/gazebo_truth_pos_ned",
    "/brov/debug/gazebo_truth_vel_ned",
    "/brov/debug/gazebo_truth_att_quat_ned",
    "/brov/debug/v_body_zup",
    "/brov/debug/v_desired_body_zup",
    "/brov/debug/q_desired_zup",
}
DVL_REQUIRED = {
    "/brov/stage2/dvl_sample",
    "/brov/stage2/dvl_schema",
    "/brov/stage2/dvl_valid",
    "/brov/stage2/dvl_status",
}


def read_bag(uri: str) -> dict[str, Any]:
    """Read only the synchronized Case-A contract topics from one rosbag."""

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=uri, storage_id="sqlite3"),
        ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    out: dict[str, Any] = {topic: [] for topic in ALL_TOPICS}
    present = {item.name: item.type for item in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic not in ALL_TOPICS:
            continue
        if topic in ARRAY32_TOPICS:
            value: object = np.asarray(
                deserialize_message(payload, Float32MultiArray).data,
                dtype=np.float64,
            )
        elif topic in ARRAY64_TOPICS:
            value = np.asarray(
                deserialize_message(payload, Float64MultiArray).data,
                dtype=np.float64,
            )
        elif topic in BOOL_TOPICS:
            value = bool(deserialize_message(payload, Bool).data)
        elif topic in INT_TOPICS:
            value = int(deserialize_message(payload, Int32).data)
        elif topic in INT_ARRAY_TOPICS:
            value = np.asarray(
                deserialize_message(payload, Int32MultiArray).data,
                dtype=np.float64,
            )
        else:
            value = str(deserialize_message(payload, String).data)
        out[topic].append((stamp_ns * 1.0e-9, value))
    out["_present"] = present
    return out


def _active_runs(events: list[tuple[float, object]]) -> list[tuple[float, float]]:
    """Return every true interval without silently hiding restarts."""

    if not events:
        return []
    runs: list[tuple[float, float]] = []
    start: float | None = None
    last_stamp = float(events[0][0])
    for raw_stamp, raw_value in events:
        stamp = float(raw_stamp)
        active = bool(raw_value)
        if active and start is None:
            start = stamp
        elif not active and start is not None:
            runs.append((start, stamp))
            start = None
        last_stamp = stamp
    if start is not None:
        runs.append((start, last_stamp))
    return runs


def _index_rle(
    events: list[tuple[float, object]],
    start: float,
    stop: float,
    debounce_samples: int,
) -> list[dict[str, float | int]]:
    """Debounce and run-length encode waypoint index transitions."""

    if not events or debounce_samples < 1:
        return []
    ordered = sorted((float(t), int(v)) for t, v in events)
    previous = [(t, v) for t, v in ordered if t <= start]
    in_window = [(t, v) for t, v in ordered if start < t <= stop]
    samples: list[tuple[float, int]] = []
    if previous:
        samples.append((start, previous[-1][1]))
    elif in_window:
        samples.append((in_window[0][0], in_window[0][1]))
        in_window = in_window[1:]
    samples.extend(in_window)
    if not samples:
        return []

    raw_runs: list[dict[str, float | int]] = []
    for stamp, value in samples:
        if raw_runs and int(raw_runs[-1]["value"]) == value:
            raw_runs[-1]["samples"] = int(raw_runs[-1]["samples"]) + 1
            raw_runs[-1]["last_sample_s"] = stamp
        else:
            raw_runs.append(
                {
                    "value": value,
                    "entry_s": stamp,
                    "last_sample_s": stamp,
                    "samples": 1,
                }
            )

    stable: list[dict[str, float | int]] = []
    for run in raw_runs:
        if int(run["samples"]) < debounce_samples:
            continue
        if stable and int(stable[-1]["value"]) == int(run["value"]):
            stable[-1]["samples"] = int(stable[-1]["samples"]) + int(
                run["samples"]
            )
            stable[-1]["last_sample_s"] = run["last_sample_s"]
        else:
            stable.append(run.copy())

    result: list[dict[str, float | int]] = []
    for i, run in enumerate(stable):
        entry = float(run["entry_s"])
        exit_stamp = (
            float(stable[i + 1]["entry_s"]) if i + 1 < len(stable) else stop
        )
        result.append(
            {
                "value": int(run["value"]),
                "entry_s": entry,
                "exit_s": exit_stamp,
                "duration_s": max(0.0, exit_stamp - entry),
                "samples": int(run["samples"]),
            }
        )
    return result


def _slice_arrays(
    series: list[tuple[float, object]], start: float, stop: float
) -> tuple[np.ndarray, np.ndarray]:
    stamps, values = stage1.arrays(series)
    use = (stamps >= start) & (stamps < stop)
    return stamps[use], values[use]


def _window_mask(
    stamps: np.ndarray, start: float, stop: float, settle_s: float = 0.0
) -> np.ndarray:
    return (stamps >= start + settle_s) & (stamps < stop)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _gate(value: Any, passed: bool, criterion: str, *, evaluated: bool = True) -> dict:
    return {
        "value": value,
        "criterion": criterion,
        "evaluated": bool(evaluated),
        "pass": bool(passed) if evaluated else False,
    }


def _all_gates_pass(gates: Iterable[dict]) -> bool:
    values = list(gates)
    return bool(values) and all(item.get("evaluated") and item.get("pass") for item in values)


def _longest_true_duration(stamps: np.ndarray, mask: np.ndarray) -> float:
    if stamps.size == 0 or mask.size == 0 or not np.any(mask):
        return 0.0
    median_dt = float(np.median(np.diff(stamps))) if stamps.size >= 2 else 0.0
    longest = 0.0
    begin: int | None = None
    for i, active in enumerate(mask.tolist() + [False]):
        if active and begin is None:
            begin = i
        elif not active and begin is not None:
            final = i - 1
            longest = max(longest, stamps[final] - stamps[begin] + median_dt)
            begin = None
    return float(longest)


def _axis_statistics(values: np.ndarray, labels: tuple[str, ...]) -> dict:
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(labels):
        return {"samples": 0}
    return {
        "samples": int(values.shape[0]),
        "signed_mean": dict(zip(labels, np.mean(values, axis=0).tolist())),
        "rms": dict(zip(labels, np.sqrt(np.mean(values**2, axis=0)).tolist())),
        "p95_abs": dict(
            zip(labels, np.percentile(np.abs(values), 95, axis=0).tolist())
        ),
        "min": dict(zip(labels, np.min(values, axis=0).tolist())),
        "max": dict(zip(labels, np.max(values, axis=0).tolist())),
        "max_abs": dict(zip(labels, np.max(np.abs(values), axis=0).tolist())),
    }


def _held_samples(
    source_t: np.ndarray, source_v: np.ndarray, query_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if source_t.size == 0 or query_t.size == 0:
        width = source_v.shape[1] if source_v.ndim == 2 else 1
        return np.full((query_t.size, width), np.nan), np.zeros(query_t.size, bool)
    index = np.searchsorted(source_t, query_t, side="right") - 1
    valid = index >= 0
    index = np.clip(index, 0, source_t.size - 1)
    result = source_v[index].copy()
    result[~valid] = np.nan
    return result, valid


def _servo_transport_metrics(
    bag: dict[str, Any], start: float, stop: float, settle_s: float
) -> dict:
    """Compare normalized sent PWM with SITL's identity SERVO_OUTPUT echo."""

    sent_t, sent = stage1.arrays(bag["/brov/thruster_pwm"])
    servo_t, servo = stage1.arrays(bag["/brov/debug/servo_output_us"])
    servo_phase = _window_mask(servo_t, start, stop, settle_s)
    if (
        sent.ndim != 2
        or sent.shape[1] != 8
        or servo.ndim != 2
        or servo.shape[1] != 8
        or np.count_nonzero(servo_phase) < 20
    ):
        return {"samples": 0}
    errors = []
    lags = []
    for index in np.flatnonzero(servo_phase):
        left = int(np.searchsorted(sent_t, servo_t[index] - 0.12, side="left"))
        right = int(np.searchsorted(sent_t, servo_t[index] + 1.0e-9, side="right"))
        if right <= left:
            continue
        candidates_us = 1500.0 + 400.0 * sent[left:right]
        candidate_error = servo[index] - candidates_us
        score = np.mean(np.abs(candidate_error), axis=1)
        selected = left + int(np.argmin(score))
        errors.append(servo[index] - (1500.0 + 400.0 * sent[selected]))
        lags.append(float(servo_t[index] - sent_t[selected]))
    if len(errors) < 20:
        return {"samples": 0}
    error = np.asarray(errors, dtype=float)
    lag = np.asarray(lags, dtype=float)
    mae = float(np.mean(np.abs(error)))
    return {
        "samples": int(error.shape[0]),
        "mapping": "SITL identity: servo_us=1500+400*sent_pwm",
        "matching": "per-servo causal content match over previous 120 ms",
        "matched_lag_median_s": float(np.median(lag)),
        "matched_lag_p95_s": stage1.percentile(lag, 95),
        "matched_lag_max_s": float(np.max(lag)),
        "mae_us": mae,
        "p95_abs_error_us": stage1.percentile(np.abs(error).reshape(-1), 95),
        "max_abs_error_us": float(np.max(np.abs(error))),
        "per_thruster_mae_us": dict(
            zip(THRUSTERS, np.mean(np.abs(error), axis=0).tolist())
        ),
        "sent_rate": stage1.rate_metrics(
            sent_t[_window_mask(sent_t, start, stop, settle_s)]
        ),
        "servo_rate": stage1.rate_metrics(servo_t[servo_phase]),
    }


def _preview_sent_metrics(
    bag: dict[str, Any], start: float, stop: float, settle_s: float
) -> dict:
    preview_t, preview = stage1.arrays(bag["/brov/policy/thruster_pwm_preview"])
    sent_t, sent = stage1.arrays(bag["/brov/thruster_pwm"])
    preview_at_sent, valid, skew = stage1.nearest(
        preview_t, preview, sent_t, tolerance_s=0.03
    )
    use = _window_mask(sent_t, start, stop, settle_s) & valid
    if (
        sent.ndim != 2
        or sent.shape[1] != 8
        or preview_at_sent.ndim != 2
        or preview_at_sent.shape[1] != 8
        or not np.any(use)
    ):
        return {"samples": 0}
    error = sent[use] - preview_at_sent[use]
    return {
        "samples": int(error.shape[0]),
        "rms": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
        "pairing_skew_p95_ms": 1.0e3
        * stage1.percentile(np.abs(skew[use]), 95),
    }


def _lifecycle_metrics(
    bag: dict[str, Any], cycle_edge_s: float, inactive_edge_s: float
) -> dict:
    sent_t, _ = stage1.arrays(bag["/brov/thruster_pwm"])
    servo_t, servo = stage1.arrays(bag["/brov/debug/servo_output_us"])
    sent_after = sent_t > inactive_edge_s + 1.0e-6
    servo_after = servo_t >= inactive_edge_s
    neutral_delay = math.nan
    final_five_neutral = False
    if servo.ndim == 2 and servo.shape[1] == 8 and np.any(servo_after):
        after_t = servo_t[servo_after]
        after = servo[servo_after]
        neutral = np.all(np.abs(after - 1500.0) <= 2.0, axis=1)
        if np.any(neutral):
            neutral_delay = float(after_t[np.flatnonzero(neutral)[0]] - inactive_edge_s)
        final_five_neutral = bool(
            after.shape[0] >= 5 and np.all(np.abs(after[-5:] - 1500.0) <= 2.0)
        )
    return {
        "cycle_edge_unix_s": cycle_edge_s,
        "inactive_edge_unix_s": inactive_edge_s,
        "cycle_edge_to_inactive_s": inactive_edge_s - cycle_edge_s,
        "sent_samples_after_cycle_before_inactive": int(
            np.count_nonzero(
                (sent_t >= cycle_edge_s) & (sent_t <= inactive_edge_s)
            )
        ),
        "sent_samples_after_inactive": int(np.count_nonzero(sent_after)),
        "first_neutral_echo_delay_s": neutral_delay,
        "final_five_servo_samples_neutral": final_five_neutral,
    }


def _initial_state_metrics(bag: dict[str, Any], active_start: float) -> dict:
    def first_vector(topic: str) -> tuple[float, np.ndarray | None]:
        stamps, values = stage1.arrays(bag[topic])
        candidates = np.flatnonzero(stamps >= active_start)
        if candidates.size == 0:
            return math.nan, None
        index = int(candidates[0])
        return float(stamps[index]), values[index].copy()

    pos_t, pos = first_vector("/brov/debug/gazebo_truth_pos_ned")
    vel_t, vel = first_vector("/brov/debug/gazebo_truth_vel_ned")
    att_t, att = first_vector("/brov/debug/gazebo_truth_att_quat_ned")
    dvl_t, dvl = first_vector("/brov/stage2/dvl_sample")
    return {
        "gazebo_truth_position_ned_m": (
            pos.tolist() if pos is not None else [math.nan] * 3
        ),
        "gazebo_truth_velocity_ned_mps": (
            vel.tolist() if vel is not None else [math.nan] * 3
        ),
        "gazebo_truth_speed_mps": (
            float(np.linalg.norm(vel)) if vel is not None else math.nan
        ),
        "gazebo_truth_attitude_wxyz": (
            att.tolist() if att is not None else [math.nan] * 4
        ),
        "sample_offsets_after_active_s": {
            "position": pos_t - active_start,
            "velocity": vel_t - active_start,
            "attitude": att_t - active_start,
            "dvl": dvl_t - active_start,
        },
        "dvl_sequence": (
            int(round(float(dvl[0])))
            if dvl is not None and dvl.shape[0] >= 1
            else None
        ),
        "dvl_source_time_s": (
            float(dvl[1]) if dvl is not None and dvl.shape[0] >= 2 else math.nan
        ),
    }


def _velocity_tracking(
    bag: dict[str, Any],
    start: float,
    stop: float,
    settle_s: float,
    pair_tolerance_s: float,
) -> dict:
    vd_t, vd = stage1.arrays(bag["/brov/debug/v_desired_body_zup"])
    selected_t, selected = stage1.arrays(bag["/brov/debug/v_body_zup"])
    selected_at_vd, selected_valid, selected_skew = stage1.nearest(
        selected_t, selected, vd_t, pair_tolerance_s
    )
    truth_v_t, truth_v = stage1.arrays(bag["/brov/debug/gazebo_truth_vel_ned"])
    truth_q_t, truth_q = stage1.arrays(bag["/brov/debug/gazebo_truth_att_quat_ned"])
    truth_v_at_vd, truth_v_valid, truth_v_skew = stage1.nearest(
        truth_v_t, truth_v, vd_t, pair_tolerance_s
    )
    truth_q_at_vd, truth_q_valid, truth_q_skew = stage1.nearest(
        truth_q_t, truth_q, vd_t, pair_tolerance_s
    )
    phase = _window_mask(vd_t, start, stop, settle_s)
    phase &= np.linalg.norm(vd, axis=1) >= 0.25
    selected_use = phase & selected_valid
    truth_use = phase & truth_v_valid & truth_q_valid
    if np.any(truth_use):
        truth_body_frd = stage1.q_rotate(
            stage1.q_conj(truth_q_at_vd[truth_use]), truth_v_at_vd[truth_use]
        )
        truth_body_zup = truth_body_frd * np.asarray([1.0, -1.0, -1.0])
    else:
        truth_body_zup = np.empty((0, 3))
    return {
        "desired_samples": int(np.count_nonzero(phase)),
        "controller_visible": stage1.tracking_metrics(
            vd[selected_use], selected_at_vd[selected_use]
        ),
        "gazebo_ground_truth": stage1.tracking_metrics(
            vd[truth_use], truth_body_zup
        ),
        "pairing_skew_p95_ms": {
            "selected_body": 1.0e3
            * stage1.percentile(np.abs(selected_skew[selected_valid & phase]), 95),
            "truth_velocity": 1.0e3
            * stage1.percentile(np.abs(truth_v_skew[truth_v_valid & phase]), 95),
            "truth_attitude": 1.0e3
            * stage1.percentile(np.abs(truth_q_skew[truth_q_valid & phase]), 95),
        },
    }


def _initial_position_offset(
    bag: dict[str, Any], start: float, stop: float, pair_tolerance_s: float
) -> np.ndarray:
    ekf_t, ekf = stage1.arrays(bag["/brov/debug/pos_ned"])
    truth_t, truth = stage1.arrays(bag["/brov/debug/gazebo_truth_pos_ned"])
    truth_at_ekf, valid, _ = stage1.nearest(truth_t, truth, ekf_t, pair_tolerance_s)
    use = valid & _window_mask(ekf_t, start, stop)
    if not np.any(use):
        return np.full(3, np.nan)
    return (ekf[use] - truth_at_ekf[use])[0]


def _estimator_metrics(
    bag: dict[str, Any],
    start: float,
    stop: float,
    settle_s: float,
    pair_tolerance_s: float,
    initial_position_offset: np.ndarray,
) -> dict:
    # Always use raw MAVLink diagnostics.  In the GT run the selected-feedback
    # diagnostics equal truth and would otherwise hide the shadow EKF error.
    pos_t, pos = stage1.arrays(bag["/brov/debug/pos_ned"])
    vel_t, vel = stage1.arrays(bag["/brov/debug/vel_ned"])
    att_t, att = stage1.arrays(bag["/brov/debug/att_quat_ned"])
    truth_pos_t, truth_pos = stage1.arrays(bag["/brov/debug/gazebo_truth_pos_ned"])
    truth_vel_t, truth_vel = stage1.arrays(bag["/brov/debug/gazebo_truth_vel_ned"])
    truth_att_t, truth_att = stage1.arrays(bag["/brov/debug/gazebo_truth_att_quat_ned"])

    truth_pos_at_ekf, pos_valid, pos_skew = stage1.nearest(
        truth_pos_t, truth_pos, pos_t, pair_tolerance_s
    )
    truth_vel_at_ekf, vel_valid, vel_skew = stage1.nearest(
        truth_vel_t, truth_vel, vel_t, pair_tolerance_s
    )
    truth_att_at_ekf, att_valid, att_skew = stage1.nearest(
        truth_att_t, truth_att, att_t, pair_tolerance_s
    )
    pos_use = _window_mask(pos_t, start, stop, settle_s) & pos_valid
    vel_phase = _window_mask(vel_t, start, stop, settle_s)
    vel_use = vel_phase & vel_valid
    att_use = _window_mask(att_t, start, stop, settle_s) & att_valid

    if np.any(pos_use) and np.all(np.isfinite(initial_position_offset)):
        position_error = (
            pos[pos_use] - truth_pos_at_ekf[pos_use] - initial_position_offset
        )
        position_norm = np.linalg.norm(position_error, axis=1)
    else:
        position_error = np.empty((0, 3))
        position_norm = np.empty(0)
    if np.any(vel_use):
        velocity_error = vel[vel_use] - truth_vel_at_ekf[vel_use]
        velocity_norm = np.linalg.norm(velocity_error, axis=1)
    else:
        velocity_error = np.empty((0, 3))
        velocity_norm = np.empty(0)
    attitude_error = (
        stage1.q_angle_deg(att[att_use], truth_att_at_ekf[att_use])
        if np.any(att_use)
        else np.empty(0)
    )

    vel_at_pos, pv_valid, _ = stage1.nearest(vel_t, vel, pos_t, pair_tolerance_s)
    consistency_use = _window_mask(pos_t, start, stop, settle_s) & pv_valid
    return {
        "position": {
            "samples": int(position_norm.size),
            "drift_rms_m": (
                float(np.sqrt(np.mean(position_norm**2)))
                if position_norm.size
                else math.nan
            ),
            "drift_p95_m": stage1.percentile(position_norm, 95),
            "drift_end_m": float(position_norm[-1]) if position_norm.size else math.nan,
            "component_bias_ned_m": (
                np.mean(position_error, axis=0).tolist()
                if position_error.size
                else [math.nan] * 3
            ),
        },
        "velocity": {
            "samples": int(velocity_norm.size),
            "vector_rmse_mps": (
                float(np.sqrt(np.mean(velocity_norm**2)))
                if velocity_norm.size
                else math.nan
            ),
            "error_p95_mps": stage1.percentile(velocity_norm, 95),
            "component_bias_ned_mps": (
                np.mean(velocity_error, axis=0).tolist()
                if velocity_error.size
                else [math.nan] * 3
            ),
            "constant_lag_search": stage1.constant_lag_metrics(
                vel_t,
                vel,
                truth_vel_t,
                truth_vel,
                vel_phase,
                pair_tolerance_s=pair_tolerance_s,
            ),
        },
        "attitude": {
            "samples": int(attitude_error.size),
            "error_rms_deg": (
                float(np.sqrt(np.mean(attitude_error**2)))
                if attitude_error.size
                else math.nan
            ),
            "error_p95_deg": stage1.percentile(attitude_error, 95),
            "error_max_deg": stage1.percentile(attitude_error, 100),
        },
        "position_velocity_consistency": stage1.consistency_metrics(
            pos_t[consistency_use], pos[consistency_use], vel_at_pos[consistency_use]
        ),
        "pairing_skew_p95_ms": {
            "position": 1.0e3
            * stage1.percentile(np.abs(pos_skew[pos_valid & pos_use]), 95),
            "velocity": 1.0e3
            * stage1.percentile(np.abs(vel_skew[vel_valid & vel_phase]), 95),
            "attitude": 1.0e3
            * stage1.percentile(
                np.abs(att_skew[att_valid & _window_mask(att_t, start, stop, settle_s)]),
                95,
            ),
        },
    }


def _physical_attitude_metrics(
    bag: dict[str, Any],
    start: float,
    stop: float,
    settle_s: float,
    active_start: float,
    pair_tolerance_s: float,
) -> dict:
    qd_t, qd = stage1.arrays(bag["/brov/debug/q_desired_zup"])
    truth_t, truth = stage1.arrays(bag["/brov/debug/gazebo_truth_att_quat_ned"])
    selected_t, selected = stage1.arrays(bag["/brov/debug/feedback_att_quat_ned"])
    if qd.size == 0 or truth.size == 0 or selected.size == 0:
        return {"samples": 0}
    first_candidates = np.flatnonzero(selected_t >= active_start)
    if first_candidates.size == 0:
        return {"samples": 0}
    yaw0 = stage1.yaw_from_q(selected[first_candidates[0]])
    truth_at_qd, valid, skew = stage1.nearest(
        truth_t, truth, qd_t, pair_tolerance_s
    )
    use = _window_mask(qd_t, start, stop, settle_s) & valid
    if not np.any(use):
        return {"samples": 0}
    count = int(np.count_nonzero(use))
    q_frame = stage1.q_mul(stage1.q_yaw(-yaw0, count), truth_at_qd[use])
    q_m = np.repeat(np.asarray([[0.0, 1.0, 0.0, 0.0]]), count, axis=0)
    truth_zup = stage1.q_mul(q_frame, q_m)
    error = stage1.q_angle_deg(qd[use], truth_zup)
    return {
        "samples": count,
        "error_rms_deg": float(np.sqrt(np.mean(error**2))),
        "error_mean_deg": float(np.mean(error)),
        "error_p95_deg": stage1.percentile(error, 95),
        "error_max_deg": stage1.percentile(error, 100),
        "pairing_skew_p95_ms": 1.0e3 * stage1.percentile(np.abs(skew[use]), 95),
    }


def _path_metrics(
    bag: dict[str, Any],
    start: float,
    stop: float,
    settle_s: float,
    active_start: float,
) -> dict:
    truth_t, truth = stage1.arrays(bag["/brov/debug/gazebo_truth_pos_ned"])
    selected_pos_t, selected_pos = stage1.arrays(
        bag["/brov/debug/feedback_pos_ned"]
    )
    selected_q_t, selected_q = stage1.arrays(
        bag["/brov/debug/feedback_att_quat_ned"]
    )
    target_t, target = stage1.arrays(bag["/brov/target_waypoint"])
    if truth.size == 0 or selected_pos.size == 0 or selected_q.size == 0:
        return {"samples": 0}
    selected_pos_active = np.flatnonzero(selected_pos_t >= active_start)
    selected_active = np.flatnonzero(selected_q_t >= active_start)
    truth_active = np.flatnonzero(truth_t >= active_start)
    use = _window_mask(truth_t, start, stop, settle_s)
    if (
        not np.any(use)
        or selected_pos_active.size == 0
        or selected_active.size == 0
        or truth_active.size == 0
    ):
        return {"samples": 0}

    # Physical path error is mission-relative, so each run's physical truth
    # must start at zero.  Using the EKF local origin here would mix an absolute
    # Gazebo depth with a start-relative waypoint and manufacture a multi-metre
    # depth error.  EKF-vs-GT offset/drift remains explicit in estimator metrics.
    origin_ned = truth[truth_active[0]]
    yaw0 = stage1.yaw_from_q(selected_q[selected_active[0]])
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    ned_to_start_heading = np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    # The runtime guidance frame is start-heading NED (+Y right, +Z down),
    # while physical reporting here is FLU/Z-up.  T3 is the same proper
    # 180-degree X rotation used by ObservationBuilder for the policy body
    # convention.  Apply it to both truth and waypoint; flipping truth alone
    # would manufacture a 0.4 m depth error for the +0.20 m-down mission.
    start_heading_ned_to_zup = np.asarray([1.0, -1.0, -1.0])
    mission_ned = (truth - origin_ned) @ ned_to_start_heading.T
    mission_zup = mission_ned * start_heading_ned_to_zup
    path = mission_zup[use]
    path_length = (
        float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        if path.shape[0] >= 2
        else 0.0
    )
    target_zup = np.full(3, np.nan)
    target_use = _window_mask(target_t, start, stop)
    if target.ndim == 2 and target.shape[1] >= 3 and np.any(target_use):
        target_ned = np.median(target[target_use, :3], axis=0)
        target_zup = target_ned * start_heading_ned_to_zup
    cross_track_error = (
        path[:, 1] - target_zup[1]
        if math.isfinite(target_zup[1])
        else np.empty(0)
    )
    depth_error = (
        path[:, 2] - target_zup[2]
        if math.isfinite(target_zup[2])
        else np.empty(0)
    )
    return {
        "frame": "start_heading_zup_flu",
        "origin_gazebo_truth_position_ned_m": origin_ned.tolist(),
        "samples": int(path.shape[0]),
        "along_track_start_end_m": [float(path[0, 0]), float(path[-1, 0])],
        "along_track_delta_m": float(path[-1, 0] - path[0, 0]),
        "path_length_m": path_length,
        "target_waypoint_mission_zup_m": target_zup.tolist(),
        "cross_track_rms_m": (
            float(np.sqrt(np.mean(cross_track_error**2)))
            if cross_track_error.size
            else math.nan
        ),
        "cross_track_max_abs_m": (
            float(np.max(np.abs(cross_track_error)))
            if cross_track_error.size
            else math.nan
        ),
        "target_vertical_mission_zup_m": float(target_zup[2]),
        "target_depth_down_m": float(-target_zup[2]),
        "depth_error_rms_m": (
            float(np.sqrt(np.mean(depth_error**2))) if depth_error.size else math.nan
        ),
        "depth_error_max_abs_m": (
            float(np.max(np.abs(depth_error))) if depth_error.size else math.nan
        ),
    }


def _action_and_actuation(
    bag: dict[str, Any],
    start: float,
    stop: float,
    settle_s: float,
    pair_tolerance_s: float,
) -> dict:
    action_t, action = stage1.arrays(bag["/brov/action"])
    use = _window_mask(action_t, start, stop, settle_s)
    action_s = action[use] if action.ndim == 2 else np.empty((0, 0))
    if action_s.ndim == 2 and action_s.shape[1] == 6 and action_s.shape[0]:
        cap = np.abs(action_s) >= 0.99
        any_cap = np.any(cap, axis=1)
        action_metrics = _axis_statistics(action_s, AXES)
        action_metrics.update(
            {
                "any_axis_ge_0p99_fraction": float(np.mean(any_cap)),
                "axis_ge_0p99_fraction": dict(
                    zip(AXES, np.mean(cap, axis=0).tolist())
                ),
                "longest_any_axis_ge_0p99_s": _longest_true_duration(
                    action_t[use], any_cap
                ),
                "delta_rms": dict(
                    zip(
                        AXES,
                        (
                            np.sqrt(np.mean(np.diff(action_s, axis=0) ** 2, axis=0))
                            if action_s.shape[0] >= 2
                            else np.zeros(6)
                        ).tolist(),
                    )
                ),
            }
        )
    else:
        action_metrics = {"samples": 0}

    raw_t, raw = stage1.arrays(bag["/brov/policy/action_raw"])
    raw_use = _window_mask(raw_t, start, stop, settle_s)
    raw_s = raw[raw_use] if raw.ndim == 2 else np.empty((0, 0))
    raw_metrics = _axis_statistics(raw_s, AXES)
    raw_metrics["any_axis_outside_unit_fraction"] = (
        float(np.mean(np.any(np.abs(raw_s) > 1.0, axis=1)))
        if raw_s.ndim == 2 and raw_s.shape == (raw_s.shape[0], 6) and raw_s.size
        else math.nan
    )

    requested_t, requested = stage1.arrays(
        bag["/brov/policy/thruster_force_requested"]
    )
    limited_t, limited = stage1.arrays(bag["/brov/policy/thruster_force_limited"])
    limited_at_requested, force_valid, _ = stage1.nearest(
        limited_t, limited, requested_t, pair_tolerance_s
    )
    force_use = _window_mask(requested_t, start, stop, settle_s) & force_valid
    force_delta = (
        requested[force_use] - limited_at_requested[force_use]
        if requested.ndim == 2 and requested.shape[1] == 8 and np.any(force_use)
        else np.empty((0, 8))
    )
    force_clamp = (
        np.any(np.abs(force_delta) > 1.0e-5, axis=1)
        if force_delta.size
        else np.empty(0, dtype=bool)
    )
    force_metrics = {
        "samples": int(force_delta.shape[0]),
        "any_clamp_fraction": (
            float(np.mean(force_clamp)) if force_clamp.size else math.nan
        ),
        "longest_any_clamp_s": (
            _longest_true_duration(requested_t[force_use], force_clamp)
            if force_clamp.size
            else 0.0
        ),
        "clamp_error_rms_N": (
            float(np.sqrt(np.mean(force_delta**2))) if force_delta.size else math.nan
        ),
    }

    wrench_t, wrench = stage1.arrays(bag["/brov/policy/wrench_requested"])
    achieved_t, achieved = stage1.arrays(
        bag["/brov/policy/wrench_after_thruster_limit"]
    )
    achieved_at_wrench, wrench_valid, _ = stage1.nearest(
        achieved_t, achieved, wrench_t, pair_tolerance_s
    )
    wrench_use = _window_mask(wrench_t, start, stop, settle_s) & wrench_valid
    if wrench.ndim == 2 and wrench.shape[1] == 6 and np.any(wrench_use):
        normalized_requested = wrench[wrench_use] / WRENCH_SCALE
        normalized_residual = (
            achieved_at_wrench[wrench_use] - wrench[wrench_use]
        ) / WRENCH_SCALE
        authority_residual = np.linalg.norm(normalized_residual, axis=1) / np.maximum(
            np.linalg.norm(normalized_requested, axis=1), 0.01
        )
    else:
        authority_residual = np.empty(0)
    wrench_metrics = {
        "samples": int(authority_residual.size),
        "normalized_authority_residual_mean": (
            float(np.mean(authority_residual))
            if authority_residual.size
            else math.nan
        ),
        "normalized_authority_residual_p95": stage1.percentile(
            authority_residual, 95
        ),
    }

    pwm_groups = {}
    for label, topic in (
        ("inverse_requested", "/brov/policy/thruster_pwm_requested"),
        ("shaped_preview", "/brov/policy/thruster_pwm_preview"),
        ("sent", "/brov/thruster_pwm"),
    ):
        pwm_t, pwm = stage1.arrays(bag[topic])
        pwm_use = _window_mask(pwm_t, start, stop, settle_s)
        pwm_s = pwm[pwm_use] if pwm.ndim == 2 else np.empty((0, 0))
        metrics = _axis_statistics(pwm_s, THRUSTERS)
        if pwm_s.ndim == 2 and pwm_s.shape[1] == 8 and pwm_s.shape[0]:
            pwm_cap = np.any(np.abs(pwm_s) >= 0.99, axis=1)
            metrics.update(
                {
                    "any_abs_ge_0p99_fraction": float(np.mean(pwm_cap)),
                    "longest_any_abs_ge_0p99_s": _longest_true_duration(
                        pwm_t[pwm_use], pwm_cap
                    ),
                }
            )
        pwm_groups[label] = metrics

    return {
        "action": action_metrics,
        "raw_action": raw_metrics,
        "thruster_force": force_metrics,
        "wrench": wrench_metrics,
        "pwm": pwm_groups,
        "preview_to_sent": _preview_sent_metrics(bag, start, stop, settle_s),
        "sent_to_servo": _servo_transport_metrics(bag, start, stop, settle_s),
    }


def _dvl_metrics(
    bag: dict[str, Any], start: float, stop: float, settle_s: float
) -> dict:
    stamps, sample = stage1.arrays(bag["/brov/stage2/dvl_sample"])
    use = _window_mask(stamps, start, stop, settle_s)
    if sample.ndim != 2 or sample.shape[1] < 21 or not np.any(use):
        return {"samples": 0}
    value = sample[use]
    realized_delay = value[:, 2] - value[:, 1]
    valid_t, valid_v = stage1.scalars(bag["/brov/stage2/dvl_valid"])
    valid_use = _window_mask(valid_t, start, stop, settle_s)
    valid_values = valid_v[valid_use].astype(bool)
    result = {
        "samples": int(value.shape[0]),
        "row_width": int(value.shape[1]),
        "recorder_rate": stage1.rate_metrics(stamps[use]),
        "source_rate": stage1.rate_metrics(value[:, 1]),
        "source_dt_median_s": float(np.median(value[:, 3])),
        "realized_delay_mean_s": float(np.mean(realized_delay)),
        "realized_delay_p95_s": stage1.percentile(realized_delay, 95),
        "fom_mps_min_max": [float(np.min(value[:, 18])), float(np.max(value[:, 18]))],
        "confidence_min_max": [
            float(np.min(value[:, 19])),
            float(np.max(value[:, 19])),
        ],
        "valid_event_count": int(valid_values.size),
        "invalid_event_fraction": (
            float(np.mean(~valid_values)) if valid_values.size else math.nan
        ),
        "invalid_event_count": int(np.count_nonzero(~valid_values)),
    }
    if value.shape[1] >= 23:
        result["altitude_m_min_max"] = [
            float(np.min(value[:, 21])),
            float(np.max(value[:, 21])),
        ]
        result["rangefinder_sent_fraction"] = float(np.mean(value[:, 22] > 0.5))
    return result


def _turn_jumps(
    bag: dict[str, Any], transition_s: float, pair_tolerance_s: float
) -> dict:
    def jump(topic: str, quaternion: bool = False) -> tuple[Any, float, float]:
        stamps, values = stage1.arrays(bag[topic])
        # q_d/v_d are published immediately before waypoint_idx on the same
        # observation tick.  Comparing samples on either side of the index
        # timestamp can therefore select two post-transition samples.  Find
        # the largest adjacent command jump in a narrow edge-local window.
        radius_s = max(0.10, 2.0 * pair_tolerance_s)
        candidates = np.flatnonzero(
            (stamps[:-1] >= transition_s - radius_s)
            & (stamps[1:] <= transition_s + radius_s)
        )
        if candidates.size == 0:
            return math.nan, math.nan, math.nan
        if quaternion:
            magnitudes = stage1.q_angle_deg(
                values[candidates], values[candidates + 1]
            )
            local = int(np.argmax(magnitudes))
            magnitude: Any = float(magnitudes[local])
        else:
            deltas = values[candidates + 1] - values[candidates]
            norms = np.linalg.norm(deltas, axis=1)
            local = int(np.argmax(norms))
            delta = deltas[local]
            magnitude = {
                "vector": delta.tolist(),
                "norm": float(np.linalg.norm(delta)),
            }
        left = int(candidates[local])
        right = left + 1
        return magnitude, float(stamps[left] - transition_s), float(
            stamps[right] - transition_s
        )

    velocity, velocity_left, velocity_right = jump(
        "/brov/debug/v_desired_body_zup"
    )
    attitude, attitude_left, attitude_right = jump(
        "/brov/debug/q_desired_zup", quaternion=True
    )
    action, action_left, action_right = jump("/brov/action")
    return {
        "desired_velocity_jump": velocity,
        "desired_attitude_geodesic_jump_deg": attitude,
        "action_jump": action,
        "sample_offsets_s": {
            "desired_velocity": [velocity_left, velocity_right],
            "desired_attitude": [attitude_left, attitude_right],
            "action": [action_left, action_right],
        },
        "pair_tolerance_s": pair_tolerance_s,
    }


def _phase_metrics(
    bag: dict[str, Any],
    name: str,
    start: float,
    stop: float,
    settle_s: float,
    active_start: float,
    pair_tolerance_s: float,
    initial_position_offset: np.ndarray,
    end_excluded_s: float = 0.0,
) -> dict:
    analysis_stop = max(start, stop - max(0.0, end_excluded_s))
    return {
        "name": name,
        "start_unix_s": start,
        "stop_unix_s": stop,
        "duration_s": stop - start,
        "settle_excluded_s": settle_s,
        "end_excluded_s": end_excluded_s,
        "analysis_window_s": [start + settle_s, analysis_stop],
        "velocity_tracking": _velocity_tracking(
            bag, start, analysis_stop, settle_s, pair_tolerance_s
        ),
        "gazebo_path": _path_metrics(
            bag, start, analysis_stop, settle_s, active_start
        ),
        "gazebo_attitude": _physical_attitude_metrics(
            bag,
            start,
            analysis_stop,
            settle_s,
            active_start,
            pair_tolerance_s,
        ),
        "raw_ekf_vs_gazebo_truth": _estimator_metrics(
            bag,
            start,
            analysis_stop,
            settle_s,
            pair_tolerance_s,
            initial_position_offset,
        ),
        "action_and_actuation": _action_and_actuation(
            bag, start, analysis_stop, settle_s, pair_tolerance_s
        ),
        "dvl": _dvl_metrics(bag, start, analysis_stop, settle_s),
    }


def _phase_gates(phase: dict) -> dict:
    truth = phase["velocity_tracking"]["gazebo_ground_truth"]
    path = phase["gazebo_path"]
    attitude = phase["gazebo_attitude"]
    action = phase["action_and_actuation"]["action"]
    force = phase["action_and_actuation"]["thruster_force"]
    estimator = phase["raw_ekf_vs_gazebo_truth"]["velocity"]
    dvl = phase["dvl"]
    samples = int(truth.get("samples", 0))
    v_parallel = _safe_float(truth.get("v_parallel_mean_mps"))
    vector_rmse = _safe_float(truth.get("vector_error_rmse_mps"))
    cross_rms = _safe_float(path.get("cross_track_rms_m"))
    cross_max = _safe_float(path.get("cross_track_max_abs_m"))
    depth_rms = _safe_float(path.get("depth_error_rms_m"))
    depth_max = _safe_float(path.get("depth_error_max_abs_m"))
    attitude_max = _safe_float(attitude.get("error_max_deg"))
    action_cap = _safe_float(action.get("any_axis_ge_0p99_fraction"))
    force_clamp = _safe_float(force.get("any_clamp_fraction"))
    ekf_rmse = _safe_float(estimator.get("vector_rmse_mps"))
    dvl_width = int(dvl.get("row_width", 0))
    dvl_rate = _safe_float(dvl.get("source_rate", {}).get("rate_hz"))
    dvl_delay = _safe_float(dvl.get("realized_delay_p95_s"))
    dvl_invalid = int(dvl.get("invalid_event_count", -1))
    dvl_fom = dvl.get("fom_mps_min_max", [math.nan, math.nan])
    dvl_confidence = dvl.get("confidence_min_max", [math.nan, math.nan])
    altitude = dvl.get("altitude_m_min_max", [math.nan, math.nan])
    rangefinder_fraction = _safe_float(dvl.get("rangefinder_sent_fraction"))
    gates = {
        "coverage": _gate(samples, samples >= 20, ">=20 synchronized steady samples"),
        "v_parallel": _gate(
            v_parallel,
            math.isfinite(v_parallel) and 0.45 <= v_parallel <= 0.55,
            "0.45 <= mean <= 0.55 m/s",
            evaluated=math.isfinite(v_parallel),
        ),
        "velocity_vector_rmse": _gate(
            vector_rmse,
            math.isfinite(vector_rmse) and vector_rmse <= 0.08,
            "<=0.08 m/s",
            evaluated=math.isfinite(vector_rmse),
        ),
        "cross_track_rms": _gate(
            cross_rms,
            math.isfinite(cross_rms) and cross_rms <= 0.15,
            "<=0.15 m",
            evaluated=math.isfinite(cross_rms),
        ),
        "cross_track_max": _gate(
            cross_max,
            math.isfinite(cross_max) and cross_max <= 0.30,
            "<=0.30 m",
            evaluated=math.isfinite(cross_max),
        ),
        "depth_rms": _gate(
            depth_rms,
            math.isfinite(depth_rms) and depth_rms <= 0.10,
            "<=0.10 m",
            evaluated=math.isfinite(depth_rms),
        ),
        "depth_max": _gate(
            depth_max,
            math.isfinite(depth_max) and depth_max <= 0.20,
            "<=0.20 m",
            evaluated=math.isfinite(depth_max),
        ),
        "attitude_max": _gate(
            attitude_max,
            math.isfinite(attitude_max) and attitude_max <= 10.0,
            "<=10 deg outside the settle window",
            evaluated=math.isfinite(attitude_max),
        ),
        "steady_action_cap": _gate(
            action_cap,
            math.isfinite(action_cap) and action_cap < 0.01,
            "<1% samples with any |action|>=0.99",
            evaluated=math.isfinite(action_cap),
        ),
        "steady_force_clamp": _gate(
            force_clamp,
            math.isfinite(force_clamp) and force_clamp < 0.01,
            "<1% requested-force clamp",
            evaluated=math.isfinite(force_clamp),
        ),
        "raw_ekf_velocity_rmse": _gate(
            ekf_rmse,
            math.isfinite(ekf_rmse) and ekf_rmse <= 0.03,
            "<=0.03 m/s",
            evaluated=math.isfinite(ekf_rmse),
        ),
        "dvl_schema": _gate(
            dvl_width,
            dvl_width == 23,
            "exact 23-column Water Linked diagnostic schema",
        ),
        "dvl_source_rate": _gate(
            dvl_rate,
            math.isfinite(dvl_rate) and 4.5 <= dvl_rate <= 10.5,
            "Water Linked auto-range source rate in [4.5,10.5] Hz",
            evaluated=math.isfinite(dvl_rate),
        ),
        "dvl_artificial_delay": _gate(
            dvl_delay,
            math.isfinite(dvl_delay) and abs(dvl_delay) <= 0.025,
            "no artificial queue delay (p95 <=25 ms scheduling tolerance)",
            evaluated=math.isfinite(dvl_delay),
        ),
        "dvl_validity": _gate(
            dvl_invalid,
            dvl_invalid == 0,
            "zero invalid DVL events",
            evaluated=dvl_invalid >= 0,
        ),
        "dvl_fom": _gate(
            dvl_fom,
            len(dvl_fom) == 2
            and np.allclose(np.asarray(dvl_fom, dtype=float), 0.0),
            "controlled driver baseline FOM exactly zero",
        ),
        "dvl_confidence": _gate(
            dvl_confidence,
            len(dvl_confidence) == 2
            and np.allclose(np.asarray(dvl_confidence, dtype=float), 100.0),
            "controlled driver baseline confidence exactly 100",
        ),
        "dvl_altitude": _gate(
            altitude,
            len(altitude) == 2
            and math.isfinite(float(altitude[0]))
            and math.isfinite(float(altitude[1]))
            and 0.3 <= float(altitude[0]) <= float(altitude[1]) <= 14.0,
            "flat-bottom altitude remains in Water Linked mode-1/2 range [0.3,14] m",
        ),
        "dvl_rangefinder": _gate(
            rangefinder_fraction,
            math.isfinite(rangefinder_fraction)
            and rangefinder_fraction == 1.0,
            "DISTANCE_SENSOR emitted for every valid DVL packet",
            evaluated=math.isfinite(rangefinder_fraction),
        ),
    }
    gates["pass"] = _all_gates_pass(gates.values())
    return gates


def analyze_one(
    uri: str,
    *,
    expected_feedback_source: str,
    settle_s: float,
    turn_half_window_s: float,
    pair_tolerance_s: float,
    debounce_samples: int,
) -> dict:
    bag = read_bag(uri)
    active_runs = _active_runs(bag["/brov/control_active"])
    if not active_runs:
        raise RuntimeError(f"{uri}: bag contains no active control interval")
    start, stop = max(active_runs, key=lambda pair: pair[1] - pair[0])
    rle = _index_rle(
        bag["/brov/waypoint_idx"], start, stop, debounce_samples
    )
    sequence = [int(item["value"]) for item in rle]
    exact_sequence = sequence == EXPECTED_RLE
    source_values = [str(value) for _, value in bag["/brov/debug/feedback_source"]]
    feedback_source = source_values[-1] if source_values else "missing"
    missing_core = sorted(CORE_REQUIRED - set(bag["_present"]))
    missing_dvl = sorted(DVL_REQUIRED - set(bag["_present"]))

    result: dict[str, Any] = {
        "bag": str(Path(uri).resolve()),
        "feedback_source": feedback_source,
        "expected_feedback_source": expected_feedback_source,
        "active": {
            "start_unix_s": start,
            "stop_unix_s": stop,
            "duration_s": stop - start,
            "active_run_count": len(active_runs),
        },
        "waypoint_rle": {
            "expected": EXPECTED_RLE,
            "observed": sequence,
            "runs": rle,
            "debounce_samples": debounce_samples,
            "exact_match": exact_sequence,
            "semantic_note": (
                "idx 0=takeoff, first idx 1=outbound, idx 2=return; "
                "final idx 1 is the return-arrival/next-outbound sentinel"
            ),
        },
        "topic_contract": {
            "missing_core": missing_core,
            "missing_dvl": missing_dvl,
        },
    }

    contract_gates = {
        "one_active_run": _gate(
            len(active_runs), len(active_runs) == 1, "exactly one active interval"
        ),
        "feedback_source": _gate(
            feedback_source,
            feedback_source == expected_feedback_source,
            f"exactly {expected_feedback_source!r}",
        ),
        "required_topics": _gate(
            {"core": missing_core, "dvl": missing_dvl},
            not missing_core and not missing_dvl,
            "no required topic missing",
        ),
        "waypoint_rle": _gate(
            sequence, exact_sequence, "exact run-length encoding [0,1,2,1]"
        ),
    }

    if not exact_sequence:
        contract_gates["pass"] = _all_gates_pass(contract_gates.values())
        result["gates"] = {"contract": contract_gates, "overall_pass": False}
        result["phases"] = {}
        result["turn"] = {}
        return result

    t0 = max(start, float(rle[0]["entry_s"]))
    t01 = float(rle[1]["entry_s"])
    t12 = float(rle[2]["entry_s"])
    t21 = float(rle[3]["entry_s"])
    initial_offset = _initial_position_offset(
        bag, t0, t21, pair_tolerance_s
    )
    phases = {
        "takeoff": _phase_metrics(
            bag,
            "takeoff",
            t0,
            t01,
            0.0,
            t0,
            pair_tolerance_s,
            initial_offset,
        ),
        "outbound": _phase_metrics(
            bag,
            "outbound",
            t01,
            t12,
            settle_s,
            t0,
            pair_tolerance_s,
            initial_offset,
            end_excluded_s=turn_half_window_s,
        ),
        "return": _phase_metrics(
            bag,
            "return",
            t12,
            t21,
            max(settle_s, turn_half_window_s),
            t0,
            pair_tolerance_s,
            initial_offset,
            # q_desired is published just before the 2->1 waypoint-index edge
            # on the same control tick; trim one tick so the next outbound
            # command is not attributed to return steady-state tracking.
            end_excluded_s=max(0.05, pair_tolerance_s),
        ),
    }
    turn_start = max(t0, t12 - turn_half_window_s)
    turn_stop = min(t21, t12 + turn_half_window_s)
    turn = _phase_metrics(
        bag,
        "turn_window",
        turn_start,
        turn_stop,
        0.0,
        t0,
        pair_tolerance_s,
        initial_offset,
    )
    turn["transition_unix_s"] = t12
    turn["relative_window_s"] = [turn_start - t12, turn_stop - t12]
    turn["jumps"] = _turn_jumps(bag, t12, pair_tolerance_s)

    outbound_gates = _phase_gates(phases["outbound"])
    return_gates = _phase_gates(phases["return"])
    action_full = _action_and_actuation(
        bag, t0, t21, 0.0, pair_tolerance_s
    )
    full_action_cap = _safe_float(
        action_full["action"].get("any_axis_ge_0p99_fraction")
    )
    full_force_clamp = _safe_float(
        action_full["thruster_force"].get("any_clamp_fraction")
    )
    action_t, _ = stage1.arrays(bag["/brov/action"])
    action_rate = stage1.rate_metrics(action_t[_window_mask(action_t, t0, t21)])
    rate_hz = _safe_float(action_rate.get("rate_hz"))
    gap_p95_ms = _safe_float(action_rate.get("gap_p95_ms"))
    gap_max_ms = _safe_float(action_rate.get("gap_max_ms"))
    lifecycle = _lifecycle_metrics(bag, t21, stop)
    neutral_delay = _safe_float(lifecycle.get("first_neutral_echo_delay_s"))
    sent_after_inactive = int(lifecycle["sent_samples_after_inactive"])
    cycle_gates = {
        "full_cycle_reached": _gate(True, True, "2->1 return-arrival edge observed"),
        "whole_cycle_action_cap": _gate(
            full_action_cap,
            math.isfinite(full_action_cap) and full_action_cap < 0.05,
            "<5% samples with any |action|>=0.99",
            evaluated=math.isfinite(full_action_cap),
        ),
        "whole_cycle_force_clamp": _gate(
            full_force_clamp,
            math.isfinite(full_force_clamp) and full_force_clamp < 0.05,
            "<5% requested-force clamp",
            evaluated=math.isfinite(full_force_clamp),
        ),
        "action_rate_hz": _gate(
            rate_hz,
            math.isfinite(rate_hz) and 24.0 <= rate_hz <= 26.0,
            "24-26 Hz",
            evaluated=math.isfinite(rate_hz),
        ),
        "action_gap_p95": _gate(
            gap_p95_ms,
            math.isfinite(gap_p95_ms) and gap_p95_ms <= 60.0,
            "<=60 ms",
            evaluated=math.isfinite(gap_p95_ms),
        ),
        "action_gap_max": _gate(
            gap_max_ms,
            math.isfinite(gap_max_ms) and gap_max_ms <= 120.0,
            "<=120 ms (0.25 s watchdog remains separate)",
            evaluated=math.isfinite(gap_max_ms),
        ),
        "inactive_sent_count": _gate(
            sent_after_inactive,
            sent_after_inactive == 0,
            "zero /brov/thruster_pwm samples after control_active=false",
        ),
        "neutral_echo": _gate(
            neutral_delay,
            math.isfinite(neutral_delay) and 0.0 <= neutral_delay <= 0.25,
            "first all-channel 1500+/-2 us echo within 0.25 s",
            evaluated=math.isfinite(neutral_delay),
        ),
        "cycle_edge_to_inactive": _gate(
            _safe_float(lifecycle.get("cycle_edge_to_inactive_s")),
            math.isfinite(_safe_float(lifecycle.get("cycle_edge_to_inactive_s")))
            and _safe_float(lifecycle.get("cycle_edge_to_inactive_s")) <= 0.25,
            "control inactive within 0.25 s of the 2->1 cycle edge",
            evaluated=math.isfinite(
                _safe_float(lifecycle.get("cycle_edge_to_inactive_s"))
            ),
        ),
    }
    contract_gates["pass"] = _all_gates_pass(contract_gates.values())
    cycle_gates["pass"] = _all_gates_pass(cycle_gates.values())
    result.update(
        {
            "phase_edges_unix_s": {
                "active": t0,
                "takeoff_to_outbound_0_to_1": t01,
                "outbound_to_return_1_to_2": t12,
                "return_arrival_2_to_1": t21,
            },
            "initial_state": _initial_state_metrics(bag, t0),
            "phases": phases,
            "turn": turn,
            "whole_cycle_action_and_actuation": action_full,
            "lifecycle": lifecycle,
            "timing": {"action_topic": action_rate},
            "gates": {
                "contract": contract_gates,
                "outbound": outbound_gates,
                "return": return_gates,
                "cycle": cycle_gates,
                "overall_pass": bool(
                    contract_gates["pass"]
                    and outbound_gates["pass"]
                    and return_gates["pass"]
                    and cycle_gates["pass"]
                ),
            },
        }
    )
    return result


def _metric_delta(first: dict, second: dict, keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        a = _safe_float(first.get(key))
        b = _safe_float(second.get(key))
        result[key] = {
            "gt": a,
            "dvl_ekf": b,
            "dvl_minus_gt": b - a if math.isfinite(a) and math.isfinite(b) else math.nan,
            "dvl_over_gt": stage1.ratio(b, a),
        }
    return result


def _equivalence_gate(
    gt_value: float,
    dvl_value: float,
    *,
    additive: float,
    multiplier: float = 1.2,
    criterion_unit: str,
) -> dict:
    threshold = (
        max(multiplier * gt_value, gt_value + additive)
        if math.isfinite(gt_value)
        else math.nan
    )
    return _gate(
        dvl_value,
        math.isfinite(dvl_value)
        and math.isfinite(threshold)
        and dvl_value <= threshold,
        f"DVL <= max({multiplier:g}*GT, GT+{additive:g} {criterion_unit}); threshold={threshold:g}",
        evaluated=math.isfinite(dvl_value) and math.isfinite(threshold),
    )


def compare(gt: dict, dvl: dict) -> dict:
    if not gt.get("phases") or not dvl.get("phases"):
        return {
            "available": False,
            "reason": "both bags must satisfy exact waypoint RLE [0,1,2,1]",
            "overall_pass": False,
        }
    gt_initial = gt["initial_state"]
    dvl_initial = dvl["initial_state"]
    gt_position = np.asarray(gt_initial["gazebo_truth_position_ned_m"], dtype=float)
    dvl_position = np.asarray(dvl_initial["gazebo_truth_position_ned_m"], dtype=float)
    gt_velocity = np.asarray(gt_initial["gazebo_truth_velocity_ned_mps"], dtype=float)
    dvl_velocity = np.asarray(dvl_initial["gazebo_truth_velocity_ned_mps"], dtype=float)
    gt_attitude = np.asarray(gt_initial["gazebo_truth_attitude_wxyz"], dtype=float)
    dvl_attitude = np.asarray(dvl_initial["gazebo_truth_attitude_wxyz"], dtype=float)
    position_delta = (
        float(np.linalg.norm(dvl_position - gt_position))
        if np.all(np.isfinite(gt_position)) and np.all(np.isfinite(dvl_position))
        else math.nan
    )
    velocity_delta = (
        float(np.linalg.norm(dvl_velocity - gt_velocity))
        if np.all(np.isfinite(gt_velocity)) and np.all(np.isfinite(dvl_velocity))
        else math.nan
    )
    attitude_delta = (
        float(
            stage1.q_angle_deg(
                gt_attitude.reshape(1, 4), dvl_attitude.reshape(1, 4)
            )[0]
        )
        if np.all(np.isfinite(gt_attitude)) and np.all(np.isfinite(dvl_attitude))
        else math.nan
    )
    gt_dvl_sequence = gt_initial.get("dvl_sequence")
    dvl_dvl_sequence = dvl_initial.get("dvl_sequence")
    sequence_delta = (
        abs(int(dvl_dvl_sequence) - int(gt_dvl_sequence))
        if gt_dvl_sequence is not None and dvl_dvl_sequence is not None
        else None
    )
    gt_dvl_source_time = _safe_float(gt_initial.get("dvl_source_time_s"))
    dvl_dvl_source_time = _safe_float(dvl_initial.get("dvl_source_time_s"))
    source_time_delta = (
        abs(dvl_dvl_source_time - gt_dvl_source_time)
        if math.isfinite(gt_dvl_source_time)
        and math.isfinite(dvl_dvl_source_time)
        else math.nan
    )
    initial_gates = {
        "position_delta": _gate(
            position_delta,
            math.isfinite(position_delta) and position_delta <= 0.02,
            "fresh-run GT start positions differ by <=0.02 m",
            evaluated=math.isfinite(position_delta),
        ),
        "velocity_delta": _gate(
            velocity_delta,
            math.isfinite(velocity_delta) and velocity_delta <= 0.02,
            "fresh-run GT start velocities differ by <=0.02 m/s",
            evaluated=math.isfinite(velocity_delta),
        ),
        "attitude_delta": _gate(
            attitude_delta,
            math.isfinite(attitude_delta) and attitude_delta <= 2.0,
            "fresh-run GT start attitudes differ by <=2 deg",
            evaluated=math.isfinite(attitude_delta),
        ),
        "dvl_source_time_delta": _gate(
            source_time_delta,
            math.isfinite(source_time_delta) and source_time_delta <= 0.05,
            "DVL source time at START differs by <=0.05 s",
            evaluated=math.isfinite(source_time_delta),
        ),
    }
    initial_gates["pass"] = _all_gates_pass(initial_gates.values())
    phases = {}
    paired_gates: dict[str, dict] = {}
    for name in ("outbound", "return"):
        gt_phase = gt["phases"][name]
        dvl_phase = dvl["phases"][name]
        gt_truth = gt_phase["velocity_tracking"]["gazebo_ground_truth"]
        dvl_truth = dvl_phase["velocity_tracking"]["gazebo_ground_truth"]
        gt_path = gt_phase["gazebo_path"]
        dvl_path = dvl_phase["gazebo_path"]
        gt_att = gt_phase["gazebo_attitude"]
        dvl_att = dvl_phase["gazebo_attitude"]
        gt_action = gt_phase["action_and_actuation"]["action"]
        dvl_action = dvl_phase["action_and_actuation"]["action"]
        gt_force = gt_phase["action_and_actuation"]["thruster_force"]
        dvl_force = dvl_phase["action_and_actuation"]["thruster_force"]
        phases[name] = {
            "duration_s": {
                "gt": gt_phase["duration_s"],
                "dvl_ekf": dvl_phase["duration_s"],
                "dvl_minus_gt": dvl_phase["duration_s"] - gt_phase["duration_s"],
                "dvl_over_gt": stage1.ratio(
                    dvl_phase["duration_s"], gt_phase["duration_s"]
                ),
            },
            "physical_velocity": _metric_delta(
                gt_truth,
                dvl_truth,
                (
                    "v_parallel_mean_mps",
                    "vector_error_rmse_mps",
                    "cross_speed_rms_mps",
                ),
            ),
            "gazebo_path": _metric_delta(
                gt_path,
                dvl_path,
                (
                    "cross_track_rms_m",
                    "cross_track_max_abs_m",
                    "depth_error_rms_m",
                    "depth_error_max_abs_m",
                ),
            ),
            "gazebo_attitude": _metric_delta(
                gt_att, dvl_att, ("error_rms_deg", "error_p95_deg", "error_max_deg")
            ),
            "action_cap": _metric_delta(
                gt_action, dvl_action, ("any_axis_ge_0p99_fraction",)
            ),
            "force_clamp": _metric_delta(
                gt_force, dvl_force, ("any_clamp_fraction",)
            ),
        }
        gt_v_rmse = _safe_float(gt_truth.get("vector_error_rmse_mps"))
        dvl_v_rmse = _safe_float(dvl_truth.get("vector_error_rmse_mps"))
        gt_cross = _safe_float(gt_path.get("cross_track_rms_m"))
        dvl_cross = _safe_float(dvl_path.get("cross_track_rms_m"))
        gt_depth = _safe_float(gt_path.get("depth_error_rms_m"))
        dvl_depth = _safe_float(dvl_path.get("depth_error_rms_m"))
        gt_att_rms = _safe_float(gt_att.get("error_rms_deg"))
        dvl_att_rms = _safe_float(dvl_att.get("error_rms_deg"))
        v_parallel_delta = abs(
            _safe_float(dvl_truth.get("v_parallel_mean_mps"))
            - _safe_float(gt_truth.get("v_parallel_mean_mps"))
        )
        duration_ratio = stage1.ratio(
            dvl_phase["duration_s"], gt_phase["duration_s"]
        )
        action_delta = _safe_float(
            dvl_action.get("any_axis_ge_0p99_fraction")
        ) - _safe_float(gt_action.get("any_axis_ge_0p99_fraction"))
        force_delta = _safe_float(dvl_force.get("any_clamp_fraction")) - _safe_float(
            gt_force.get("any_clamp_fraction")
        )
        gates = {
            "velocity_vector_rmse": _equivalence_gate(
                gt_v_rmse,
                dvl_v_rmse,
                additive=0.02,
                criterion_unit="m/s",
            ),
            "cross_track_rms": _equivalence_gate(
                gt_cross,
                dvl_cross,
                additive=0.05,
                criterion_unit="m",
            ),
            "depth_rms": _equivalence_gate(
                gt_depth,
                dvl_depth,
                additive=0.05,
                criterion_unit="m",
            ),
            "attitude_rmse": _equivalence_gate(
                gt_att_rms,
                dvl_att_rms,
                additive=2.0,
                criterion_unit="deg",
            ),
            "v_parallel_delta": _gate(
                v_parallel_delta,
                math.isfinite(v_parallel_delta) and v_parallel_delta <= 0.05,
                "absolute DVL-GT delta <=0.05 m/s",
                evaluated=math.isfinite(v_parallel_delta),
            ),
            "duration_ratio": _gate(
                duration_ratio,
                math.isfinite(duration_ratio) and 0.90 <= duration_ratio <= 1.10,
                "0.90 <= DVL/GT <= 1.10",
                evaluated=math.isfinite(duration_ratio),
            ),
            "action_cap_degradation": _gate(
                action_delta,
                math.isfinite(action_delta) and action_delta <= 0.05,
                "DVL-GT <=0.05 fraction",
                evaluated=math.isfinite(action_delta),
            ),
            "force_clamp_degradation": _gate(
                force_delta,
                math.isfinite(force_delta) and force_delta <= 0.05,
                "DVL-GT <=0.05 fraction",
                evaluated=math.isfinite(force_delta),
            ),
        }
        gates["pass"] = _all_gates_pass(gates.values())
        paired_gates[name] = gates

    turn_gt = gt["turn"]
    turn_dvl = dvl["turn"]
    turn = {
        "duration_s": {
            "gt": turn_gt["duration_s"],
            "dvl_ekf": turn_dvl["duration_s"],
        },
        "desired_attitude_jump_deg": {
            "gt": turn_gt["jumps"]["desired_attitude_geodesic_jump_deg"],
            "dvl_ekf": turn_dvl["jumps"]["desired_attitude_geodesic_jump_deg"],
        },
        "gazebo_attitude": _metric_delta(
            turn_gt["gazebo_attitude"],
            turn_dvl["gazebo_attitude"],
            ("error_rms_deg", "error_p95_deg", "error_max_deg"),
        ),
        "action_cap": _metric_delta(
            turn_gt["action_and_actuation"]["action"],
            turn_dvl["action_and_actuation"]["action"],
            ("any_axis_ge_0p99_fraction",),
        ),
    }
    all_paired = [paired_gates[name] for name in ("outbound", "return")]
    return {
        "available": True,
        "order": [gt["feedback_source"], dvl["feedback_source"]],
        "initial_condition_deltas": {
            "position_norm_m": position_delta,
            "velocity_norm_mps": velocity_delta,
            "attitude_geodesic_deg": attitude_delta,
            "dvl_sequence_abs_delta": sequence_delta,
            "dvl_source_time_abs_delta_s": source_time_delta,
            "dvl_sequence_note": (
                "Sequence counters are process-local diagnostics; source time is the "
                "cross-run acquisition-phase contract."
            ),
        },
        "phases": phases,
        "turn": turn,
        "gates": {
            "initial_conditions": initial_gates,
            "phase_equivalence": paired_gates,
        },
        "overall_pass": bool(
            gt["gates"]["overall_pass"]
            and dvl["gates"]["overall_pass"]
            and initial_gates["pass"]
            and all(item["pass"] for item in all_paired)
        ),
        "interpretation_rule": (
            "GT absolute pass + DVL absolute/equivalence fail isolates the estimator/"
            "feedback path. Similar GT and DVL physical failures are common-mode policy/"
            "guidance/action/actuator/plant evidence, not proof that DVL is causal."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_bag", help="Case-A rosbag using gazebo_truth feedback")
    parser.add_argument("dvl_bag", help="Case-A rosbag using no-GPS DVL/EKF feedback")
    parser.add_argument("--output", help="optional strict JSON output path")
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--turn-half-window-s", type=float, default=2.0)
    parser.add_argument("--pair-tolerance-s", type=float, default=0.03)
    parser.add_argument("--debounce-samples", type=int, default=3)
    args = parser.parse_args()
    for name in ("settle_s", "turn_half_window_s", "pair_tolerance_s"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.debounce_samples < 1:
        parser.error("--debounce-samples must be >=1")

    gt = analyze_one(
        args.gt_bag,
        expected_feedback_source="gazebo_truth",
        settle_s=args.settle_s,
        turn_half_window_s=args.turn_half_window_s,
        pair_tolerance_s=args.pair_tolerance_s,
        debounce_samples=args.debounce_samples,
    )
    dvl = analyze_one(
        args.dvl_bag,
        expected_feedback_source="mavlink_ekf",
        settle_s=args.settle_s,
        turn_half_window_s=args.turn_half_window_s,
        pair_tolerance_s=args.pair_tolerance_s,
        debounce_samples=args.debounce_samples,
    )
    result = {
        "schema": "brov_stage2_case_a_gt_dvl_ab_v1",
        "contract": {
            "expected_waypoint_rle": EXPECTED_RLE,
            "settle_s": args.settle_s,
            "turn_half_window_s": args.turn_half_window_s,
            "straight_window": {
                "outbound": (
                    "exclude first settle_s and final turn_half_window_s"
                ),
                "return": (
                    "exclude first max(settle_s, turn_half_window_s) and "
                    "final max(0.05, pair_tolerance_s)"
                ),
            },
            "pair_tolerance_s": args.pair_tolerance_s,
            "debounce_samples": args.debounce_samples,
        },
        "runs": {"gazebo_truth": gt, "dvl_ekf": dvl},
        "comparison": compare(gt, dvl),
    }
    encoded = json.dumps(stage1.json_safe(result), indent=2, allow_nan=False)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

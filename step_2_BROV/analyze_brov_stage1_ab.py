#!/usr/bin/env python3
"""Compare BROV Stage-1 0.5 m/s Gazebo-feedback/EKF-feedback bags.

The script intentionally uses only topics published by ``brov_ros2`` and
standard ROS messages.  It never writes into a bag.  Values in the
``horizontal_steady`` section use waypoint index 1 and exclude the first
second after entering that index; this matches
``mission_sim2sim_straight_0p5.yaml`` (takeoff, then one 5 m straight leg).

Run after sourcing ROS and the built workspace, for example::

  source /opt/ros/humble/setup.bash
  source /home/pyo/Programing/brov_ros2-main/install/setup.bash
  /usr/bin/python3 step_2_BROV/analyze_brov_stage1_ab.py GT_BAG EKF_BAG \
      --output /tmp/sim2sim_0p5_ab.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, Int32, String


ARRAY32_TOPICS = {
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
    "/brov/debug/feedback_pos_ned",
    "/brov/debug/feedback_vel_ned",
    "/brov/debug/feedback_att_quat_ned",
    "/brov/debug/feedback_body_rates_frd",
    "/brov/debug/gazebo_truth_pos_ned",
    "/brov/debug/gazebo_truth_vel_ned",
    "/brov/debug/gazebo_truth_att_quat_ned",
    "/brov/debug/pos_mission",
    "/brov/debug/v_body_zup",
    "/brov/debug/v_desired_body_zup",
    "/brov/debug/q_desired_zup",
}
ARRAY64_TOPICS = {"/brov/debug/feedback_timing"}
BOOL_TOPICS = {"/brov/control_active", "/brov/mission_complete"}
INT_TOPICS = {"/brov/waypoint_idx"}
STRING_TOPICS = {
    "/brov/debug/feedback_source",
    "/brov/debug/feedback_timing_schema",
}
ALL_TOPICS = ARRAY32_TOPICS | ARRAY64_TOPICS | BOOL_TOPICS | INT_TOPICS | STRING_TOPICS
AXES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _reader(uri: str) -> SequentialReader:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=uri, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def read_bag(uri: str) -> dict[str, list[tuple[float, object]]]:
    out = {topic: [] for topic in ALL_TOPICS}
    reader = _reader(uri)
    present = {item.name: item.type for item in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic not in ALL_TOPICS:
            continue
        if topic in ARRAY32_TOPICS:
            value = np.asarray(
                deserialize_message(payload, Float32MultiArray).data, dtype=np.float64
            )
        elif topic in ARRAY64_TOPICS:
            value = np.asarray(
                deserialize_message(payload, Float64MultiArray).data, dtype=np.float64
            )
        elif topic in BOOL_TOPICS:
            value = bool(deserialize_message(payload, Bool).data)
        elif topic in INT_TOPICS:
            value = int(deserialize_message(payload, Int32).data)
        else:
            value = str(deserialize_message(payload, String).data)
        out[topic].append((stamp_ns * 1.0e-9, value))
    out["_present"] = present
    return out


def longest_active_run(events: list[tuple[float, object]]) -> tuple[float, float]:
    if not events:
        raise RuntimeError("/brov/control_active is absent")
    runs: list[tuple[float, float]] = []
    start = None
    last = events[0][0]
    for stamp, raw in events:
        active = bool(raw)
        if active and start is None:
            start = stamp
        elif not active and start is not None:
            runs.append((start, stamp))
            start = None
        last = stamp
    if start is not None:
        runs.append((start, last))
    if not runs:
        raise RuntimeError("bag contains no active control interval")
    return max(runs, key=lambda pair: pair[1] - pair[0])


def arrays(series: list[tuple[float, object]]) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.empty(0), np.empty((0, 0))
    return (
        np.asarray([item[0] for item in series], dtype=np.float64),
        np.stack([np.asarray(item[1], dtype=np.float64) for item in series]),
    )


def scalars(series: list[tuple[float, object]]) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.empty(0), np.empty(0)
    return (
        np.asarray([item[0] for item in series], dtype=np.float64),
        np.asarray([item[1] for item in series]),
    )


def nearest(
    source_t: np.ndarray,
    source_v: np.ndarray,
    query_t: np.ndarray,
    tolerance_s: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour join with an explicit recorder-time tolerance."""
    if source_t.size == 0 or query_t.size == 0:
        width = source_v.shape[1] if source_v.ndim == 2 else 1
        return (
            np.full((query_t.size, width), np.nan),
            np.zeros(query_t.size, dtype=bool),
            np.full(query_t.size, np.nan),
        )
    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, source_t.size - 1)
    left = np.clip(right - 1, 0, source_t.size - 1)
    use_left = np.abs(source_t[left] - query_t) <= np.abs(source_t[right] - query_t)
    index = np.where(use_left, left, right)
    skew = source_t[index] - query_t
    valid = np.abs(skew) <= tolerance_s
    result = source_v[index].copy()
    result[~valid] = np.nan
    return result, valid, skew


def interpolate(
    source_t: np.ndarray,
    source_v: np.ndarray,
    query_t: np.ndarray,
    max_bracket_s: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly sample a vector trace while rejecting gaps/extrapolation."""
    if source_t.size < 2 or query_t.size == 0:
        width = source_v.shape[1] if source_v.ndim == 2 else 1
        return np.full((query_t.size, width), np.nan), np.zeros(query_t.size, bool)
    right = np.searchsorted(source_t, query_t, side="right")
    left = right - 1
    valid = (left >= 0) & (right < source_t.size)
    left = np.clip(left, 0, source_t.size - 1)
    right = np.clip(right, 0, source_t.size - 1)
    bracket = source_t[right] - source_t[left]
    valid &= (bracket > 0.0) & (bracket <= max_bracket_s)
    fraction = np.zeros(query_t.size, dtype=np.float64)
    fraction[valid] = (
        (query_t[valid] - source_t[left[valid]]) / bracket[valid]
    )
    result = source_v[left] + fraction[:, None] * (source_v[right] - source_v[left])
    result[~valid] = np.nan
    return result, valid


def last_value(
    source_t: np.ndarray, source_v: np.ndarray, query_t: np.ndarray, default: int | bool
) -> np.ndarray:
    if source_t.size == 0:
        return np.full(query_t.size, default)
    index = np.searchsorted(source_t, query_t, side="right") - 1
    valid = index >= 0
    index = np.clip(index, 0, source_t.size - 1)
    result = source_v[index].copy()
    result[~valid] = default
    return result


def finite_rows(*values: np.ndarray) -> np.ndarray:
    mask = np.ones(values[0].shape[0], dtype=bool)
    for value in values:
        mask &= np.all(np.isfinite(value), axis=1) if value.ndim == 2 else np.isfinite(value)
    return mask


def percentile(value: np.ndarray, q: float) -> float:
    return float(np.percentile(value, q)) if value.size else math.nan


def rate_metrics(stamps: np.ndarray) -> dict:
    if stamps.size < 2:
        return {"count": int(stamps.size), "rate_hz": math.nan}
    gap = np.diff(stamps)
    return {
        "count": int(stamps.size),
        "rate_hz": float((stamps.size - 1) / (stamps[-1] - stamps[0])),
        "gap_median_ms": 1.0e3 * percentile(gap, 50),
        "gap_p95_ms": 1.0e3 * percentile(gap, 95),
        "gap_max_ms": 1.0e3 * percentile(gap, 100),
    }


def q_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)


def q_conj(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def q_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = q_normalize(q)
    zeros = np.zeros((*vector.shape[:-1], 1), dtype=np.float64)
    return q_mul(q_mul(q, np.concatenate((zeros, vector), axis=-1)), q_conj(q))[..., 1:]


def q_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = q_normalize(a), q_normalize(b)
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def yaw_from_q(q: np.ndarray) -> float:
    w, x, y, z = q_normalize(np.asarray(q, dtype=np.float64))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def q_yaw(angle: float, count: int = 1) -> np.ndarray:
    value = np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])
    return np.repeat(value[None, :], count, axis=0)


def tracking_metrics(desired: np.ndarray, actual: np.ndarray) -> dict:
    valid = finite_rows(desired, actual) & (np.linalg.norm(desired, axis=1) > 1.0e-6)
    desired, actual = desired[valid], actual[valid]
    if desired.size == 0:
        return {"samples": 0}
    dnorm = np.linalg.norm(desired, axis=1)
    anorm = np.linalg.norm(actual, axis=1)
    unit = desired / dnorm[:, None]
    parallel = np.sum(actual * unit, axis=1)
    cross = np.linalg.norm(actual - parallel[:, None] * unit, axis=1)
    error = np.linalg.norm(actual - desired, axis=1)
    command = float(np.mean(dnorm))
    return {
        "samples": int(desired.shape[0]),
        "desired_speed_mean_mps": command,
        "actual_speed_mean_mps": float(np.mean(anorm)),
        "actual_speed_p95_mps": percentile(anorm, 95),
        "actual_speed_max_mps": percentile(anorm, 100),
        "v_parallel_mean_mps": float(np.mean(parallel)),
        "v_parallel_p05_mps": percentile(parallel, 5),
        "v_parallel_p95_mps": percentile(parallel, 95),
        "v_parallel_over_command": float(np.mean(parallel) / command),
        "target_opposite_fraction": float(np.mean(parallel < 0.0)),
        "vector_error_rmse_mps": float(np.sqrt(np.mean(error**2))),
        "vector_error_rmse_over_command": float(np.sqrt(np.mean(error**2)) / command),
        "cross_speed_rms_mps": float(np.sqrt(np.mean(cross**2))),
        "cross_speed_rms_over_command": float(np.sqrt(np.mean(cross**2)) / command),
    }


def consistency_metrics(stamps: np.ndarray, position: np.ndarray, velocity: np.ndarray) -> dict:
    valid = finite_rows(position, velocity)
    stamps, position, velocity = stamps[valid], position[valid], velocity[valid]
    if stamps.size < 2:
        return {"samples": int(stamps.size)}
    integral = np.trapz(velocity, stamps, axis=0)
    overall = (position[-1] - position[0]) - integral
    result = {
        "samples": int(stamps.size),
        "duration_s": float(stamps[-1] - stamps[0]),
        "overall_residual_xyz_m": overall.tolist(),
        "overall_residual_norm_m": float(np.linalg.norm(overall)),
        "windows": {},
    }
    for window_s in (1.0, 5.0, 10.0):
        residuals = []
        for i, start in enumerate(stamps):
            j = int(np.searchsorted(stamps, start + window_s, side="left"))
            if j >= stamps.size:
                break
            integral = np.trapz(velocity[i : j + 1], stamps[i : j + 1], axis=0)
            residuals.append(np.linalg.norm((position[j] - position[i]) - integral))
        value = np.asarray(residuals)
        result["windows"][f"{window_s:g}s"] = {
            "count": int(value.size),
            "median_m": percentile(value, 50),
            "p95_m": percentile(value, 95),
            "max_m": percentile(value, 100),
        }
    return result


def constant_lag_metrics(
    feedback_t: np.ndarray,
    feedback_v: np.ndarray,
    truth_t: np.ndarray,
    truth_v: np.ndarray,
    sample_mask: np.ndarray,
    *,
    pair_tolerance_s: float,
    max_lag_s: float = 0.25,
    lag_step_s: float = 0.005,
) -> dict:
    """Search a constant recorder-time lag without hiding scale/frame errors.

    ``truth_query_minus_feedback_s`` is the offset added to each feedback
    timestamp before sampling truth.  A negative optimum means that feedback
    most closely resembles an earlier physical state, i.e. feedback is delayed.
    """
    best = None
    for lag_s in np.arange(-max_lag_s, max_lag_s + 0.5 * lag_step_s, lag_step_s):
        truth_at_feedback, valid = interpolate(
            truth_t, truth_v, feedback_t + lag_s
        )
        use = sample_mask & valid
        if np.sum(use) < 2:
            continue
        error = feedback_v[use] - truth_at_feedback[use]
        rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
        candidate = (rmse, float(lag_s), int(np.sum(use)))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return {"samples": 0}

    zero_truth, zero_valid = interpolate(truth_t, truth_v, feedback_t)
    zero_use = sample_mask & zero_valid
    zero_error = feedback_v[zero_use] - zero_truth[zero_use]
    zero_rmse = float(np.sqrt(np.mean(np.sum(zero_error**2, axis=1))))
    return {
        "samples": best[2],
        "zero_lag_rmse_mps": zero_rmse,
        "best_rmse_mps": best[0],
        "truth_query_minus_feedback_s": best[1],
        "estimated_feedback_delay_s": -best[1],
        "rmse_improvement_fraction": (
            float((zero_rmse - best[0]) / zero_rmse) if zero_rmse > 0.0 else 0.0
        ),
        "search_range_s": max_lag_s,
        "step_s": lag_step_s,
    }


def active_slice(
    series: list[tuple[float, object]], start: float, stop: float
) -> tuple[np.ndarray, np.ndarray]:
    stamps, values = arrays(series)
    mask = (stamps >= start) & (stamps <= stop)
    return stamps[mask], values[mask]


def phase_mask(
    stamps: np.ndarray,
    idx_t: np.ndarray,
    idx_v: np.ndarray,
    complete_t: np.ndarray,
    complete_v: np.ndarray,
    horizontal_index: int,
    settle_s: float,
) -> np.ndarray:
    idx = last_value(idx_t, idx_v, stamps, default=-1).astype(int)
    complete = last_value(complete_t, complete_v, stamps, default=False).astype(bool)
    mask = (idx == horizontal_index) & ~complete
    transitions = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    age = np.full(stamps.size, -np.inf)
    for begin, end in zip(transitions, list(transitions[1:]) + [stamps.size]):
        # Only fill while this specific horizontal run remains active.
        run_end = begin
        while run_end < stamps.size and mask[run_end]:
            run_end += 1
        age[begin:run_end] = stamps[begin:run_end] - stamps[begin]
    return mask & (age >= settle_s)


def attitude_from_observation(observation: np.ndarray) -> dict:
    if observation.size == 0 or observation.shape[1] < 4:
        return {"samples": 0}
    q = q_normalize(observation[:, :4])
    angle = np.rad2deg(2.0 * np.arccos(np.clip(np.abs(q[:, 0]), 0.0, 1.0)))
    return {
        "samples": int(angle.size),
        "mean_deg": float(np.mean(angle)),
        "p95_deg": percentile(angle, 95),
        "max_deg": percentile(angle, 100),
    }


def analyze_one(
    uri: str, *, horizontal_index: int, settle_s: float, pair_tolerance_s: float
) -> dict:
    bag = read_bag(uri)
    start, stop = longest_active_run(bag["/brov/control_active"])
    idx_t, idx_v = scalars(bag["/brov/waypoint_idx"])
    complete_t, complete_v = scalars(bag["/brov/mission_complete"])
    complete_active_t = [
        t for t, value in bag["/brov/mission_complete"] if start <= t <= stop and bool(value)
    ]
    precomplete_stop = complete_active_t[0] if complete_active_t else stop

    vd_t, vd = active_slice(bag["/brov/debug/v_desired_body_zup"], start, stop)
    selected_body_t, selected_body = active_slice(bag["/brov/debug/v_body_zup"], start, stop)
    selected_body_at_vd, selected_valid, selected_skew = nearest(
        selected_body_t, selected_body, vd_t, pair_tolerance_s
    )
    steady = phase_mask(
        vd_t, idx_t, idx_v, complete_t, complete_v, horizontal_index, settle_s
    )
    steady &= selected_valid & (np.linalg.norm(vd, axis=1) >= 0.25)

    feedback_q_t, feedback_q = active_slice(
        bag["/brov/debug/feedback_att_quat_ned"], start, stop
    )
    truth_q_t, truth_q = active_slice(
        bag["/brov/debug/gazebo_truth_att_quat_ned"], start, stop
    )
    truth_v_t, truth_v_ned = active_slice(
        bag["/brov/debug/gazebo_truth_vel_ned"], start, stop
    )
    truth_q_at_vd, truth_q_valid, truth_q_skew = nearest(
        truth_q_t, truth_q, vd_t, pair_tolerance_s
    )
    truth_v_at_vd, truth_v_valid, truth_v_skew = nearest(
        truth_v_t, truth_v_ned, vd_t, pair_tolerance_s
    )
    truth_pair = steady & truth_q_valid & truth_v_valid
    if (
        np.any(truth_pair)
        and truth_q_at_vd.shape[1:] == (4,)
        and truth_v_at_vd.shape[1:] == (3,)
    ):
        truth_body_frd = q_rotate(
            q_conj(truth_q_at_vd[truth_pair]), truth_v_at_vd[truth_pair]
        )
        truth_body_zup = truth_body_frd * np.array([1.0, -1.0, -1.0])
    else:
        truth_pair = np.zeros(vd_t.size, dtype=bool)
        truth_body_zup = np.empty((0, 3))

    feedback_pos_t, feedback_pos = active_slice(
        bag["/brov/debug/feedback_pos_ned"], start, stop
    )
    feedback_vel_t, feedback_vel = active_slice(
        bag["/brov/debug/feedback_vel_ned"], start, stop
    )
    truth_pos_t, truth_pos = active_slice(
        bag["/brov/debug/gazebo_truth_pos_ned"], start, stop
    )
    feedback_vel_at_pos, feedback_pv_valid, _ = nearest(
        feedback_vel_t, feedback_vel, feedback_pos_t, pair_tolerance_s
    )
    truth_vel_at_pos, truth_pv_valid, _ = nearest(
        truth_v_t, truth_v_ned, truth_pos_t, pair_tolerance_s
    )

    # Estimator-to-truth comparison.  Position has an arbitrary local origin,
    # therefore remove the first paired offset and report subsequent drift.
    truth_pos_at_feedback, pos_pair_valid, pos_pair_skew = nearest(
        truth_pos_t, truth_pos, feedback_pos_t, pair_tolerance_s
    )
    truth_vel_at_feedback, vel_pair_valid, vel_pair_skew = nearest(
        truth_v_t, truth_v_ned, feedback_vel_t, pair_tolerance_s
    )
    truth_q_at_feedback, q_pair_valid, q_pair_skew = nearest(
        truth_q_t, truth_q, feedback_q_t, pair_tolerance_s
    )
    if np.any(pos_pair_valid):
        residual = feedback_pos[pos_pair_valid] - truth_pos_at_feedback[pos_pair_valid]
        initial_offset = residual[0].copy()
        drift = residual - initial_offset
        drift_norm = np.linalg.norm(drift, axis=1)
    else:
        initial_offset = np.full(3, np.nan)
        drift = np.empty((0, 3))
        drift_norm = np.empty(0)
    if np.any(vel_pair_valid):
        vel_error = feedback_vel[vel_pair_valid] - truth_vel_at_feedback[vel_pair_valid]
        vel_error_norm = np.linalg.norm(vel_error, axis=1)
    else:
        vel_error = np.empty((0, 3))
        vel_error_norm = np.empty(0)
    orientation_error = (
        q_angle_deg(feedback_q[q_pair_valid], truth_q_at_feedback[q_pair_valid])
        if np.any(q_pair_valid)
        else np.empty(0)
    )

    feedback_vel_steady = phase_mask(
        feedback_vel_t,
        idx_t,
        idx_v,
        complete_t,
        complete_v,
        horizontal_index,
        settle_s,
    )
    horizontal_vel_valid = feedback_vel_steady & vel_pair_valid
    horizontal_vel_error = (
        feedback_vel[horizontal_vel_valid]
        - truth_vel_at_feedback[horizontal_vel_valid]
    )
    horizontal_vel_error_norm = (
        np.linalg.norm(horizontal_vel_error, axis=1)
        if horizontal_vel_error.size
        else np.empty(0)
    )
    feedback_pos_steady = phase_mask(
        feedback_pos_t,
        idx_t,
        idx_v,
        complete_t,
        complete_v,
        horizontal_index,
        settle_s,
    )
    horizontal_pos_valid = feedback_pos_steady & pos_pair_valid
    if np.any(horizontal_pos_valid):
        horizontal_position_residual = (
            feedback_pos[horizontal_pos_valid]
            - truth_pos_at_feedback[horizontal_pos_valid]
        )
        horizontal_position_drift = (
            horizontal_position_residual - horizontal_position_residual[0]
        )
        horizontal_position_drift_norm = np.linalg.norm(
            horizontal_position_drift, axis=1
        )
    else:
        horizontal_position_drift_norm = np.empty(0)

    feedback_precomplete = feedback_pv_valid & (feedback_pos_t <= precomplete_stop)
    truth_precomplete = truth_pv_valid & (truth_pos_t <= precomplete_stop)
    feedback_consistency_steady = feedback_pv_valid & feedback_pos_steady
    truth_pos_steady = phase_mask(
        truth_pos_t,
        idx_t,
        idx_v,
        complete_t,
        complete_v,
        horizontal_index,
        settle_s,
    )
    truth_consistency_steady = truth_pv_valid & truth_pos_steady

    obs_t, obs = active_slice(bag["/brov/observation"], start, stop)
    obs_steady = phase_mask(
        obs_t, idx_t, idx_v, complete_t, complete_v, horizontal_index, settle_s
    )

    # Reconstruct the desired-versus-physical attitude error.  The runtime's
    # start_heading frame removes initial selected-feedback yaw and then right
    # multiplies Q_M=[0,1,0,0] to change FRD body axes to FLU/Z-up axes.
    physical_attitude = {"samples": 0}
    qd_t, qd = active_slice(bag["/brov/debug/q_desired_zup"], start, stop)
    if qd.shape[0] and feedback_q.shape[0] and truth_q.shape[0]:
        qd_steady = phase_mask(
            qd_t,
            idx_t,
            idx_v,
            complete_t,
            complete_v,
            horizontal_index,
            settle_s,
        )
        truth_at_qd, qd_truth_valid, _ = nearest(
            truth_q_t, truth_q, qd_t, pair_tolerance_s
        )
        qd_mask = qd_steady & qd_truth_valid
        if np.any(qd_mask):
            yaw0 = yaw_from_q(feedback_q[0])
            count = int(np.sum(qd_mask))
            q_frame = q_mul(q_yaw(-yaw0, count), truth_at_qd[qd_mask])
            q_m = np.repeat(np.array([[0.0, 1.0, 0.0, 0.0]]), count, axis=0)
            q_truth_zup = q_mul(q_frame, q_m)
            physical_angle = q_angle_deg(qd[qd_mask], q_truth_zup)
            physical_attitude = {
                "samples": count,
                "mean_deg": float(np.mean(physical_angle)),
                "p95_deg": percentile(physical_angle, 95),
                "max_deg": percentile(physical_angle, 100),
                "note": "reconstructed with first-active selected-feedback yaw",
            }

    # Physical cross-track/depth in the mission start-heading frame.  The
    # mission origin in GT is the first active truth sample; yaw is the selected
    # feedback yaw used by the runtime's start_heading reset.
    gt_path = {}
    if truth_pos.shape[0] and feedback_q.shape[0]:
        yaw0 = yaw_from_q(feedback_q[0])
        c, s = math.cos(-yaw0), math.sin(-yaw0)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        mission_pos = (truth_pos - truth_pos[0]) @ rotation.T
        path_steady = phase_mask(
            truth_pos_t,
            idx_t,
            idx_v,
            complete_t,
            complete_v,
            horizontal_index,
            settle_s,
        )
        path = mission_pos[path_steady]
        if path.size:
            gt_path = {
                "samples": int(path.shape[0]),
                "along_track_start_end_m": [float(path[0, 0]), float(path[-1, 0])],
                "cross_track_rms_m": float(np.sqrt(np.mean(path[:, 1] ** 2))),
                "cross_track_max_abs_m": float(np.max(np.abs(path[:, 1]))),
                # Stage-1 mission has depth +0.20 m in NED/start_heading.
                "depth_error_rms_m": float(np.sqrt(np.mean((path[:, 2] - 0.20) ** 2))),
                "depth_error_max_abs_m": float(np.max(np.abs(path[:, 2] - 0.20))),
            }

    action_t, action = active_slice(bag["/brov/action"], start, stop)
    action_steady = phase_mask(
        action_t, idx_t, idx_v, complete_t, complete_v, horizontal_index, settle_s
    )
    action_s = action[action_steady]
    cap = np.abs(action_s) >= 0.99 if action_s.size else np.empty((0, 6), dtype=bool)

    raw_t, raw = active_slice(bag["/brov/policy/action_raw"], start, stop)
    raw_steady = phase_mask(
        raw_t, idx_t, idx_v, complete_t, complete_v, horizontal_index, settle_s
    )
    raw_s = raw[raw_steady]

    requested_t, requested = active_slice(
        bag["/brov/policy/thruster_force_requested"], start, stop
    )
    limited_t, limited = active_slice(
        bag["/brov/policy/thruster_force_limited"], start, stop
    )
    limited_at_requested, force_valid, _ = nearest(
        limited_t, limited, requested_t, pair_tolerance_s
    )
    force_steady = phase_mask(
        requested_t, idx_t, idx_v, complete_t, complete_v, horizontal_index, settle_s
    ) & force_valid
    force_delta = requested[force_steady] - limited_at_requested[force_steady]

    timing_t, timing = active_slice(bag["/brov/debug/feedback_timing"], start, stop)
    timing_valid = timing.shape[1] == 10 if timing.ndim == 2 and timing.size else False
    source_values = [str(value) for _, value in bag["/brov/debug/feedback_source"]]

    index_active = last_value(idx_t, idx_v, vd_t, default=-1).astype(int)
    sequence = []
    for value in index_active:
        if not sequence or sequence[-1] != int(value):
            sequence.append(int(value))
    return {
        "bag": str(Path(uri).resolve()),
        "feedback_source": source_values[-1] if source_values else "missing",
        "active": {
            "start_unix_s": start,
            "stop_unix_s": stop,
            "duration_s": stop - start,
            "waypoint_sequence": sequence,
            "mission_complete_observed": bool(complete_active_t),
            "mission_complete_after_start_s": (
                float(complete_active_t[0] - start) if complete_active_t else None
            ),
        },
        "horizontal_steady": {
            "definition": (
                f"waypoint_idx={horizontal_index}, mission_complete=false, "
                f"first {settle_s:g}s excluded, ||v_desired||>=0.25 m/s"
            ),
            "controller_visible": tracking_metrics(vd[steady], selected_body_at_vd[steady]),
            "gazebo_ground_truth": tracking_metrics(vd[truth_pair], truth_body_zup),
            "controller_attitude_error": attitude_from_observation(obs[obs_steady]),
            "gazebo_ground_truth_attitude_error": physical_attitude,
            "gazebo_path": gt_path,
        },
        "feedback_vs_gazebo_truth": {
            "position_initial_offset_ned_m": initial_offset.tolist(),
            "position_drift_rms_m": (
                float(np.sqrt(np.mean(drift_norm**2))) if drift_norm.size else math.nan
            ),
            "position_drift_p95_m": percentile(drift_norm, 95),
            "position_drift_end_m": (
                float(drift_norm[-1]) if drift_norm.size else math.nan
            ),
            "velocity_vector_rmse_mps": (
                float(np.sqrt(np.mean(vel_error_norm**2)))
                if vel_error_norm.size
                else math.nan
            ),
            "velocity_error_p95_mps": percentile(vel_error_norm, 95),
            "velocity_component_bias_ned_mps": (
                np.mean(vel_error, axis=0).tolist() if vel_error.size else [math.nan] * 3
            ),
            "attitude_error_median_deg": percentile(orientation_error, 50),
            "attitude_error_p95_deg": percentile(orientation_error, 95),
            "attitude_error_max_deg": percentile(orientation_error, 100),
            "pairing_recorder_skew_p95_ms": {
                "position": 1.0e3 * percentile(np.abs(pos_pair_skew[pos_pair_valid]), 95),
                "velocity": 1.0e3 * percentile(np.abs(vel_pair_skew[vel_pair_valid]), 95),
                "attitude": 1.0e3 * percentile(np.abs(q_pair_skew[q_pair_valid]), 95),
            },
            "horizontal_steady": {
                "velocity_vector_rmse_mps": (
                    float(np.sqrt(np.mean(horizontal_vel_error_norm**2)))
                    if horizontal_vel_error_norm.size
                    else math.nan
                ),
                "velocity_error_p95_mps": percentile(
                    horizontal_vel_error_norm, 95
                ),
                "velocity_component_bias_ned_mps": (
                    np.mean(horizontal_vel_error, axis=0).tolist()
                    if horizontal_vel_error.size
                    else [math.nan] * 3
                ),
                "position_drift_rms_m": (
                    float(np.sqrt(np.mean(horizontal_position_drift_norm**2)))
                    if horizontal_position_drift_norm.size
                    else math.nan
                ),
                "position_drift_p95_m": percentile(
                    horizontal_position_drift_norm, 95
                ),
                "position_drift_end_m": (
                    float(horizontal_position_drift_norm[-1])
                    if horizontal_position_drift_norm.size
                    else math.nan
                ),
                "constant_lag_search": constant_lag_metrics(
                    feedback_vel_t,
                    feedback_vel,
                    truth_v_t,
                    truth_v_ned,
                    feedback_vel_steady,
                    pair_tolerance_s=pair_tolerance_s,
                ),
            },
        },
        "position_velocity_consistency": {
            "selected_feedback": consistency_metrics(
                feedback_pos_t[feedback_pv_valid],
                feedback_pos[feedback_pv_valid],
                feedback_vel_at_pos[feedback_pv_valid],
            ),
            "gazebo_truth": consistency_metrics(
                truth_pos_t[truth_pv_valid],
                truth_pos[truth_pv_valid],
                truth_vel_at_pos[truth_pv_valid],
            ),
            "pre_mission_complete": {
                "selected_feedback": consistency_metrics(
                    feedback_pos_t[feedback_precomplete],
                    feedback_pos[feedback_precomplete],
                    feedback_vel_at_pos[feedback_precomplete],
                ),
                "gazebo_truth": consistency_metrics(
                    truth_pos_t[truth_precomplete],
                    truth_pos[truth_precomplete],
                    truth_vel_at_pos[truth_precomplete],
                ),
            },
            "horizontal_steady": {
                "selected_feedback": consistency_metrics(
                    feedback_pos_t[feedback_consistency_steady],
                    feedback_pos[feedback_consistency_steady],
                    feedback_vel_at_pos[feedback_consistency_steady],
                ),
                "gazebo_truth": consistency_metrics(
                    truth_pos_t[truth_consistency_steady],
                    truth_pos[truth_consistency_steady],
                    truth_vel_at_pos[truth_consistency_steady],
                ),
            },
        },
        "action_and_actuation": {
            "steady_samples": int(action_s.shape[0]),
            "limited_any_axis_ge_0p99_fraction": (
                float(np.mean(np.any(cap, axis=1))) if cap.size else math.nan
            ),
            "limited_axis_ge_0p99_fraction": (
                dict(zip(AXES, np.mean(cap, axis=0).tolist()))
                if cap.size
                else {axis: math.nan for axis in AXES}
            ),
            "raw_any_axis_outside_unit_fraction": (
                float(np.mean(np.any(np.abs(raw_s) > 1.0, axis=1)))
                if raw_s.size
                else math.nan
            ),
            "raw_axis_max_abs": (
                dict(zip(AXES, np.max(np.abs(raw_s), axis=0).tolist()))
                if raw_s.size
                else {axis: math.nan for axis in AXES}
            ),
            "thruster_force_any_clamp_fraction": (
                float(np.mean(np.any(np.abs(force_delta) > 1.0e-5, axis=1)))
                if force_delta.size
                else math.nan
            ),
            "thruster_force_clamp_error_rms_N": (
                float(np.sqrt(np.mean(force_delta**2)))
                if force_delta.size
                else math.nan
            ),
        },
        "timing": {
            "desired_velocity_topic": rate_metrics(vd_t),
            "action_topic": rate_metrics(action_t),
            "selected_body_pair_skew_p95_ms": 1.0e3
            * percentile(np.abs(selected_skew[selected_valid]), 95),
            "truth_velocity_pair_skew_p95_ms": 1.0e3
            * percentile(np.abs(truth_v_skew[truth_v_valid]), 95),
            "truth_attitude_pair_skew_p95_ms": 1.0e3
            * percentile(np.abs(truth_q_skew[truth_q_valid]), 95),
            "feedback_timing_samples": int(timing_t.size),
            "feedback_source_codes": (
                sorted(set(timing[:, 0].astype(int).tolist())) if timing_valid else []
            ),
            "selected_age_p95_ms": (
                1.0e3 * percentile(timing[:, 3], 95) if timing_valid else math.nan
            ),
            "selected_age_max_ms": (
                1.0e3 * percentile(timing[:, 3], 100) if timing_valid else math.nan
            ),
            "mav_att_age_p95_ms": (
                1.0e3 * percentile(timing[:, 6], 95) if timing_valid else math.nan
            ),
            "mav_pos_age_p95_ms": (
                1.0e3 * percentile(timing[:, 7], 95) if timing_valid else math.nan
            ),
            "gazebo_age_p95_ms": (
                1.0e3 * percentile(timing[:, 9], 95) if timing_valid else math.nan
            ),
        },
        "topic_contract": {
            "missing_required": sorted(
                topic
                for topic in (
                    "/brov/control_active",
                    "/brov/waypoint_idx",
                    "/brov/observation",
                    "/brov/action",
                    "/brov/debug/feedback_source",
                    "/brov/debug/feedback_timing",
                    "/brov/debug/feedback_pos_ned",
                    "/brov/debug/feedback_vel_ned",
                    "/brov/debug/feedback_att_quat_ned",
                    "/brov/debug/gazebo_truth_pos_ned",
                    "/brov/debug/gazebo_truth_vel_ned",
                    "/brov/debug/gazebo_truth_att_quat_ned",
                    "/brov/debug/v_body_zup",
                    "/brov/debug/v_desired_body_zup",
                    "/brov/policy/action_raw",
                    "/brov/policy/thruster_force_requested",
                    "/brov/policy/thruster_force_limited",
                )
                if topic not in bag["_present"]
            )
        },
    }


def ratio(a: float, b: float) -> float:
    return float(a / b) if math.isfinite(a) and math.isfinite(b) and b != 0.0 else math.nan


def comparison(first: dict, second: dict) -> dict:
    a_gt = first["horizontal_steady"]["gazebo_ground_truth"]
    b_gt = second["horizontal_steady"]["gazebo_ground_truth"]
    a_est = first["feedback_vs_gazebo_truth"]
    b_est = second["feedback_vs_gazebo_truth"]
    keys = ("v_parallel_mean_mps", "vector_error_rmse_mps", "cross_speed_rms_mps")
    return {
        "order": [first["feedback_source"], second["feedback_source"]],
        "physical_gt_delta_second_minus_first": {
            key: float(b_gt.get(key, math.nan) - a_gt.get(key, math.nan)) for key in keys
        },
        "feedback_truth_velocity_rmse_ratio_second_over_first": ratio(
            float(b_est["velocity_vector_rmse_mps"]),
            float(a_est["velocity_vector_rmse_mps"]),
        ),
        "feedback_truth_position_drift_rms_ratio_second_over_first": ratio(
            float(b_est["position_drift_rms_m"]),
            float(a_est["position_drift_rms_m"]),
        ),
        "interpretation_rule": (
            "If gazebo_truth feedback tracks physically but mavlink_ekf feedback does not, "
            "the feedback/estimator path is causal. If both physical GT runs fail similarly, "
            "inspect policy/action/actuator/plant/guidance before retraining. If G0/G1 are "
            "not separately exercised, this A/B does not isolate actuator from plant."
        ),
    }


def json_safe(value):
    """Replace non-finite metrics with JSON null for strict downstream tools."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bags", nargs="+", help="one or two rosbag2 directory paths")
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument("--horizontal-index", type=int, default=1)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--pair-tolerance-s", type=float, default=0.03)
    args = parser.parse_args()
    if len(args.bags) > 2:
        parser.error("provide one bag for inspection or exactly two for A/B")

    runs = [
        analyze_one(
            uri,
            horizontal_index=args.horizontal_index,
            settle_s=args.settle_s,
            pair_tolerance_s=args.pair_tolerance_s,
        )
        for uri in args.bags
    ]
    result = {"schema": "brov_sim2sim_0p5_ab_analysis_v1", "runs": runs}
    if len(runs) == 2:
        result["comparison"] = comparison(runs[0], runs[1])
    encoded = json.dumps(json_safe(result), indent=2, allow_nan=False)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

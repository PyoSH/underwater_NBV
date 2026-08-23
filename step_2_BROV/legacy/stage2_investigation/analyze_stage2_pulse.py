#!/usr/bin/env python3
"""Analyze the synchronized Stage-2 Gazebo/EKF open-loop pulse rosbag."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pymavlink import mavutil
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Int32MultiArray, String


ARRAY32 = {
    "/brov/debug/feedback_pos_ned",
    "/brov/debug/feedback_vel_ned",
    "/brov/debug/feedback_att_quat_ned",
    "/brov/debug/gazebo_truth_pos_ned",
    "/brov/debug/gazebo_truth_vel_ned",
    "/brov/debug/gazebo_truth_att_quat_ned",
}
ARRAY64 = {
    "/brov/debug/feedback_timing",
    "/brov/stage2/dvl_sample",
    "/brov/stage2/pulse_pwm",
    "/brov/stage2/mavlink_snapshot",
}
INT_ARRAY = {"/brov/debug/servo_output_us"}
STRING = {
    "/brov/debug/feedback_source",
    "/brov/stage2/dvl_status",
    "/brov/stage2/phase",
}
TOPICS = ARRAY32 | ARRAY64 | INT_ARRAY | STRING
PHASE_NAMES = {1: "surge_pos", 2: "surge_neg", 3: "sway_pos", 4: "sway_neg"}


def read_bag(uri: str) -> dict[str, list[tuple[float, object]]]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=uri, storage_id="sqlite3"),
        ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    out = {name: [] for name in TOPICS}
    present = {item.name: item.type for item in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic not in TOPICS:
            continue
        if topic in ARRAY32:
            value = np.asarray(
                deserialize_message(payload, Float32MultiArray).data, dtype=float
            )
        elif topic in ARRAY64:
            value = np.asarray(
                deserialize_message(payload, Float64MultiArray).data, dtype=float
            )
        elif topic in INT_ARRAY:
            value = np.asarray(
                deserialize_message(payload, Int32MultiArray).data, dtype=float
            )
        else:
            value = str(deserialize_message(payload, String).data)
        out[topic].append((stamp_ns * 1.0e-9, value))
    out["_present"] = present
    return out


def arrays(series: list[tuple[float, object]]) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.empty(0), np.empty((0, 0))
    return np.asarray([x[0] for x in series]), np.stack([x[1] for x in series])


def interp(
    source_t: np.ndarray,
    source_v: np.ndarray,
    query_t: np.ndarray,
    max_gap_s: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    if source_t.size < 2:
        width = source_v.shape[1] if source_v.ndim == 2 else 1
        return np.full((query_t.size, width), np.nan), np.zeros(query_t.size, bool)
    right = np.searchsorted(source_t, query_t, side="right")
    left = right - 1
    valid = (left >= 0) & (right < source_t.size)
    left = np.clip(left, 0, source_t.size - 1)
    right = np.clip(right, 0, source_t.size - 1)
    dt = source_t[right] - source_t[left]
    valid &= (dt > 0.0) & (dt <= max_gap_s)
    alpha = np.zeros(query_t.size)
    alpha[valid] = (query_t[valid] - source_t[left[valid]]) / dt[valid]
    result = source_v[left] + alpha[:, None] * (source_v[right] - source_v[left])
    result[~valid] = np.nan
    return result, valid


def rate(stamps: np.ndarray) -> dict:
    if stamps.size < 2:
        return {"count": int(stamps.size), "hz": math.nan}
    gap = np.diff(stamps)
    return {
        "count": int(stamps.size),
        "hz": float((stamps.size - 1) / (stamps[-1] - stamps[0])),
        "gap_median_ms": float(1e3 * np.median(gap)),
        "gap_p95_ms": float(1e3 * np.percentile(gap, 95)),
        "gap_max_ms": float(1e3 * np.max(gap)),
    }


def lag_fit(
    feedback_t: np.ndarray,
    feedback_v: np.ndarray,
    truth_t: np.ndarray,
    truth_v: np.ndarray,
    mask: np.ndarray,
) -> dict:
    candidates = np.arange(-0.250, 0.2501, 0.001)
    best = None
    zero_rmse = math.nan
    for lag in candidates:
        truth, valid = interp(truth_t, truth_v, feedback_t + lag)
        use = valid & mask & np.all(np.isfinite(feedback_v), axis=1)
        if np.count_nonzero(use) < 20:
            continue
        error = feedback_v[use] - truth[use]
        rmse = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
        if abs(lag) < 0.0005:
            zero_rmse = rmse
        if best is None or rmse < best[0]:
            best = (rmse, float(lag), truth, use)
    if best is None:
        return {"rmse_zero_lag_mps": math.nan, "best": None}
    rmse, lag, truth, use = best
    error = feedback_v[use] - truth[use]
    return {
        "rmse_zero_lag_mps": zero_rmse,
        "best_truth_query_minus_feedback_s": lag,
        "estimated_feedback_delay_s": -lag,
        "best_rmse_mps": rmse,
        "p95_error_mps": float(np.percentile(np.linalg.norm(error, axis=1), 95)),
        "bias_ned_mps": np.mean(error, axis=0).tolist(),
        "truth_at_best": truth,
        "use_at_best": use,
    }


def closure(
    pos_t: np.ndarray,
    pos: np.ndarray,
    vel_t: np.ndarray,
    vel: np.ndarray,
    start: float,
    stop: float,
) -> dict:
    grid = pos_t[(pos_t >= start) & (pos_t <= stop)]
    if grid.size < 3:
        return {"duration_s": math.nan, "residual_ned_m": [math.nan] * 3}
    velocity, valid = interp(vel_t, vel, grid)
    grid = grid[valid]
    position = pos[(pos_t >= start) & (pos_t <= stop)][valid]
    velocity = velocity[valid]
    if grid.size < 3:
        return {"duration_s": math.nan, "residual_ned_m": [math.nan] * 3}
    integral = np.trapz(velocity, grid, axis=0)
    displacement = position[-1] - position[0]
    residual = displacement - integral
    return {
        "duration_s": float(grid[-1] - grid[0]),
        "displacement_ned_m": displacement.tolist(),
        "integrated_velocity_ned_m": integral.tolist(),
        "residual_ned_m": residual.tolist(),
        "residual_norm_m": float(np.linalg.norm(residual)),
        "horizontal_residual_m": float(np.linalg.norm(residual[:2])),
    }


def rolling_closure(
    pos_t: np.ndarray,
    pos: np.ndarray,
    vel_t: np.ndarray,
    vel: np.ndarray,
    start: float,
    stop: float,
    window_s: float = 10.0,
    stride_s: float = 0.5,
) -> dict:
    starts = np.arange(start, stop - window_s + 1.0e-9, stride_s)
    values = []
    for window_start in starts:
        item = closure(
            pos_t,
            pos,
            vel_t,
            vel,
            float(window_start),
            float(window_start + window_s),
        )
        if math.isfinite(item.get("horizontal_residual_m", math.nan)):
            values.append(item)
    if not values:
        return {"window_s": window_s, "count": 0}
    horizontal = np.asarray([x["horizontal_residual_m"] for x in values])
    norm = np.asarray([x["residual_norm_m"] for x in values])
    return {
        "window_s": window_s,
        "stride_s": stride_s,
        "count": len(values),
        "horizontal_residual_median_m": float(np.median(horizontal)),
        "horizontal_residual_p95_m": float(np.percentile(horizontal, 95)),
        "horizontal_residual_max_m": float(np.max(horizontal)),
        "residual_norm_median_m": float(np.median(norm)),
        "residual_norm_p95_m": float(np.percentile(norm, 95)),
        "residual_norm_max_m": float(np.max(norm)),
    }


def quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate batched vectors with body->world wxyz quaternions."""

    q = np.asarray(quaternion, dtype=float)
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / norm
    qv = q[:, 1:]
    twice_cross = 2.0 * np.cross(qv, v)
    return v + q[:, :1] * twice_cross + np.cross(qv, twice_cross)


def quat_rotate_inverse_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    inverse = np.asarray(quaternion, dtype=float).copy()
    inverse[:, 1:] *= -1.0
    return quat_rotate_wxyz(inverse, vector)


def vector_error(reference: np.ndarray, estimate: np.ndarray) -> dict:
    error = estimate - reference
    norm = np.linalg.norm(error, axis=1)
    return {
        "samples": int(error.shape[0]),
        "bias": np.mean(error, axis=0).tolist(),
        "axis_rmse": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "vector_rmse": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
        "p95_norm": float(np.percentile(norm, 95)),
        "max_norm": float(np.max(norm)),
    }


def dataflash_metrics(
    path: str, boot_start_s: float, boot_stop_s: float
) -> dict:
    log = mavutil.mavlink_connection(path)
    rows: dict[str, list[dict]] = {
        "SIM2": [],
        "VISO": [],
        "XKF1": [],
        "XKFD": [],
    }
    counts: dict[str, int] = {}
    while True:
        message = log.recv_match()
        if message is None:
            break
        kind = message.get_type()
        counts[kind] = counts.get(kind, 0) + 1
        if kind not in rows:
            continue
        value = message.to_dict()
        stamp = float(value["TimeUS"]) * 1.0e-6
        if boot_start_s - 0.5 <= stamp <= boot_stop_s + 0.5:
            if kind in {"XKF1", "XKFD"} and int(value["C"]) != 0:
                continue
            rows[kind].append(value)

    def field(kind: str, name: str) -> np.ndarray:
        return np.asarray([float(x[name]) for x in rows[kind]], dtype=float)

    result = {
        "path": path,
        "pulse_boot_window_s": [boot_start_s, boot_stop_s],
        "message_counts_whole_log": {
            name: int(counts.get(name, 0))
            for name in ("GPS", "GPA", "SIM2", "VISO", "XKF1", "XKFD")
        },
    }
    viso_t = field("VISO", "TimeUS") * 1.0e-6
    if viso_t.size:
        viso_dt = field("VISO", "dt")
        viso_delta = np.column_stack(
            [field("VISO", key) for key in ("PosDX", "PosDY", "PosDZ")]
        )
        result["viso"] = {
            "rate": rate(viso_t),
            "dt_median_s": float(np.median(viso_dt)),
            "dt_values_s": sorted(np.unique(np.round(viso_dt, 6)).tolist()),
            "confidence_min": float(np.min(field("VISO", "conf"))),
            "confidence_max": float(np.max(field("VISO", "conf"))),
            "finite_delta_fraction": float(np.mean(np.isfinite(viso_delta).all(axis=1))),
        }

    sim_t = field("SIM2", "TimeUS") * 1.0e-6
    sim_p = (
        np.column_stack([field("SIM2", key) for key in ("PN", "PE", "PD")])
        if sim_t.size
        else np.empty((0, 3))
    )
    sim_v = (
        np.column_stack([field("SIM2", key) for key in ("VN", "VE", "VD")])
        if sim_t.size
        else np.empty((0, 3))
    )
    xkf_t = field("XKF1", "TimeUS") * 1.0e-6
    xkf_p = (
        np.column_stack([field("XKF1", key) for key in ("PN", "PE", "PD")])
        if xkf_t.size
        else np.empty((0, 3))
    )
    xkf_v = (
        np.column_stack([field("XKF1", key) for key in ("VN", "VE", "VD")])
        if xkf_t.size
        else np.empty((0, 3))
    )
    if sim_t.size and xkf_t.size:
        truth, valid = interp(sim_t, sim_v, xkf_t, max_gap_s=0.02)
        use = valid & (xkf_t >= boot_start_s) & (xkf_t <= boot_stop_s)
        result["xkf1_velocity_vs_sim2"] = vector_error(truth[use], xkf_v[use])
        result["closure"] = {
            "sim2": closure(
                sim_t, sim_p, sim_t, sim_v, boot_start_s, boot_stop_s
            ),
            "xkf1": closure(
                xkf_t, xkf_p, xkf_t, xkf_v, boot_start_s, boot_stop_s
            ),
            "sim2_rolling_10s": rolling_closure(
                sim_t, sim_p, sim_t, sim_v, boot_start_s, boot_stop_s
            ),
            "xkf1_rolling_10s": rolling_closure(
                xkf_t, xkf_p, xkf_t, xkf_v, boot_start_s, boot_stop_s
            ),
        }
    innov_t = field("XKFD", "TimeUS") * 1.0e-6
    if innov_t.size:
        select = (innov_t >= boot_start_s) & (innov_t <= boot_stop_s)
        innovation = np.column_stack(
            [field("XKFD", key) for key in ("IX", "IY", "IZ")]
        )[select]
        innovation_variance = np.column_stack(
            [field("XKFD", key) for key in ("IVX", "IVY", "IVZ")]
        )[select]
        result["body_odom_innovation"] = {
            "count": int(innovation.shape[0]),
            "axis_mean_mps": np.mean(innovation, axis=0).tolist(),
            "axis_rms_mps": np.sqrt(np.mean(innovation * innovation, axis=0)).tolist(),
            "axis_p95_abs_mps": np.percentile(
                np.abs(innovation), 95, axis=0
            ).tolist(),
            "innovation_variance_mean": np.mean(
                innovation_variance, axis=0
            ).tolist(),
        }
    return result


def held_command(
    command_t: np.ndarray, command_pwm: np.ndarray, query_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    index = np.searchsorted(command_t, query_t, side="right") - 1
    valid = index >= 0
    index = np.clip(index, 0, command_t.size - 1)
    result = command_pwm[index].copy()
    result[~valid] = np.nan
    return result, valid


def servo_metrics(
    command_t: np.ndarray,
    command_pwm: np.ndarray,
    servo_t: np.ndarray,
    servo_us: np.ndarray,
) -> dict:
    best = None
    for lag in np.arange(0.0, 0.1001, 0.001):
        expected_pwm, valid = held_command(command_t, command_pwm, servo_t - lag)
        use = valid & (servo_t >= command_t[0]) & (servo_t <= command_t[-1] + 0.2)
        if np.count_nonzero(use) < 20:
            continue
        expected_us = 1500.0 + 400.0 * expected_pwm[use]
        error = servo_us[use] - expected_us
        mae = float(np.mean(np.abs(error)))
        if best is None or mae < best[0]:
            best = (mae, lag, error)
    if best is None:
        return {"best_lag_s": math.nan, "mae_us": math.nan}
    mae, lag, error = best
    end_mask = servo_t >= command_t[-1]
    final_neutral = (
        bool(np.all(np.abs(servo_us[end_mask][-5:] - 1500.0) <= 2.0))
        if np.count_nonzero(end_mask) >= 5
        else False
    )
    return {
        "best_lag_s": float(lag),
        "mae_us": mae,
        "p95_abs_error_us": float(np.percentile(np.abs(error), 95)),
        "max_abs_error_us": float(np.max(np.abs(error))),
        "final_five_samples_neutral": final_neutral,
    }


def analyze(uri: str, dataflash: str | None = None) -> dict:
    bag = read_bag(uri)
    cmd_t, cmd = arrays(bag["/brov/stage2/pulse_pwm"])
    if cmd_t.size == 0 or cmd.shape[1] != 10:
        raise RuntimeError("missing or malformed /brov/stage2/pulse_pwm")
    phase_code = cmd[:, 0].astype(int)
    command_pwm = cmd[:, 2:]
    start, stop = float(cmd_t[0]), float(cmd_t[-1])

    fb_pt, fb_p = arrays(bag["/brov/debug/feedback_pos_ned"])
    fb_vt, fb_v = arrays(bag["/brov/debug/feedback_vel_ned"])
    gt_pt, gt_p = arrays(bag["/brov/debug/gazebo_truth_pos_ned"])
    gt_vt, gt_v = arrays(bag["/brov/debug/gazebo_truth_vel_ned"])
    gt_qt, gt_q = arrays(bag["/brov/debug/gazebo_truth_att_quat_ned"])
    dvl_t, dvl = arrays(bag["/brov/stage2/dvl_sample"])
    snapshot_t, snapshot = arrays(bag["/brov/stage2/mavlink_snapshot"])
    servo_t, servo = arrays(bag["/brov/debug/servo_output_us"])
    timing_t, timing = arrays(bag["/brov/debug/feedback_timing"])

    in_window = (fb_vt >= start) & (fb_vt <= stop)
    lag = lag_fit(fb_vt, fb_v, gt_vt, gt_v, in_window)
    truth_at_best = lag.pop("truth_at_best", None)
    use_at_best = lag.pop("use_at_best", None)

    matrix = None
    if truth_at_best is not None and use_at_best is not None:
        dynamic = use_at_best & (np.linalg.norm(truth_at_best[:, :2], axis=1) >= 0.03)
        if np.count_nonzero(dynamic) >= 20:
            x = np.column_stack(
                (truth_at_best[dynamic, :2], np.ones(np.count_nonzero(dynamic)))
            )
            y = fb_v[dynamic, :2]
            coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
            prediction = x @ coefficients
            residual = y - prediction
            a = coefficients[:2].T
            matrix = {
                "A_feedback_from_truth_xy": a.tolist(),
                "bias_xy_mps": coefficients[2].tolist(),
                "determinant": float(np.linalg.det(a)),
                "singular_values": np.linalg.svd(a, compute_uv=False).tolist(),
                "fit_rmse_mps": float(
                    np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
                ),
                "samples": int(np.count_nonzero(dynamic)),
            }

    first_pulse = float(cmd_t[np.flatnonzero(phase_code != 0)[0]])
    fb_initial, fb_initial_valid = interp(fb_vt, fb_v, cmd_t)
    gt_initial, gt_initial_valid = interp(gt_vt, gt_v, cmd_t)
    neutral = (
        (phase_code == 0)
        & (cmd_t < first_pulse)
        & fb_initial_valid
        & gt_initial_valid
    )
    stationary_error = fb_initial[neutral] - gt_initial[neutral]

    phases = {}
    for code, name in PHASE_NAMES.items():
        select = phase_code == code
        query = cmd_t[select]
        feedback, fv = interp(fb_vt, fb_v, query)
        truth, tv = interp(gt_vt, gt_v, query)
        attitude, qv = interp(gt_qt, gt_q, query)
        valid = fv & tv & qv
        error = feedback[valid] - truth[valid]
        truth_body = quat_rotate_inverse_wxyz(attitude[valid], truth[valid])
        phases[name] = {
            "duration_s": float(query[-1] - query[0]) if query.size else math.nan,
            "samples": int(np.count_nonzero(valid)),
            "command_pwm": command_pwm[select][0].tolist() if query.size else [],
            "truth_velocity_mean_ned_mps": (
                np.mean(truth[valid], axis=0).tolist() if np.any(valid) else []
            ),
            "truth_velocity_mean_body_frd_mps": (
                np.mean(truth_body, axis=0).tolist() if truth_body.size else []
            ),
            "feedback_velocity_mean_ned_mps": (
                np.mean(feedback[valid], axis=0).tolist() if np.any(valid) else []
            ),
            "velocity_error_bias_ned_mps": (
                np.mean(error, axis=0).tolist() if error.size else []
            ),
            "velocity_error_rmse_mps": (
                float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
                if error.size
                else math.nan
            ),
        }

    dvl_metrics = {"samples": 0}
    if dvl.size and dvl.shape[1] >= 21:
        dvl_window = (dvl_t >= start) & (dvl_t <= stop)
        d = dvl[dvl_window]
        dt = d[:, 3]
        realized_delay = d[:, 2] - d[:, 1]
        noise = d[:, 15:18]
        dvl_metrics = {
            "samples": int(d.shape[0]),
            "recorder_rate": rate(dvl_t[dvl_window]),
            "source_rate": rate(d[:, 1]),
            "source_dt_median_s": float(np.median(dt)),
            "source_dt_values_s": sorted(np.unique(np.round(dt, 6)).tolist()),
            "realized_source_delay_mean_s": float(np.mean(realized_delay)),
            "realized_source_delay_p95_s": float(
                np.percentile(realized_delay, 95)
            ),
            "world_z_min_m": float(np.min(d[:, 5])),
            "world_z_max_m": float(np.max(d[:, 5])),
            "noise_bias_body_frd_mps": np.mean(noise, axis=0).tolist(),
            "noise_std_body_frd_mps": np.std(noise, axis=0).tolist(),
            "noise_vector_rms_mps": float(
                np.sqrt(np.mean(np.sum(noise * noise, axis=1)))
            ),
            "confidence_min": float(np.min(d[:, 19])),
            "confidence_max": float(np.max(d[:, 19])),
        }
        source_query_t = dvl_t[dvl_window] - realized_delay
        source_attitude, av = interp(gt_qt, gt_q, source_query_t)
        source_truth, vv = interp(gt_vt, gt_v, source_query_t)
        source_position, pv = interp(gt_pt, gt_p, source_query_t)
        previous_position, ppv = interp(gt_pt, gt_p, source_query_t - dt)
        use = av & vv
        if np.count_nonzero(use):
            true_ned = quat_rotate_wxyz(source_attitude[use], d[use, 6:9])
            measured_ned = quat_rotate_wxyz(source_attitude[use], d[use, 9:12])
            dvl_metrics["stored_true_body_vs_ros_truth"] = vector_error(
                source_truth[use], true_ned
            )
            dvl_metrics["measured_body_vs_ros_truth"] = vector_error(
                source_truth[use], measured_ned
            )
        # VISION_POSITION_DELTA is fused as a body-frame displacement over
        # ``dt``.  Compare that exact injected quantity, after applying the
        # current body->NED attitude, with the independent Gazebo displacement
        # over the same source-time interval.  This separates an input/delta
        # construction error from an error introduced later by EKF fusion.
        delta_use = av & pv & ppv
        if np.count_nonzero(delta_use):
            injected_delta_ned = quat_rotate_wxyz(
                source_attitude[delta_use], d[delta_use, 12:15]
            )
            truth_delta_ned = (
                source_position[delta_use] - previous_position[delta_use]
            )
            delta_error = injected_delta_ned - truth_delta_ned
            cumulative_error = np.sum(delta_error, axis=0)
            dvl_metrics["injected_delta_vs_truth_displacement"] = {
                "samples": int(np.count_nonzero(delta_use)),
                "interval_error_bias_ned_m": np.mean(
                    delta_error, axis=0
                ).tolist(),
                "interval_error_vector_rmse_m": float(
                    np.sqrt(np.mean(np.sum(delta_error * delta_error, axis=1)))
                ),
                "interval_error_p95_m": float(
                    np.percentile(np.linalg.norm(delta_error, axis=1), 95)
                ),
                "cumulative_error_ned_m": cumulative_error.tolist(),
                "cumulative_error_horizontal_m": float(
                    np.linalg.norm(cumulative_error[:2])
                ),
                "cumulative_error_norm_m": float(np.linalg.norm(cumulative_error)),
            }

    source_values = [value for _, value in bag["/brov/debug/feedback_source"]]
    timing_window = (timing_t >= start) & (timing_t <= stop)
    selected_age = timing[timing_window, 3] if timing.size else np.empty(0)
    truth_age = timing[timing_window, 9] if timing.size else np.empty(0)
    result = {
        "bag": uri,
        "window": {"start_s": start, "stop_s": stop, "duration_s": stop - start},
        "feedback_source": source_values[-1] if source_values else "missing",
        "rates": {
            "pulse_command": rate(cmd_t),
            "feedback_velocity": rate(fb_vt[(fb_vt >= start) & (fb_vt <= stop)]),
            "truth_velocity": rate(gt_vt[(gt_vt >= start) & (gt_vt <= stop)]),
            "servo": rate(servo_t[(servo_t >= start) & (servo_t <= stop)]),
        },
        "dvl_injection": dvl_metrics,
        "initial_neutral": {
            "samples": int(np.count_nonzero(neutral)),
            "feedback_velocity_mean_ned_mps": (
                np.mean(fb_initial[neutral], axis=0).tolist() if np.any(neutral) else []
            ),
            "truth_velocity_mean_ned_mps": (
                np.mean(gt_initial[neutral], axis=0).tolist() if np.any(neutral) else []
            ),
            "feedback_minus_truth_bias_ned_mps": (
                np.mean(stationary_error, axis=0).tolist()
                if stationary_error.size
                else []
            ),
        },
        "velocity_feedback_vs_truth": lag,
        "frame_scale_fit": matrix,
        "phases": phases,
        "kinematic_closure": {
            "feedback": closure(fb_pt, fb_p, fb_vt, fb_v, start, stop),
            "gazebo_truth": closure(gt_pt, gt_p, gt_vt, gt_v, start, stop),
            "feedback_rolling_10s": rolling_closure(
                fb_pt, fb_p, fb_vt, fb_v, start, stop
            ),
            "gazebo_truth_rolling_10s": rolling_closure(
                gt_pt, gt_p, gt_vt, gt_v, start, stop
            ),
        },
        "servo_transport": servo_metrics(cmd_t, command_pwm, servo_t, servo),
        "timing": {
            "selected_age_p95_s": (
                float(np.percentile(selected_age, 95)) if selected_age.size else math.nan
            ),
            "truth_age_p95_s": (
                float(np.percentile(truth_age, 95)) if truth_age.size else math.nan
            ),
        },
    }
    if dataflash is not None and snapshot.size and snapshot.shape[1] >= 5:
        snap_window = (snapshot_t >= start) & (snapshot_t <= stop)
        boot_s = snapshot[snap_window, 4] * 1.0e-3
        boot_s = boot_s[boot_s >= 0.0]
        if boot_s.size >= 2:
            result["dataflash"] = dataflash_metrics(
                dataflash, float(boot_s[0]), float(boot_s[-1])
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataflash")
    args = parser.parse_args()
    result = analyze(args.bag, args.dataflash)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

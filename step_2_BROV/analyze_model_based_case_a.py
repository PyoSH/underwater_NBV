#!/usr/bin/env python3
"""Analyze a model-based-controller Case-A GT/DVL-EKF deployment pair.

This is a causal-isolation companion to ``analyze_mk2_case_a_ab.py``: it
reuses the generic (policy-agnostic) phase/velocity-tracking logic from
``analyze_stage2_case_a_ab`` and adds a direct PWM-based jitter metric
computed from ``/brov/thruster_pwm`` (a topic the model-based controller
also publishes, on the same [-1, 1] normalized scale as the RL policy).

The RL-specific ``action_and_actuation`` sub-metrics in the reused phase
dict will be all-NaN for these bags (the model-based controller does not
publish ``/brov/action`` or ``/brov/policy/*``) and should be ignored.

All rosbag access is read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_brov_stage1_ab as stage1
import analyze_stage2_case_a_ab as stage2


def _pwm_jitter(bag: dict, start: float, stop: float, settle_s: float) -> dict:
    t, pwm = stage1.arrays(bag["/brov/thruster_pwm"])
    window_start = start + settle_s
    mask = (t >= window_start) & (t <= stop)
    t = t[mask]
    pwm = pwm[mask]
    if pwm.size == 0:
        return {"samples": 0}
    any_ge_0p99 = np.any(np.abs(pwm) >= 0.99, axis=1)
    sign = np.sign(pwm)
    flips = np.abs(np.diff(sign, axis=0)) >= 2.0  # -1 -> +1 or +1 -> -1
    flip_rate_per_axis = flips.mean(axis=0) if flips.shape[0] > 0 else np.zeros(8)
    return {
        "samples": int(pwm.shape[0]),
        "any_axis_ge_0p99_fraction": float(np.mean(any_ge_0p99)),
        "per_axis_ge_0p99_fraction": np.mean(np.abs(pwm) >= 0.99, axis=0).tolist(),
        "per_axis_sign_flip_rate": flip_rate_per_axis.tolist(),
        "max_abs_per_axis": np.max(np.abs(pwm), axis=0).tolist(),
        "rms_per_axis": np.sqrt(np.mean(pwm**2, axis=0)).tolist(),
    }


def analyze_model_based_one(
    uri: str,
    *,
    expected_feedback_source: str,
    settle_s: float,
    turn_half_window_s: float,
    pair_tolerance_s: float,
    debounce_samples: int,
) -> dict:
    result = stage2.analyze_one(
        uri,
        expected_feedback_source=expected_feedback_source,
        settle_s=settle_s,
        turn_half_window_s=turn_half_window_s,
        pair_tolerance_s=pair_tolerance_s,
        debounce_samples=debounce_samples,
    )
    bag = stage2.read_bag(uri)
    pwm = {}
    if result.get("phases"):
        for phase_name in ("outbound", "return"):
            phase = result["phases"][phase_name]
            pwm[phase_name] = _pwm_jitter(
                bag,
                float(phase["start_unix_s"]) + float(phase["settle_excluded_s"]),
                float(phase["stop_unix_s"]) - float(phase["end_excluded_s"]),
                0.0,
            )
        active = result["active"]
        pwm["whole_cycle"] = _pwm_jitter(
            bag, float(active["start_unix_s"]), float(active["stop_unix_s"]), settle_s
        )
    result["pwm_jitter"] = pwm
    return result


def _summary(result: dict) -> dict:
    out = {"waypoint_rle": result["waypoint_rle"].get("observed")}
    for phase_name in ("outbound", "return"):
        phase = result.get("phases", {}).get(phase_name)
        if not phase:
            continue
        vt = phase["velocity_tracking"]["gazebo_ground_truth"]
        pwm = result["pwm_jitter"].get(phase_name, {})
        out[phase_name] = {
            "vector_error_rmse_mps": vt.get("vector_error_rmse_mps"),
            "v_parallel_mean_mps": vt.get("v_parallel_mean_mps"),
            "cross_speed_rms_mps": vt.get("cross_speed_rms_mps"),
            "pwm_any_axis_ge_0p99_fraction": pwm.get("any_axis_ge_0p99_fraction"),
            "pwm_max_sign_flip_rate": (
                max(pwm["per_axis_sign_flip_rate"])
                if pwm.get("per_axis_sign_flip_rate")
                else None
            ),
        }
    whole = result["pwm_jitter"].get("whole_cycle", {})
    out["whole_cycle"] = {
        "pwm_any_axis_ge_0p99_fraction": whole.get("any_axis_ge_0p99_fraction"),
        "pwm_max_sign_flip_rate": (
            max(whole["per_axis_sign_flip_rate"])
            if whole.get("per_axis_sign_flip_rate")
            else None
        ),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_bag")
    parser.add_argument("dvl_bag")
    parser.add_argument("--output", required=True)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--turn-half-window-s", type=float, default=0.05)
    parser.add_argument("--pair-tolerance-s", type=float, default=0.03)
    parser.add_argument("--debounce-samples", type=int, default=3)
    args = parser.parse_args()

    gt = analyze_model_based_one(
        args.gt_bag,
        expected_feedback_source="gazebo_truth",
        settle_s=args.settle_s,
        turn_half_window_s=args.turn_half_window_s,
        pair_tolerance_s=args.pair_tolerance_s,
        debounce_samples=args.debounce_samples,
    )
    dvl = analyze_model_based_one(
        args.dvl_bag,
        expected_feedback_source="mavlink_ekf",
        settle_s=args.settle_s,
        turn_half_window_s=args.turn_half_window_s,
        pair_tolerance_s=args.pair_tolerance_s,
        debounce_samples=args.debounce_samples,
    )
    output = {
        "schema": "brov_model_based_case_a_gt_dvl_ab_v1",
        "runs": {"gazebo_truth": gt, "dvl_ekf": dvl},
        "summary": {
            "gazebo_truth": _summary(gt),
            "dvl_ekf": _summary(dvl),
        },
    }
    encoded = json.dumps(stage1.json_safe(output), indent=2, allow_nan=True)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze one Case-C (5 m square, random-attitude-per-corner) Gazebo bag.

Deliberately lighter-weight than analyze_stage2_case_a_ab.py: Case-A's
outbound/return/turn phase model and strict pass/fail gate system are
specific to its 3-point takeoff_then_align sequence and are not reused here.
This reports continuous, directly-comparable metrics across all five
controllers under test (model-based, i299/deploy_v2, deploy_v3, deploy_v4,
deploy_v5) instead of a gate verdict:

* whole-cycle velocity tracking RMSE (common to every controller)
* whole-cycle actuator saturation -- raw policy action (MK2 profiles) or
  final PWM (model-based, reusing analyze_model_based_case_a.py's own
  _pwm_jitter, which is controller-agnostic) since MK2 and model-based
  don't share a raw-action representation
* per-corner settle time: how long after each waypoint_idx transition the
  body rate stays above a calming threshold, a frame-agnostic proxy for
  "how long did the random-attitude reorientation disturb the vehicle"
  that avoids the NED/ZUP frame-transform ambiguity a direct quaternion
  comparison would need (see diagnose_attitude_torque_budget.py's own
  docstring for why that direct comparison was avoided there too)

Read-only rosbag access.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import analyze_brov_stage1_ab as stage1
from analyze_model_based_case_a import _pwm_jitter

# Not in stage1.ALL_TOPICS by default (analyze_mk2_case_a_ab.py registers it
# on stage2's separate topic set instead, since this script calls
# stage1.read_bag() directly rather than going through stage2).
_ARTIFACT_TOPIC = "/brov/policy/artifact_contract"
stage1.STRING_TOPICS.add(_ARTIFACT_TOPIC)
stage1.ALL_TOPICS.add(_ARTIFACT_TOPIC)

MK2_ACTION_CONTRACT = "explicit_flu_zup_to_sname_frd_v1"
MK2_OBSERVATION_CONTRACT = "brov_velocity_observation_v2"
WRENCH_SCALE = np.asarray([85.0, 85.0, 120.0, 26.0, 14.0, 22.0])
T6 = np.asarray([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
AXIS_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
SETTLE_OMEGA_DEG_S = 10.0


def _gate(value, passed: bool, criterion: str) -> dict:
    return {"value": value, "criterion": criterion, "pass": bool(passed)}


def _artifact_contract(bag: dict, *, expected_policy_sha256: str, expected_profile: str):
    topic = "/brov/policy/artifact_contract"
    payloads = []
    for _, raw in bag.get(topic, []):
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    canonical = {
        json.dumps(item, sort_keys=True, separators=(",", ":")): item
        for item in payloads
    }
    unique = list(canonical.values())
    contract = unique[0] if len(unique) == 1 else {}
    checks = {
        "one_unique_contract": _gate(len(unique), len(unique) == 1, "exactly one contract value"),
        "metadata_verified": _gate(
            contract.get("metadata_verified"), contract.get("metadata_verified") is True,
            "runtime metadata verification is true",
        ),
        "policy_sha256": _gate(
            contract.get("policy_sha256"), contract.get("policy_sha256") == expected_policy_sha256,
            f"policy SHA256 == {expected_policy_sha256}",
        ),
        "profile": _gate(
            contract.get("profile"), contract.get("profile") == expected_profile,
            f"profile == {expected_profile}",
        ),
    }
    checks["pass"] = all(v["pass"] for v in checks.values() if isinstance(v, dict))
    return contract, checks


def _velocity_tracking(bag: dict, start: float, stop: float, settle_s: float) -> dict:
    vt, v_act = stage1.arrays(bag["/brov/debug/v_body_zup"])
    vdt, v_des = stage1.arrays(bag["/brov/debug/v_desired_body_zup"])
    window_start = start + settle_s
    mask = (vt >= window_start) & (vt <= stop)
    query_t = vt[mask]
    query_act = v_act[mask]
    paired_des, valid, _ = stage1.nearest(vdt, v_des, query_t, tolerance_s=0.05)
    finite = valid & np.all(np.isfinite(query_act), axis=1) & np.all(np.isfinite(paired_des), axis=1)
    if not np.any(finite):
        return {"samples": 0}
    err = query_act[finite] - paired_des[finite]
    speed_des = np.linalg.norm(paired_des[finite], axis=1)
    return {
        "samples": int(np.count_nonzero(finite)),
        "desired_speed_mean_mps": float(np.mean(speed_des)),
        "vector_error_rmse_mps": float(np.sqrt(np.mean((err**2).sum(-1)))),
        "v_parallel_over_command": float(
            np.mean(np.sum(query_act[finite] * paired_des[finite], axis=1) / np.clip(speed_des**2, 1e-6, None))
        ),
    }


def _mk2_action_saturation(bag: dict, start: float, stop: float, settle_s: float) -> dict:
    at, action = stage1.arrays(bag["/brov/action"])
    window_start = start + settle_s
    mask = (at >= window_start) & (at <= stop)
    action = action[mask]
    if action.shape[0] == 0:
        return {"samples": 0}
    wt, wrench = stage1.arrays(bag["/brov/policy/wrench_requested"])
    paired_wrench, wvalid, _ = stage1.nearest(wt, wrench, at[mask], tolerance_s=0.05)
    ft, f_req = stage1.arrays(bag["/brov/policy/thruster_force_requested"])
    flt, f_lim = stage1.arrays(bag["/brov/policy/thruster_force_limited"])
    paired_freq, frvalid, _ = stage1.nearest(ft, f_req, at[mask], tolerance_s=0.05)
    paired_flim, flvalid, _ = stage1.nearest(flt, f_lim, at[mask], tolerance_s=0.05)
    force_valid = frvalid & flvalid
    clamp_residual = paired_freq[force_valid] - paired_flim[force_valid]
    per_axis = {
        name: float(np.mean(np.abs(action[:, i]) >= 0.99)) for i, name in enumerate(AXIS_NAMES)
    }
    return {
        "samples": int(action.shape[0]),
        "any_axis_ge_0p99_fraction": float(np.mean(np.any(np.abs(action) >= 0.99, axis=1))),
        "axis_ge_0p99_fraction": per_axis,
        "wrench_mean_abs_Nm": (
            np.mean(np.abs(paired_wrench[wvalid]), axis=0).tolist() if np.any(wvalid) else None
        ),
        "thruster_force_any_clamp_fraction": (
            float(np.mean(np.any(np.abs(clamp_residual) > 1e-3, axis=1)))
            if clamp_residual.size
            else None
        ),
    }


def _corner_settle(bag: dict, start: float, stop: float) -> dict:
    wt, wpidx = stage1.arrays(bag["/brov/waypoint_idx"])
    ot, omega = stage1.arrays(bag["/brov/debug/feedback_body_rates_frd"])
    mask = (wt >= start) & (wt <= stop)
    wt = wt[mask]
    wpidx = wpidx[mask].astype(int).reshape(-1)
    transitions = []
    for k in range(1, len(wpidx)):
        if wpidx[k] != wpidx[k - 1]:
            transitions.append((float(wt[k]), int(wpidx[k])))
    corner_results = []
    for i, (t_transition, idx) in enumerate(transitions):
        t_next = transitions[i + 1][0] if i + 1 < len(transitions) else stop
        window = (ot >= t_transition) & (ot < t_next)
        om = np.degrees(np.linalg.norm(omega[window], axis=1))
        if om.size == 0:
            continue
        below = om < SETTLE_OMEGA_DEG_S
        # "Settled" = the first point after which body rate stays under the
        # threshold for a sustained 0.5 s run, not literally forever --
        # requiring it to hold to the end of an up-to-8s window made every
        # single noise blip anywhere later invalidate detection entirely.
        sustain_samples = max(1, int(round(0.5 / max(np.median(np.diff(ot[window])), 1e-3))))
        settle_s = None
        for j in range(len(below) - sustain_samples + 1):
            if np.all(below[j : j + sustain_samples]):
                settle_s = float(ot[window][j] - t_transition)
                break
        corner_results.append(
            {
                "corner_idx": idx,
                "t_transition_s": t_transition,
                "peak_omega_deg_s": float(np.max(om)),
                "settle_s": settle_s,
            }
        )
    return {"transitions": transitions, "corners": corner_results}


def analyze_one(
    uri: str,
    *,
    controller: str,
    expected_feedback_source: str,
    settle_s: float,
    expected_policy_sha256: str | None = None,
    expected_profile: str | None = None,
) -> dict:
    bag = stage1.read_bag(uri)
    start_s, stop_s = stage1.longest_active_run(bag["/brov/control_active"])
    result: dict = {
        "bag": uri,
        "controller": controller,
        "feedback_source_expected": expected_feedback_source,
        "active_window_s": [start_s, stop_s],
        "duration_s": stop_s - start_s,
    }

    if controller == "mk2":
        contract, checks = _artifact_contract(
            bag,
            expected_policy_sha256=expected_policy_sha256,
            expected_profile=expected_profile,
        )
        result["mk2_artifact"] = {"contract": contract, "checks": checks}
        result["action_saturation"] = _mk2_action_saturation(bag, start_s, stop_s, settle_s)
    else:
        result["pwm_jitter"] = _pwm_jitter(bag, start_s, stop_s, settle_s)

    result["velocity_tracking"] = _velocity_tracking(bag, start_s, stop_s, settle_s)
    result["corner_settle"] = _corner_settle(bag, start_s, stop_s)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--controller", required=True, choices=["mk2", "model_based"])
    parser.add_argument("--feedback-source", required=True, choices=["gazebo_truth", "mavlink_ekf"])
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--policy-sha256", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = analyze_one(
        args.bag,
        controller=args.controller,
        expected_feedback_source=args.feedback_source,
        settle_s=args.settle_s,
        expected_policy_sha256=args.policy_sha256,
        expected_profile=args.profile,
    )
    encoded = json.dumps(stage1.json_safe(result), indent=2, allow_nan=False)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze the metadata-bound MK2 Case-A GT/DVL-EKF deployment pair.

This extends :mod:`analyze_stage2_case_a_ab` with two deployment gates that
cannot be inferred from trajectory quality alone:

* both bags must contain one verified MK2 artifact contract with the expected
  policy, observation and action-frame hashes/contracts;
* every active requested wrench must equal the clipped FLU/Z-up action scaled
  by ``[85,85,120,26,14,22]`` and transformed by
  ``T6=diag(1,-1,-1,1,-1,-1)`` before allocation.

All rosbag access is read-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import analyze_brov_stage1_ab as stage1
import analyze_stage2_case_a_ab as stage2


ARTIFACT_TOPIC = "/brov/policy/artifact_contract"
EXPECTED_POLICY_SHA256 = (
    "c185869418f13d868b8d71c4ca8f6f245a9d7103bca36704870df4a738ac2c4f"
)
EXPECTED_VEHICLE_SHA256 = (
    "8bb397f4a8a0d50c11bfaf1f88143b4375b84dbd77f41fbe7c512ced1b15be12"
)
EXPECTED_ACTION_CONTRACT = "explicit_flu_zup_to_sname_frd_v1"
EXPECTED_OBSERVATION_CONTRACT = "brov_velocity_observation_v2"
WRENCH_SCALE = np.asarray([85.0, 85.0, 120.0, 26.0, 14.0, 22.0])
T6 = np.asarray([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

# The Stage-2 reader dispatches from module-level topic sets.  Extend those
# before calling it so the provenance topic participates in its required-topic
# gate as well as the MK2 semantic checks below.
stage2.STRING_TOPICS.add(ARTIFACT_TOPIC)
stage2.ALL_TOPICS.add(ARTIFACT_TOPIC)
stage2.CORE_REQUIRED.add(ARTIFACT_TOPIC)


def _gate(value, passed: bool, criterion: str, *, evaluated: bool = True) -> dict:
    return {
        "value": value,
        "criterion": criterion,
        "evaluated": bool(evaluated),
        "pass": bool(evaluated and passed),
    }


def _artifact_contract(
    bag: dict, *, expected_policy_sha256: str, expected_profile: str
) -> tuple[dict, dict]:
    payloads = []
    parse_errors = []
    for _, raw in bag.get(ARTIFACT_TOPIC, []):
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
        else:
            parse_errors.append("artifact payload is not a JSON object")

    canonical = {
        json.dumps(item, sort_keys=True, separators=(",", ":")): item
        for item in payloads
    }
    unique = list(canonical.values())
    contract = unique[0] if len(unique) == 1 else {}
    checks = {
        "one_unique_contract": _gate(
            len(unique),
            len(unique) == 1 and not parse_errors,
            "exactly one parseable artifact contract value",
        ),
        "metadata_verified": _gate(
            contract.get("metadata_verified"),
            contract.get("metadata_verified") is True,
            "runtime metadata verification is true",
        ),
        "policy_sha256": _gate(
            contract.get("policy_sha256"),
            contract.get("policy_sha256") == expected_policy_sha256,
            f"exact MK2 policy SHA256 {expected_policy_sha256}",
        ),
        "vehicle_model_sha256": _gate(
            contract.get("vehicle_model_sha256"),
            contract.get("vehicle_model_sha256") == EXPECTED_VEHICLE_SHA256,
            f"exact vehicle-model SHA256 {EXPECTED_VEHICLE_SHA256}",
        ),
        "action_contract": _gate(
            contract.get("action_contract"),
            contract.get("action_contract") == EXPECTED_ACTION_CONTRACT,
            f"exact action contract {EXPECTED_ACTION_CONTRACT}",
        ),
        "observation_contract": _gate(
            contract.get("observation_contract"),
            contract.get("observation_contract")
            == EXPECTED_OBSERVATION_CONTRACT,
            f"exact observation contract {EXPECTED_OBSERVATION_CONTRACT}",
        ),
        "profile": _gate(
            contract.get("profile"),
            contract.get("profile") == expected_profile,
            f"profile is {expected_profile}",
        ),
    }
    checks["pass"] = all(
        item["pass"] for item in checks.values() if isinstance(item, dict)
    )
    return {
        "message_count": len(payloads),
        "unique_value_count": len(unique),
        "parse_errors": parse_errors,
        "contract": contract,
    }, checks


def _action_to_wrench(
    bag: dict, start: float, stop: float, tolerance_s: float
) -> tuple[dict, dict]:
    action_t, action = stage1.arrays(bag["/brov/action"])
    wrench_t, wrench = stage1.arrays(bag["/brov/policy/wrench_requested"])
    in_active = (action_t >= start) & (action_t <= stop)
    query_t = action_t[in_active]
    query_action = action[in_active]
    paired_wrench, valid, skew = stage1.nearest(
        wrench_t, wrench, query_t, tolerance_s=tolerance_s
    )
    finite = valid
    finite &= np.all(np.isfinite(query_action), axis=1)
    finite &= np.all(np.isfinite(paired_wrench), axis=1)
    if np.any(finite):
        expected = query_action[finite] * WRENCH_SCALE * T6
        residual = paired_wrench[finite] - expected
        max_abs = float(np.max(np.abs(residual)))
        rms = float(np.sqrt(np.mean(residual**2)))
        skew_p95_ms = float(np.percentile(np.abs(skew[finite]), 95) * 1000.0)
    else:
        max_abs = math.nan
        rms = math.nan
        skew_p95_ms = math.nan
    metrics = {
        "action_frame": "body_flu_zup",
        "allocation_wrench_frame": "body_frd_sname",
        "wrench_scale": WRENCH_SCALE.tolist(),
        "t6_diagonal": T6.tolist(),
        "active_action_samples": int(query_t.size),
        "paired_samples": int(np.count_nonzero(finite)),
        "pairing_skew_p95_ms": skew_p95_ms,
        "residual_max_abs": max_abs,
        "residual_rms": rms,
    }
    gates = {
        "coverage": _gate(
            int(np.count_nonzero(finite)),
            query_t.size > 0 and np.count_nonzero(finite) == query_t.size,
            "every active action has a paired requested wrench",
        ),
        "t6_wrench_identity": _gate(
            max_abs,
            math.isfinite(max_abs) and max_abs <= 1.0e-4,
            "max |wrench - action*scale*T6| <= 1e-4",
            evaluated=math.isfinite(max_abs),
        ),
    }
    gates["pass"] = all(
        item["pass"] for item in gates.values() if isinstance(item, dict)
    )
    return metrics, gates


def analyze_mk2_one(
    uri: str,
    *,
    expected_feedback_source: str,
    settle_s: float,
    turn_half_window_s: float,
    pair_tolerance_s: float,
    debounce_samples: int,
    expected_policy_sha256: str = EXPECTED_POLICY_SHA256,
    expected_profile: str = "deploy_v2",
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
    artifact, artifact_gates = _artifact_contract(
        bag,
        expected_policy_sha256=expected_policy_sha256,
        expected_profile=expected_profile,
    )
    active = result["active"]
    transform, transform_gates = _action_to_wrench(
        bag,
        float(active["start_unix_s"]),
        float(active["stop_unix_s"]),
        pair_tolerance_s,
    )
    mk2_gates = {
        "artifact": artifact_gates,
        "action_to_wrench": transform_gates,
        "pass": bool(artifact_gates["pass"] and transform_gates["pass"]),
    }
    result["mk2_artifact"] = artifact
    result["mk2_action_to_wrench"] = transform
    result["gates"]["mk2"] = mk2_gates
    result["gates"]["overall_pass"] = bool(
        result["gates"]["overall_pass"] and mk2_gates["pass"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_bag")
    parser.add_argument("dvl_bag")
    parser.add_argument("--output", required=True)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--turn-half-window-s", type=float, default=0.05)
    parser.add_argument("--pair-tolerance-s", type=float, default=0.03)
    parser.add_argument("--debounce-samples", type=int, default=3)
    parser.add_argument(
        "--policy-sha256",
        default=EXPECTED_POLICY_SHA256,
        help="expected policy_sha256 in /brov/policy/artifact_contract "
        "(default: the sim2swim_deploy_v2_mk2_s42_i49 policy)",
    )
    parser.add_argument(
        "--profile",
        default="deploy_v2",
        choices=["deploy_v2", "deploy_v3", "deploy_v4", "deploy_v5"],
        help="expected training profile in /brov/policy/artifact_contract",
    )
    args = parser.parse_args()

    gt = analyze_mk2_one(
        args.gt_bag,
        expected_feedback_source="gazebo_truth",
        settle_s=args.settle_s,
        turn_half_window_s=args.turn_half_window_s,
        pair_tolerance_s=args.pair_tolerance_s,
        debounce_samples=args.debounce_samples,
        expected_policy_sha256=args.policy_sha256,
        expected_profile=args.profile,
    )
    dvl = analyze_mk2_one(
        args.dvl_bag,
        expected_feedback_source="mavlink_ekf",
        settle_s=args.settle_s,
        turn_half_window_s=args.turn_half_window_s,
        pair_tolerance_s=args.pair_tolerance_s,
        debounce_samples=args.debounce_samples,
        expected_policy_sha256=args.policy_sha256,
        expected_profile=args.profile,
    )
    comparison = stage2.compare(gt, dvl)
    same_contract = (
        gt["mk2_artifact"]["contract"] == dvl["mk2_artifact"]["contract"]
        and bool(gt["mk2_artifact"]["contract"])
    )
    comparison["mk2_artifact_exact_match"] = _gate(
        same_contract,
        same_contract,
        "GT and DVL-EKF bags contain the identical verified artifact contract",
    )
    comparison["overall_pass"] = bool(
        comparison.get("overall_pass", False) and same_contract
    )
    output = {
        "schema": "brov_mk2_case_a_gt_dvl_ab_v1",
        "contract": {
            "expected_waypoint_rle": stage2.EXPECTED_RLE,
            "settle_s": args.settle_s,
            "turn_half_window_s": args.turn_half_window_s,
            "pair_tolerance_s": args.pair_tolerance_s,
            "policy_sha256": args.policy_sha256,
            "action_contract": EXPECTED_ACTION_CONTRACT,
            "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        },
        "runs": {"gazebo_truth": gt, "dvl_ekf": dvl},
        "comparison": comparison,
    }
    encoded = json.dumps(stage1.json_safe(output), indent=2, allow_nan=False)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

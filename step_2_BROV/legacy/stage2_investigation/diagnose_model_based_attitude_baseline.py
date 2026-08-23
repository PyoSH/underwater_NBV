#!/usr/bin/env python3
"""Sanity-check baseline: does the SAME Gazebo environment/sensors produce
comparable attitude oscillation under the model-based controller (already
known to have ~0% saturation)? This sidesteps reconstructing the RL
pipeline's exact q_e frame chain -- body rate and commanded wrench are
recorded in the same units/topics for both controllers, so they are
directly comparable without a frame transform.

Read-only rosbag access. No IsaacLab/torch dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import analyze_brov_stage1_ab as stage1

F_MAX = {"roll": 26.0, "pitch": 14.0, "yaw": 22.0}
AXIS_IDX = {"surge": 0, "sway": 1, "heave": 2, "roll": 3, "pitch": 4, "yaw": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    bag = stage1.read_bag(args.bag)
    start_s, stop_s = stage1.longest_active_run(bag["/brov/control_active"])

    omega_t, omega = stage1.arrays(bag["/brov/debug/feedback_body_rates_frd"])
    in_active_omega = (omega_t >= start_s) & (omega_t <= stop_s)
    omega_active = omega[in_active_omega]
    omega_deg_s = np.degrees(omega_active)

    payload = {
        "bag": args.bag,
        "duration_s": float(stop_s - start_s),
        "body_rate_deg_s": {
            axis: {
                "mean_abs": float(np.mean(np.abs(omega_deg_s[:, i]))),
                "p95_abs": float(np.percentile(np.abs(omega_deg_s[:, i]), 95)),
                "max_abs": float(np.max(np.abs(omega_deg_s[:, i]))),
            }
            for axis, i in {"roll": 0, "pitch": 1, "yaw": 2}.items()
        },
    }

    wrench_topic = "/brov/model_based/wrench_zup"
    if wrench_topic in bag and bag[wrench_topic]:
        wrench_t, wrench = stage1.arrays(bag[wrench_topic])
        in_active_w = (wrench_t >= start_s) & (wrench_t <= stop_s)
        w_active = wrench[in_active_w]
        payload["commanded_wrench"] = {
            axis: {
                "mean_abs_Nm": float(np.mean(np.abs(w_active[:, i]))),
                "at_or_above_0p99_fmax_fraction": float(
                    np.mean(np.abs(w_active[:, i]) >= 0.99 * F_MAX[axis])
                ),
            }
            for axis, i in {"roll": 3, "pitch": 4, "yaw": 5}.items()
        }

    Path(args.output_json).write_text(
        json.dumps(stage1.json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stage1.json_safe(payload), indent=2))


if __name__ == "__main__":
    main()

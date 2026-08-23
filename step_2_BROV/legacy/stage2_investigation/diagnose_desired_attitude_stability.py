#!/usr/bin/env python3
"""Check whether the remaining roll/pitch/yaw saturation is caused by an
unstable/noisy *desired* attitude (guidance problem) or by the vehicle
failing to track an otherwise-stable desired attitude (torque/tracking
problem).

Two independent, frame-agnostic signals:
1. q_e (observation[0:4], the quaternion error the policy itself consumes)
   -- its instantaneous angle IS the tracking error already computed by the
   deployed pipeline, no NED/ZUP conversion needed.
2. q_desired_zup frame-to-frame delta angle / dt -- the *implied* commanded
   angular rate of the desired attitude itself. If this is large/noisy
   throughout cruise (not just at the one expected ~180 deg return-leg
   reversal), the target itself is unstable (guidance problem). If it is
   small/smooth except at that one event, the target is fine and the
   vehicle is failing to track it (torque/tracking problem).

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


def q_error_angle_deg(q_e: np.ndarray) -> np.ndarray:
    w = np.clip(np.abs(q_e[:, 0]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    bag = stage1.read_bag(args.bag)
    start_s, stop_s = stage1.longest_active_run(bag["/brov/control_active"])

    obs_t, obs = stage1.arrays(bag["/brov/observation"])
    qdes_t, qdes = stage1.arrays(bag["/brov/debug/q_desired_zup"])
    wpidx_t, wpidx = stage1.arrays(bag["/brov/waypoint_idx"])

    in_active = (obs_t >= start_s) & (obs_t <= stop_s)
    t0 = obs_t[in_active][0]
    t = obs_t[in_active] - t0
    obs_active = obs[in_active]
    q_e = obs_active[:, 0:4]
    error_angle_deg = q_error_angle_deg(q_e)

    # q_desired_zup frame-to-frame delta, resampled onto the same timeline.
    qdes_paired, qdes_valid, _ = stage1.nearest(
        qdes_t, qdes, obs_t[in_active], tolerance_s=0.05
    )
    qdes_paired = stage1.q_normalize(qdes_paired)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.nan
    desired_delta_deg = np.zeros(len(t))
    desired_delta_deg[1:] = stage1.q_angle_deg(qdes_paired[:-1], qdes_paired[1:])
    implied_desired_rate_deg_s = desired_delta_deg / np.where(
        np.isnan(dt), np.nan, dt
    )

    paired_wpidx, wpidx_valid, _ = stage1.nearest(
        wpidx_t, wpidx, obs_t[in_active], tolerance_s=0.05
    )
    idx_vals = paired_wpidx[wpidx_valid].astype(int)
    idx_times = t[wpidx_valid]
    transitions = []
    for k in range(1, len(idx_vals)):
        if idx_vals[k] != idx_vals[k - 1]:
            transitions.append((float(idx_times[k]), int(idx_vals[k])))
    return_window = None
    if len(transitions) >= 3:
        return_window = (transitions[-2][0], transitions[-1][0])

    # Exclude a short window right after each transition (genuine
    # re-target events) to characterize *cruise* stability specifically.
    settle_s = 1.0
    near_transition = np.zeros(len(t), dtype=bool)
    for tt, _ in transitions:
        near_transition |= (t >= tt) & (t < tt + settle_s)
    near_transition |= t < settle_s
    cruise_mask = ~near_transition & qdes_valid[: len(t)]

    payload = {
        "bag": args.bag,
        "duration_s": float(stop_s - start_s),
        "waypoint_transitions_s": transitions,
        "return_window_s": list(return_window) if return_window else None,
        "tracking_error_deg": {
            "whole_window_mean": float(np.mean(error_angle_deg)),
            "whole_window_p95": float(np.percentile(error_angle_deg, 95)),
            "whole_window_max": float(np.max(error_angle_deg)),
            "cruise_only_mean": float(np.mean(error_angle_deg[cruise_mask])),
            "cruise_only_p95": float(np.percentile(error_angle_deg[cruise_mask], 95)),
        },
        "implied_desired_attitude_rate_deg_s": {
            "cruise_only_mean_abs": float(
                np.nanmean(np.abs(implied_desired_rate_deg_s[cruise_mask]))
            ),
            "cruise_only_p95_abs": float(
                np.nanpercentile(np.abs(implied_desired_rate_deg_s[cruise_mask]), 95)
            ),
            "cruise_only_max_abs": float(
                np.nanmax(np.abs(implied_desired_rate_deg_s[cruise_mask]))
            ),
            "whole_window_max_abs": float(
                np.nanmax(np.abs(implied_desired_rate_deg_s))
            ),
        },
    }
    Path(args.output_json).write_text(
        json.dumps(stage1.json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    ax = axes[0]
    ax.plot(t, error_angle_deg, color="tab:purple", label="q_e angle (policy-seen tracking error)")
    ax.set_ylabel("attitude error (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(args.title or Path(args.bag).parent.name)

    ax = axes[1]
    ax.plot(
        t,
        implied_desired_rate_deg_s,
        color="tab:orange",
        label="implied desired-attitude rate (deg/s)",
    )
    ax.set_ylabel("implied desired\nangular rate (deg/s)")
    ax.set_xlabel("t (s, active window)")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        if return_window is not None:
            ax.axvspan(return_window[0], return_window[1], color="orange", alpha=0.15)
        for tt, _ii in transitions:
            ax.axvline(tt, color="gray", linestyle="-", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=130)

    print(json.dumps(stage1.json_safe(payload), indent=2))


if __name__ == "__main__":
    main()

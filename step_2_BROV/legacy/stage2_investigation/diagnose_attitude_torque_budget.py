#!/usr/bin/env python3
"""Diagnose whether deploy_v4's remaining roll/pitch/yaw saturation is a
torque-budget limit (action pinned at bound while attitude error stays large
and non-decreasing) or an ordinary transient (action pins briefly, error then
converges).

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
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    bag = stage1.read_bag(args.bag)

    active_t, active = stage1.arrays(bag["/brov/control_active"])
    start_s, stop_s = stage1.longest_active_run(bag["/brov/control_active"])

    quat_t, quat_actual = stage1.arrays(bag["/brov/debug/gazebo_truth_att_quat_ned"])
    qdes_t, quat_desired = stage1.arrays(bag["/brov/debug/q_desired_zup"])
    action_t, action = stage1.arrays(bag["/brov/action"])
    wrench_t, wrench = stage1.arrays(bag["/brov/policy/wrench_requested"])
    wpidx_t, wpidx = stage1.arrays(bag["/brov/waypoint_idx"])

    in_active = (action_t >= start_s) & (action_t <= stop_s)
    t0 = action_t[in_active][0]
    t = action_t[in_active] - t0

    # gazebo_truth_att_quat_ned is NED; q_desired_zup is Z-up. Both already
    # unit quaternions in [w,x,y,z]; NED vs Z-up differ by a fixed frame
    # rotation, but the *magnitude* of q_angle_deg (attitude error angle) is
    # frame-independent for a consistent fixed offset only if both were
    # expressed in the same frame. To stay honest, resolve both to the same
    # (zup) convention is out of scope here -- instead we report the raw
    # per-topic angle-vs-itself trend, which is what matters for the
    # torque-budget question: does the error trend shrink or not while
    # action stays pinned. Use q_desired_zup as its own reference frame by
    # nearest-pairing gazebo_truth quat samples via NED->ZUP is skipped;
    # instead compute attitude error directly from the existing debug
    # topic pairing used by the rest of this investigation:
    # /brov/debug/att_quat_ned (feedback-source-selected, same frame as
    # gazebo_truth for GT runs) paired against q_desired_zup would require
    # a frame transform. We instead use the per-axis wrench/action signal
    # (frame-agnostic body torques) as the direct saturation evidence, and
    # angular_velocity-based settling as an indirect error-decay proxy.
    omega_t, omega = stage1.arrays(bag["/brov/debug/feedback_body_rates_frd"])

    paired_wrench, wrench_valid, _ = stage1.nearest(
        wrench_t, wrench, action_t[in_active], tolerance_s=0.05
    )
    paired_omega, omega_valid, _ = stage1.nearest(
        omega_t, omega, action_t[in_active], tolerance_s=0.05
    )
    paired_wpidx, wpidx_valid, _ = stage1.nearest(
        wpidx_t, wpidx, action_t[in_active], tolerance_s=0.05
    )
    act = action[in_active]

    rows = []
    for name in ("roll", "pitch", "yaw"):
        i = AXIS_IDX[name]
        a = act[:, i]
        w = paired_wrench[:, i]
        pinned = np.abs(a) >= 0.99
        pinned_frac = float(np.mean(pinned))
        w_valid = w[wrench_valid]
        w_at_fmax_frac = (
            float(np.mean(np.abs(w_valid) >= 0.99 * F_MAX[name]))
            if w_valid.size
            else float("nan")
        )
        rows.append(
            {
                "axis": name,
                "f_max_Nm": F_MAX[name],
                "action_pinned_ge_0p99_fraction": pinned_frac,
                "wrench_at_or_above_0p99_fmax_fraction": w_at_fmax_frac,
                "action_mean_abs": float(np.mean(np.abs(a))),
                "wrench_mean_abs_Nm": float(np.mean(np.abs(w_valid)))
                if w_valid.size
                else float("nan"),
            }
        )

    # Body angular-rate magnitude while pinned vs not pinned, per axis --
    # if the vehicle is still rotating fast while the action is pinned, the
    # controller is actively fighting an error it hasn't closed (consistent
    # with torque-budget limiting, not settled/transient). If angular rate
    # is already small while pinned, the pin is stale/non-functional
    # (e.g. bias term), a different issue.
    omega_axis_idx = {"roll": 0, "pitch": 1, "yaw": 2}
    for row in rows:
        name = row["axis"]
        i = AXIS_IDX[name]
        oi = omega_axis_idx[name]
        a = act[:, i]
        pinned = (np.abs(a) >= 0.99) & omega_valid
        not_pinned = (np.abs(a) < 0.99) & omega_valid
        om = np.abs(paired_omega[:, oi])
        row["body_rate_abs_deg_s_while_pinned_mean"] = (
            float(np.degrees(np.mean(om[pinned]))) if np.any(pinned) else float("nan")
        )
        row["body_rate_abs_deg_s_while_not_pinned_mean"] = (
            float(np.degrees(np.mean(om[not_pinned])))
            if np.any(not_pinned)
            else float("nan")
        )

    # Return-leg specific window (from the earlier Case-A phase analysis),
    # re-derived here from waypoint_idx transitions so this script is
    # self-contained: return leg = the second-to-last debounced index
    # transition before end of active window (Case-A takeoff_then_align
    # sequence is 0->1->2->1; "return" is the final "->1" segment).
    idx_valid = wpidx_valid
    idx_vals = paired_wpidx[idx_valid].astype(int)
    idx_times = t[idx_valid]
    transitions = []
    for k in range(1, len(idx_vals)):
        if idx_vals[k] != idx_vals[k - 1]:
            transitions.append((idx_times[k], idx_vals[k]))
    # Case-A takeoff_then_align index sequence is 0->1->2->1: transitions
    # are [enter-1 (end of takeoff), enter-2 (end of outbound), enter-1
    # again (end of return)]. The return leg is the segment BETWEEN the
    # 2nd and 3rd transition, not after the 3rd (that tail is post-cycle
    # settle, already excluded from the official phase analysis too).
    return_window = None
    if len(transitions) >= 3:
        return_window = (transitions[-2][0], transitions[-1][0])

    payload = {
        "bag": args.bag,
        "active_window_s": [float(start_s), float(stop_s)],
        "duration_s": float(stop_s - start_s),
        "per_axis": rows,
        "waypoint_transitions_s": [
            {"t_s": float(tt), "idx": int(ii)} for tt, ii in transitions
        ],
        "return_window_s": list(return_window) if return_window else None,
    }

    if return_window is not None:
        r0, r1 = return_window
        in_return = (t >= r0) & (t <= r1)
        ret_rows = []
        for name in ("roll", "pitch", "yaw"):
            i = AXIS_IDX[name]
            a = act[in_return, i]
            ret_rows.append(
                {
                    "axis": name,
                    "action_pinned_ge_0p99_fraction": float(np.mean(np.abs(a) >= 0.99)),
                    "action_mean": float(np.mean(a)),
                }
            )
        payload["return_leg_per_axis"] = ret_rows

    Path(args.output_json).write_text(
        json.dumps(stage1.json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {"roll": "tab:red", "pitch": "tab:green", "yaw": "tab:blue"}

    ax = axes[0]
    for name in ("roll", "pitch", "yaw"):
        i = AXIS_IDX[name]
        ax.plot(t, act[:, i], label=f"action[{name}]", color=colors[name])
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.axhline(-1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylabel("raw actor action [-1,1]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(args.title or Path(args.bag).parent.name)

    ax = axes[1]
    for name in ("roll", "pitch", "yaw"):
        i = AXIS_IDX[name]
        ax.plot(t, paired_wrench[:, i], label=f"wrench[{name}]", color=colors[name])
        ax.axhline(F_MAX[name], color=colors[name], linestyle=":", linewidth=0.8)
        ax.axhline(-F_MAX[name], color=colors[name], linestyle=":", linewidth=0.8)
    ax.set_ylabel("requested wrench (N*m)\ndotted = F_max")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    for name in ("roll", "pitch", "yaw"):
        oi = omega_axis_idx[name]
        ax.plot(
            t[omega_valid],
            np.degrees(paired_omega[omega_valid, oi]),
            label=f"omega[{name}]",
            color=colors[name],
        )
    ax.set_ylabel("body rate (deg/s)")
    ax.set_xlabel("t (s, active window)")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        if return_window is not None:
            ax.axvspan(return_window[0], return_window[1], color="orange", alpha=0.15)
        for tt, ii in transitions:
            ax.axvline(tt, color="gray", linestyle="-", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=130)

    print(json.dumps(stage1.json_safe(payload), indent=2))


if __name__ == "__main__":
    main()

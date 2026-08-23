#!/usr/bin/env python3
"""Time-series + 3D trajectory plot for one Case-A Gazebo/SITL rosbag.

Mirrors ``test_policy.py._plot_results`` (Sim2Swim Fig.4 style: u/v/w
velocity panels + a combined attitude panel, dashed=desired/solid=actual,
surge-roll=red/sway-pitch=green/heave-yaw=blue) but reads directly from a
recorded rosbag instead of an in-process IsaacLab env, so it works for any
Gazebo/SITL Case-A run recorded by run_mk2_case_a_deploy.sh or
run_case_a_deploy_model_based.sh (RL policy or the classical model-based
controller) without needing IsaacLab/torch.

All rosbag access is read-only.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

import analyze_brov_stage1_ab as stage1
import analyze_stage2_case_a_ab as stage2

_FORWARD_ARROW_LEN = 0.5


def _quat_to_euler_zyx_deg(quat: np.ndarray) -> np.ndarray:
    """quat order [w,x,y,z] -> [roll,pitch,yaw] deg. Same formula as test_policy.py."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1) * (180.0 / np.pi)


def _forward_dir_from_quat(quat: np.ndarray) -> np.ndarray:
    """Body +x axis expressed in the world frame (same convention as test_policy.py)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    fz = 2 * (x * z - w * y)
    return np.stack([fx, fy, fz], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--output", required=True, help="output PNG path")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--pair-tolerance-s", type=float, default=0.1,
        help="nearest-neighbour join tolerance to the position timeline",
    )
    args = parser.parse_args()

    bag = stage2.read_bag(args.bag)
    active_runs = stage2._active_runs(bag["/brov/control_active"])
    if not active_runs:
        raise RuntimeError(f"{args.bag}: no active control interval")
    start, stop = max(active_runs, key=lambda pair: pair[1] - pair[0])

    pos_t, pos = stage1.arrays(bag["/brov/debug/gazebo_truth_pos_ned"])
    quat_t, quat = stage1.arrays(bag["/brov/debug/gazebo_truth_att_quat_ned"])
    vact_t, vact = stage1.arrays(bag["/brov/debug/v_body_zup"])
    vdes_t, vdes = stage1.arrays(bag["/brov/debug/v_desired_body_zup"])
    qdes_t, qdes = stage1.arrays(bag["/brov/debug/q_desired_zup"])
    wp_t, wp = stage1.arrays(bag["/brov/target_waypoint"])

    mask = (pos_t >= start) & (pos_t <= stop)
    query_t = pos_t[mask]
    if query_t.size == 0:
        raise RuntimeError(f"{args.bag}: no gazebo_truth_pos_ned samples in active window")
    t = query_t - start
    pos_w = pos[mask]

    tol = args.pair_tolerance_s
    quat_w, quat_valid, _ = stage1.nearest(quat_t, quat, query_t, tolerance_s=tol)
    vact_w, vact_valid, _ = stage1.nearest(vact_t, vact, query_t, tolerance_s=tol)
    vdes_w, vdes_valid, _ = stage1.nearest(vdes_t, vdes, query_t, tolerance_s=tol)
    qdes_w, qdes_valid, _ = stage1.nearest(qdes_t, qdes, query_t, tolerance_s=tol)

    valid = quat_valid & vact_valid & vdes_valid & qdes_valid
    t, pos_w, quat_w, vact_w, vdes_w, qdes_w = (
        t[valid], pos_w[valid], quat_w[valid], vact_w[valid], vdes_w[valid], qdes_w[valid]
    )
    if t.size == 0:
        raise RuntimeError(f"{args.bag}: no jointly-valid samples after nearest-neighbour join")

    euler = _quat_to_euler_zyx_deg(quat_w)
    qdes_euler = _quat_to_euler_zyx_deg(qdes_w)

    wp_mask = (wp_t >= start) & (wp_t <= stop)
    wp_pts = np.unique(np.round(wp[wp_mask], 3), axis=0) if wp_mask.any() else np.empty((0, 3))

    fig = plt.figure(figsize=(7, 13))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1.3, 2.2], hspace=0.35)

    colors = ["red", "green", "blue"]  # surge/roll, sway/pitch, heave/yaw
    vel_labels = ["u", "v", "w"]
    vel_axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0], sharex=vel_axes[0] if vel_axes else None)
        ax.plot(t, vdes_w[:, i], ":", color=colors[i], label=f"{vel_labels[i]}$_d$")
        ax.plot(t, vact_w[:, i], "-", color=colors[i], label=vel_labels[i])
        ax.set_ylabel("Velocities [m/s]" if i == 0 else f"{vel_labels[i]} [m/s]")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)
        vel_axes.append(ax)

    ax_att = fig.add_subplot(gs[3, 0], sharex=vel_axes[0])
    att_labels = ["$\\phi$(roll)", "$\\theta$(pitch)", "$\\psi$(yaw)"]
    for i in range(3):
        ax_att.plot(t, qdes_euler[:, i], ":", color=colors[i], label=f"{att_labels[i]}$_d$")
        ax_att.plot(t, euler[:, i], "-", color=colors[i], label=att_labels[i])
    ax_att.set_ylabel("Attitude [deg]")
    ax_att.set_xlabel("Time [s]")
    ax_att.legend(loc="upper right", fontsize=7, ncol=2)
    ax_att.grid(alpha=0.3)

    # NED: x=North, y=East, z=Down (z inverted below so "up" reads intuitively)
    ax3d = fig.add_subplot(gs[4, 0], projection="3d")
    ax3d.plot(pos_w[:, 0], pos_w[:, 1], -pos_w[:, 2], color="tab:blue", linewidth=1.5, label="Position")
    if wp_pts.size:
        ax3d.scatter(wp_pts[:, 0], wp_pts[:, 1], -wp_pts[:, 2], color="orange", s=25, label="Waypoints")

    n_arrows = min(8, len(pos_w))
    if n_arrows > 0:
        idxs = np.linspace(0, len(pos_w) - 1, n_arrows).astype(int)
        fwd = _forward_dir_from_quat(quat_w[idxs])
        ax3d.quiver(
            pos_w[idxs, 0], pos_w[idxs, 1], -pos_w[idxs, 2],
            fwd[:, 0], fwd[:, 1], -fwd[:, 2],
            length=_FORWARD_ARROW_LEN, color="purple", label="Forward direction",
        )

    ax3d.scatter(pos_w[0, 0], pos_w[0, 1], -pos_w[0, 2], color="green", marker="X", s=90, label="start")
    ax3d.scatter(pos_w[-1, 0], pos_w[-1, 1], -pos_w[-1, 2], color="red", marker="X", s=90, label="end")
    ax3d.set_xlabel("North [m]")
    ax3d.set_ylabel("East [m]")
    ax3d.set_zlabel("Up [m] (-Down)")
    ax3d.xaxis.set_major_locator(MultipleLocator(1))
    ax3d.yaxis.set_major_locator(MultipleLocator(1))
    ax3d.zaxis.set_major_locator(MultipleLocator(1))
    ax3d.legend(loc="upper left", fontsize=7)

    fig.suptitle(args.title or os.path.basename(os.path.normpath(args.bag)))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=120)
    plt.close(fig)
    print(f"[INFO] plot saved: {args.output}")

    vel_err = np.linalg.norm(vact_w - vdes_w, axis=-1)
    tail_mask = t > (t[-1] - 2.0)
    print(f"  mean velocity error norm      : {vel_err.mean():.4f} m/s")
    print(f"  last-2s mean velocity error   : {vel_err[tail_mask].mean():.4f} m/s (steady-state proxy)")


if __name__ == "__main__":
    main()

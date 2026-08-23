#!/usr/bin/env python3
"""Compare deployed observation streams against the as-trained distribution.

For each of the 16 BROVVelEnv observation dimensions (q_e[4], v_e_b[3],
omega_b[3], z_v[3], z_q[3]), this reports how far a deployed run's
``/brov/observation`` samples fall outside the training-time distribution
collected by ``collect_obs_distribution.py`` (deploy_v2, no guidance
attached -> the training command scheduler, same policy checkpoint).

Real-vehicle data is the priority reference; Gazebo bags are secondary.
All rosbag access is read-only.
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
import analyze_stage2_case_a_ab as stage2

_LABELS = (
    [f"q_e[{i}]" for i in range(4)]
    + [f"v_e_b[{i}]" for i in range(3)]
    + [f"omega_b[{i}]" for i in range(3)]
    + [f"z_v[{i}]" for i in range(3)]
    + [f"z_q[{i}]" for i in range(3)]
)
_GROUPS = {
    "q_e": list(range(0, 4)),
    "v_e_b": list(range(4, 7)),
    "omega_b": list(range(7, 10)),
    "z_v": list(range(10, 13)),
    "z_q": list(range(13, 16)),
}


def _load_observations(bag_uri: str) -> tuple[np.ndarray, np.ndarray]:
    bag = stage2.read_bag(bag_uri)
    active_runs = stage2._active_runs(bag["/brov/control_active"])
    if not active_runs:
        raise RuntimeError(f"{bag_uri}: no active control interval")
    start, stop = max(active_runs, key=lambda pair: pair[1] - pair[0])
    t, obs = stage1.arrays(bag["/brov/observation"])
    mask = (t >= start) & (t <= stop)
    return t[mask], obs[mask]


def _deviation_report(train_stats: dict, obs: np.ndarray, *, low_p: str, high_p: str) -> dict:
    per_dim = {}
    for idx, label in enumerate(_LABELS):
        pct = train_stats["stats"][label]["percentiles"]
        lo, hi = pct[low_p], pct[high_p]
        col = obs[:, idx]
        below = col < lo
        above = col > hi
        outside = below | above
        excess = np.where(below, lo - col, np.where(above, col - hi, 0.0))
        std = train_stats["stats"][label]["std"] or 1e-9
        per_dim[label] = {
            "train_band": [lo, hi],
            "deployed_min": float(col.min()),
            "deployed_max": float(col.max()),
            "outside_band_fraction": float(outside.mean()),
            "max_excess_abs": float(excess.max()),
            "max_excess_in_train_std": float(excess.max() / std),
            "mean_excess_when_outside": float(excess[outside].mean()) if outside.any() else 0.0,
        }
    group_summary = {
        group: {
            "mean_outside_band_fraction": float(
                np.mean([per_dim[_LABELS[i]]["outside_band_fraction"] for i in idxs])
            ),
            "max_excess_in_train_std": float(
                max(per_dim[_LABELS[i]]["max_excess_in_train_std"] for i in idxs)
            ),
        }
        for group, idxs in _GROUPS.items()
    }
    return {"per_dim": per_dim, "group_summary": group_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-stats", required=True)
    parser.add_argument("--real-bag", required=True, help="real-vehicle Case-A bag (priority reference)")
    parser.add_argument("--gazebo-bags", nargs="*", default=[], help="optional secondary Gazebo bags")
    parser.add_argument("--gazebo-labels", nargs="*", default=[])
    parser.add_argument("--low-percentile", default="0.5")
    parser.add_argument("--high-percentile", default="99.5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.train_stats, encoding="utf-8") as stream:
        train_stats = json.load(stream)

    real_t, real_obs = _load_observations(args.real_bag)
    real_report = _deviation_report(
        train_stats, real_obs, low_p=args.low_percentile, high_p=args.high_percentile
    )
    real_report["bag"] = args.real_bag
    real_report["sample_count"] = int(real_obs.shape[0])

    gazebo_reports = {}
    for label, bag_uri in zip(args.gazebo_labels, args.gazebo_bags):
        t, obs = _load_observations(bag_uri)
        report = _deviation_report(
            train_stats, obs, low_p=args.low_percentile, high_p=args.high_percentile
        )
        report["bag"] = bag_uri
        report["sample_count"] = int(obs.shape[0])
        gazebo_reports[label] = report

    output = {
        "schema": "brov_obs_distribution_gap_v1",
        "train_stats_source": args.train_stats,
        "band": [args.low_percentile, args.high_percentile],
        "priority_reference": "real_vehicle",
        "real_vehicle": real_report,
        "gazebo_secondary": gazebo_reports,
    }
    encoded = json.dumps(stage1.json_safe(output), indent=2, allow_nan=True)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")

    print("=== REAL VEHICLE (priority reference) ===")
    for group, summary in real_report["group_summary"].items():
        print(
            f"  {group:10s} outside-band frac={summary['mean_outside_band_fraction']:.3f}"
            f"  max excess={summary['max_excess_in_train_std']:.1f} train-std"
        )
    for label, report in gazebo_reports.items():
        print(f"=== GAZEBO ({label}), secondary ===")
        for group, summary in report["group_summary"].items():
            print(
                f"  {group:10s} outside-band frac={summary['mean_outside_band_fraction']:.3f}"
                f"  max excess={summary['max_excess_in_train_std']:.1f} train-std"
            )


if __name__ == "__main__":
    main()

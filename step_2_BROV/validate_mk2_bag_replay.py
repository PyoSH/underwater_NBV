#!/usr/bin/env python3
"""Replay MK2 rosbag observations through TorchScript without changing state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from std_msgs.msg import Float32MultiArray


def _read(uri: str) -> dict[str, list[tuple[float, np.ndarray]]]:
    wanted = {
        "/brov/observation",
        "/brov/action",
        "/brov/policy/action_raw",
    }
    result = {name: [] for name in wanted}
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=uri, storage_id="sqlite3"),
        ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic not in wanted:
            continue
        message = deserialize_message(payload, Float32MultiArray)
        result[topic].append(
            (stamp_ns * 1.0e-9, np.asarray(message.data, dtype=np.float32))
        )
    return result


def _arrays(series):
    return (
        np.asarray([item[0] for item in series], dtype=np.float64),
        np.stack([item[1] for item in series]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("policy")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    series = _read(args.bag)
    observation_t, observation = _arrays(series["/brov/observation"])
    action_t, action = _arrays(series["/brov/action"])
    raw_t, runtime_raw = _arrays(series["/brov/policy/action_raw"])
    if observation.shape[1:] != (16,) or action.shape[1:] != (6,):
        raise RuntimeError("unexpected observation/action dimensions")

    model = torch.jit.load(args.policy, map_location="cpu")
    model.eval()
    with torch.inference_mode():
        actor_raw = model(torch.from_numpy(observation)).cpu().numpy()
    actor_clipped = np.clip(actor_raw, -1.0, 1.0)
    if not (
        observation.shape[0] == action.shape[0] == runtime_raw.shape[0]
    ):
        raise RuntimeError(
            "observation/action topic counts differ; callback-order replay is invalid"
        )
    # PolicyNode publishes exactly one action pair for every observation
    # callback. Recorder receive timestamps can reorder adjacent DDS topics,
    # so sequence index is the causal join and timestamps are diagnostics only.
    paired_action = action
    paired_runtime_raw = runtime_raw
    action_skew = action_t - observation_t
    raw_skew = raw_t - observation_t

    action_error = paired_action - actor_clipped
    runtime_raw_error = paired_runtime_raw - actor_clipped
    result = {
        "schema": "brov_mk2_bag_torchscript_replay_v1",
        "bag": str(Path(args.bag).resolve()),
        "policy": str(Path(args.policy).resolve()),
        "samples": int(observation.shape[0]),
        "action_samples": int(action.shape[0]),
        "policy_action_raw_topic_samples": int(runtime_raw.shape[0]),
        "actor_raw_min_per_axis": actor_raw.min(axis=0).tolist(),
        "actor_raw_max_per_axis": actor_raw.max(axis=0).tolist(),
        "actor_raw_any_outside_unit_fraction": float(
            np.mean(np.any(np.abs(actor_raw) > 1.0, axis=1))
        ),
        "clipped_actor_vs_action": {
            "max_abs": float(np.max(np.abs(action_error))),
            "rms": float(np.sqrt(np.mean(action_error**2))),
            "pairing_skew_p95_ms": float(
                np.percentile(np.abs(action_skew), 95) * 1000.0
            ),
        },
        "clipped_actor_vs_policy_action_raw_topic": {
            "max_abs": float(np.max(np.abs(runtime_raw_error))),
            "rms": float(np.sqrt(np.mean(runtime_raw_error**2))),
            "pairing_skew_p95_ms": float(
                np.percentile(np.abs(raw_skew), 95) * 1000.0
            ),
        },
        "pass": bool(
            np.max(np.abs(action_error)) <= 1.0e-6
            and np.max(np.abs(runtime_raw_error)) <= 1.0e-6
        ),
        "semantic_note": (
            "PolicyRunner clips TorchScript output before both ROS action topics; "
            "policy/action_raw is therefore runtime-clipped, not unbounded actor raw."
        ),
    }
    encoded = json.dumps(result, indent=2)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

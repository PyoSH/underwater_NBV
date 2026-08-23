#!/usr/bin/env python3
"""Render an MP4 from the ``/brov/sim/observer_camera`` topic in a Case-A rosbag.

Requires the fixed ``observer_camera`` model in
``stage2_bluerov2_heavy_underwater_8p5m.sdf`` and the camera bridge/bag-record
wiring added to ``run_mk2_case_a_deploy.sh`` / ``run_case_a_deploy_model_based.sh``.
Older bags recorded before that wiring existed will not have this topic.

All rosbag access is read-only.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

_ENCODING_TO_CV2 = {
    "rgb8": cv2.COLOR_RGB2BGR,
    "rgba8": cv2.COLOR_RGBA2BGR,
    "bgr8": None,
    "bgra8": cv2.COLOR_BGRA2BGR,
    "mono8": cv2.COLOR_GRAY2BGR,
}


def _read_frames(uri: str, topic: str) -> list[tuple[float, np.ndarray, str]]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=uri, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    frames: list[tuple[float, np.ndarray, str]] = []
    while reader.has_next():
        got_topic, payload, stamp_ns = reader.read_next()
        if got_topic != topic:
            continue
        msg = deserialize_message(payload, Image)
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(msg.encoding)
        if channels is None:
            raise ValueError(f"unsupported image encoding: {msg.encoding!r}")
        arr = arr.reshape(msg.height, msg.width, channels)
        frames.append((stamp_ns * 1.0e-9, arr, msg.encoding))
    return frames


def _read_active_window(uri: str, topic: str = "/brov/control_active") -> tuple[float, float] | None:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=uri, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    runs: list[tuple[float, float]] = []
    start: float | None = None
    last = None
    while reader.has_next():
        got_topic, payload, stamp_ns = reader.read_next()
        if got_topic != topic:
            continue
        stamp = stamp_ns * 1.0e-9
        active = bool(deserialize_message(payload, Bool).data)
        if active and start is None:
            start = stamp
        elif not active and start is not None:
            runs.append((start, stamp))
            start = None
        last = stamp
    if start is not None and last is not None:
        runs.append((start, last))
    if not runs:
        return None
    return max(runs, key=lambda pair: pair[1] - pair[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--output", required=True, help="output MP4 path")
    parser.add_argument("--topic", default="/brov/sim/observer_camera")
    parser.add_argument(
        "--pad-s", type=float, default=2.0,
        help="seconds of context to keep before/after the active control window",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="render every recorded frame instead of trimming to the active window",
    )
    parser.add_argument("--fps", type=float, default=None, help="override output fps")
    args = parser.parse_args()

    frames = _read_frames(args.bag, args.topic)
    if not frames:
        raise RuntimeError(f"{args.bag}: no messages on {args.topic}")

    if not args.full:
        window = _read_active_window(args.bag)
        if window is not None:
            start, stop = window
            start -= args.pad_s
            stop += args.pad_s
            frames = [f for f in frames if start <= f[0] <= stop]
    if not frames:
        raise RuntimeError(f"{args.bag}: no frames left after windowing")

    frames.sort(key=lambda f: f[0])
    timestamps = np.asarray([f[0] for f in frames])
    dt = np.diff(timestamps)
    fps = args.fps or (1.0 / np.median(dt) if dt.size else 10.0)

    height, width = frames[0][1].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {args.output}")
    for _, frame, encoding in frames:
        conversion = _ENCODING_TO_CV2[encoding]
        bgr = frame if conversion is None else cv2.cvtColor(frame, conversion)
        writer.write(bgr)
    writer.release()

    duration_s = timestamps[-1] - timestamps[0]
    print(
        f"[INFO] video saved: {args.output} "
        f"({len(frames)} frames, {fps:.2f} fps, {duration_s:.1f}s span)"
    )


if __name__ == "__main__":
    main()

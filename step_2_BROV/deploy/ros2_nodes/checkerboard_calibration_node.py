#!/usr/bin/env python3
"""GUI 없이 checkerboard 표본을 자동 수집해 monocular intrinsic을 계산한다."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CheckerboardCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("brov_checkerboard_calibration")
        self.declare_parameter("columns", 8)  # 내부 코너 수
        self.declare_parameter("rows", 6)
        self.declare_parameter("square_size_m", 0.030)
        self.declare_parameter("target_samples", 30)
        self.declare_parameter("min_interval_s", 0.5)
        self.declare_parameter("min_descriptor_distance", 0.06)
        self.declare_parameter("output_path", "/workspace/deploy/config/camera_intrinsics.yaml")

        self._cols = int(self.get_parameter("columns").value)
        self._rows = int(self.get_parameter("rows").value)
        self._square = float(self.get_parameter("square_size_m").value)
        self._target = int(self.get_parameter("target_samples").value)
        self._interval = float(self.get_parameter("min_interval_s").value)
        self._distance = float(self.get_parameter("min_descriptor_distance").value)
        self._path = Path(str(self.get_parameter("output_path").value))
        self._bridge = CvBridge()
        self._object_points: list[np.ndarray] = []
        self._image_points: list[np.ndarray] = []
        self._descriptors: list[np.ndarray] = []
        self._last_sample_t = 0.0
        self._image_size = None
        self._done = False

        grid = np.zeros((self._rows * self._cols, 3), np.float32)
        grid[:, :2] = np.mgrid[0:self._cols, 0:self._rows].T.reshape(-1, 2)
        self._grid = grid * self._square

        self.pub_debug = self.create_publisher(
            Image, "/brov/camera/calibration_debug", qos_profile_sensor_data
        )
        self.create_subscription(
            Image, "/brov/camera/image_raw", self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"checkerboard {self._cols}x{self._rows}, square={self._square:.4f} m, "
            f"목표 {self._target}장 — 보드를 화면 중앙/모서리/기울기에 고르게 이동할 것"
        )

    def _descriptor(self, corners: np.ndarray, width: int, height: int) -> np.ndarray:
        pts = corners.reshape(-1, 2)
        center = pts.mean(axis=0) / np.array([width, height])
        span = (pts.max(axis=0) - pts.min(axis=0)) / np.array([width, height])
        edge = pts[self._cols - 1] - pts[0]
        angle = np.arctan2(edge[1], edge[0]) / np.pi
        return np.array([center[0], center[1], span[0], span[1], angle])

    def _save(self, width: int, height: int, k: np.ndarray, d: np.ndarray) -> None:
        p = np.zeros((3, 4), dtype=float)
        p[:, :3] = k
        data = {
            "image_width": width,
            "image_height": height,
            "camera_name": "brov_camera",
            "camera_matrix": {"rows": 3, "cols": 3, "data": k.reshape(-1).tolist()},
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {
                "rows": 1, "cols": int(d.size), "data": d.reshape(-1).tolist()
            },
            "rectification_matrix": {
                "rows": 3, "cols": 3, "data": np.eye(3).reshape(-1).tolist()
            },
            "projection_matrix": {"rows": 3, "cols": 4, "data": p.reshape(-1).tolist()},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)

    def _calibrate(self) -> None:
        rms, k, d, _, _ = cv2.calibrateCamera(
            self._object_points, self._image_points, self._image_size, None, None
        )
        width, height = self._image_size
        self._save(width, height, k, d)
        self._done = True
        self.get_logger().info(
            f"보정 완료: RMS reprojection error={rms:.4f}, 저장={self._path}. "
            "camera_stream_node를 재시작해 새 intrinsic을 로드할 것"
        )

    def _on_image(self, msg: Image) -> None:
        if self._done:
            return
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width = frame.shape[:2]
        if self._image_size is None:
            self._image_size = (width, height)
        elif self._image_size != (width, height):
            self.get_logger().error("수집 중 영상 해상도가 변경됨 — 노드를 재시작할 것")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(
            gray, (self._cols, self._rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        accepted = False
        if found:
            descriptor = self._descriptor(corners, width, height)
            diverse = not self._descriptors or min(
                np.linalg.norm(descriptor - old) for old in self._descriptors
            ) >= self._distance
            now = time.monotonic()
            if diverse and now - self._last_sample_t >= self._interval:
                self._object_points.append(self._grid.copy())
                self._image_points.append(corners.astype(np.float32))
                self._descriptors.append(descriptor)
                self._last_sample_t = now
                accepted = True
                self.get_logger().info(f"표본 {len(self._image_points)}/{self._target} 수집")
            cv2.drawChessboardCorners(frame, (self._cols, self._rows), corners, found)

        color = (0, 255, 0) if accepted else (0, 180, 255)
        cv2.putText(
            frame, f"samples {len(self._image_points)}/{self._target}", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
        debug = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug.header = msg.header
        self.pub_debug.publish(debug)

        if len(self._image_points) >= self._target:
            self._calibrate()


def main() -> None:
    rclpy.init()
    node = CheckerboardCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

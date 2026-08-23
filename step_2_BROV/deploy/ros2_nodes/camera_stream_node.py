#!/usr/bin/env python3
"""BlueOS RTP/H264 UDP stream을 ROS 2 Image/CameraInfo로 발행한다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import rclpy
import numpy as np
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.srv import SetCameraInfo


def _matrix(values: dict, expected: int, default: list[float]) -> list[float]:
    data = values.get("data", default) if isinstance(values, dict) else default
    return [float(v) for v in data] if len(data) == expected else default


class CameraStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("brov_camera_node")
        self.declare_parameter("udp_port", 5600)
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("camera_info_path", "/workspace/deploy/config/camera_intrinsics.yaml")
        self.declare_parameter("latency_ms", 200)

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._info_path = Path(str(self.get_parameter("camera_info_path").value))
        self._info = self._load_camera_info(self._info_path)
        self._bridge = CvBridge()
        self._frame_count = 0
        self._last_diag_count = 0
        self._last_diag_ns = self.get_clock().now().nanoseconds

        self.pub_image = self.create_publisher(Image, "/brov/camera/image_raw", qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, "/brov/camera/camera_info", qos_profile_sensor_data)
        self.srv_info = self.create_service(
            SetCameraInfo, "/brov/camera/set_camera_info", self._set_camera_info
        )

        Gst.init(None)
        port = int(self.get_parameter("udp_port").value)
        latency = int(self.get_parameter("latency_ms").value)
        pipeline = (
            f"udpsrc name=source port={port} buffer-size=2097152 "
            "caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264\" "
            f"! rtpjitterbuffer name=jitter latency={latency} "
            "drop-on-latency=false do-lost=true "
            "! rtph264depay ! h264parse ! avdec_h264 "
            "! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream "
            "! videoconvert ! video/x-raw,format=BGR "
            "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        self._pipeline = Gst.parse_launch(pipeline)
        self._bus = self._pipeline.get_bus()
        self._jitter = self._pipeline.get_by_name("jitter")
        sink = self._pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_sample)
        self._pipeline.set_state(Gst.State.PLAYING)
        self.create_timer(0.2, self._poll_gstreamer_bus)
        self.create_timer(5.0, self._report_stream_stats)
        self.get_logger().info(
            f"BlueOS H264 수신 대기: udp://0.0.0.0:{port} → /brov/camera/image_raw"
        )

    def _report_stream_stats(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        elapsed = (now_ns - self._last_diag_ns) / 1e9
        frames = self._frame_count - self._last_diag_count
        fps = frames / elapsed if elapsed > 0.0 else 0.0
        self._last_diag_ns = now_ns
        self._last_diag_count = self._frame_count

        stats = self._jitter.get_property("stats")
        values = {}
        if stats is not None:
            for key in ("num-pushed", "num-lost", "num-late", "num-duplicates"):
                try:
                    values[key] = stats.get_value(key)
                except Exception:
                    values[key] = "?"
        self.get_logger().info(
            f"camera decode={fps:.1f} fps, RTP pushed={values.get('num-pushed', '?')}, "
            f"lost={values.get('num-lost', '?')}, late={values.get('num-late', '?')}, "
            f"duplicates={values.get('num-duplicates', '?')}"
        )

    def _poll_gstreamer_bus(self) -> None:
        message = self._bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if message is None:
            return
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            self.get_logger().error(f"GStreamer 오류: {error}; {debug or 'no details'}")
        else:
            self.get_logger().warn("GStreamer stream 종료(EOS)")

    def _load_camera_info(self, path: Path) -> CameraInfo:
        msg = CameraInfo()
        msg.header.frame_id = self._frame_id
        if not path.is_file():
            self.get_logger().warn(f"camera calibration 파일 없음: {path}")
            return msg
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        msg.width = int(data.get("image_width", 0))
        msg.height = int(data.get("image_height", 0))
        msg.distortion_model = str(data.get("distortion_model", "plumb_bob"))
        msg.d = _matrix(data.get("distortion_coefficients", {}), 5, [0.0] * 5)
        msg.k = _matrix(data.get("camera_matrix", {}), 9, [0.0] * 9)
        msg.r = _matrix(
            data.get("rectification_matrix", {}), 9,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        msg.p = _matrix(data.get("projection_matrix", {}), 12, [0.0] * 12)
        return msg

    def _save_camera_info(self, msg: CameraInfo) -> None:
        data = {
            "image_width": int(msg.width),
            "image_height": int(msg.height),
            "camera_name": "brov_camera",
            "camera_matrix": {"rows": 3, "cols": 3, "data": list(msg.k)},
            "distortion_model": msg.distortion_model,
            "distortion_coefficients": {"rows": 1, "cols": len(msg.d), "data": list(msg.d)},
            "rectification_matrix": {"rows": 3, "cols": 3, "data": list(msg.r)},
            "projection_matrix": {"rows": 3, "cols": 4, "data": list(msg.p)},
        }
        self._info_path.parent.mkdir(parents=True, exist_ok=True)
        with self._info_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)

    def _set_camera_info(self, request, response):
        try:
            self._info = request.camera_info
            self._info.header.frame_id = self._frame_id
            self._save_camera_info(self._info)
            response.success = True
            response.status_message = f"saved: {self._info_path}"
            self.get_logger().info(response.status_message)
        except Exception as exc:
            response.success = False
            response.status_message = str(exc)
            self.get_logger().error(f"camera info 저장 실패: {exc}")
        return response

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            # sensor_msgs/Image.data에 bytes를 직접 대입하면 rclpy가 각 byte를
            # Python sequence로 변환해 매우 느리다. mapped buffer의 NumPy view를
            # cv_bridge의 C 복사 경로로 즉시 메시지화한다.
            frame = np.frombuffer(mapped.data, dtype=np.uint8).reshape((height, width, 3))
            image = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        finally:
            buffer.unmap(mapped)

        stamp = self.get_clock().now().to_msg()
        image.header.stamp = stamp
        image.header.frame_id = self._frame_id
        info = CameraInfo()
        info.header = image.header
        info.width, info.height = width, height
        info.distortion_model = self._info.distortion_model
        info.d, info.k, info.r, info.p = self._info.d, self._info.k, self._info.r, self._info.p
        self.pub_image.publish(image)
        self.pub_info.publish(info)
        self._frame_count += 1
        return Gst.FlowReturn.OK

    def destroy_node(self):
        self._pipeline.set_state(Gst.State.NULL)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CameraStreamNode()
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

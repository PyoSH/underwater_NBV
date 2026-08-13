#!/usr/bin/env python3
"""보정된 카메라 영상에서 ArUco marker 및 marker 기준 로봇 pose를 계산한다."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


def _quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → ROS quaternion [x,y,z,w]."""
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
            0.25 * s,
        ])
    i = int(np.argmax(np.diag(r)))
    if i == 0:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        return np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s,
                         (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s])
    if i == 1:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        return np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s,
                         (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s])
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
    return np.array([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s,
                     0.25 * s, (r[1, 0] - r[0, 1]) / s])


def _matrix_from_xyz_rpy(xyz: list[float], rpy: list[float]) -> np.ndarray:
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    t = np.eye(4)
    t[:3, :3] = rz @ ry @ rx
    t[:3, 3] = xyz
    return t


def _fill_pose(msg: PoseStamped, transform: np.ndarray) -> None:
    q = _quat_from_matrix(transform[:3, :3])
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = transform[:3, 3]
    msg.pose.orientation.x, msg.pose.orientation.y = float(q[0]), float(q[1])
    msg.pose.orientation.z, msg.pose.orientation.w = float(q[2]), float(q[3])


class ArucoPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("brov_aruco_pose_node")
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_id", 0)
        self.declare_parameter("marker_length_m", 0.15)
        self.declare_parameter("marker_frame", "aruco_reference")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_robot_pose", False)
        self.declare_parameter("base_to_camera_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("base_to_camera_rpy", [0.0, 0.0, 0.0])

        dictionary_name = str(self.get_parameter("dictionary").value)
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"지원하지 않는 ArUco dictionary: {dictionary_name}")
        self._dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        # rclpy.Node가 내부적으로 사용하는 ``_parameters`` 이름은 덮어쓰지 않는다.
        self._detector_parameters = cv2.aruco.DetectorParameters_create()
        self._marker_id = int(self.get_parameter("marker_id").value)
        self._marker_length = float(self.get_parameter("marker_length_m").value)
        self._marker_frame = str(self.get_parameter("marker_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._publish_robot = bool(self.get_parameter("publish_robot_pose").value)
        xyz = [float(v) for v in self.get_parameter("base_to_camera_xyz").value]
        rpy = [float(v) for v in self.get_parameter("base_to_camera_rpy").value]
        self._t_base_camera = _matrix_from_xyz_rpy(xyz, rpy)

        self._bridge = CvBridge()
        self._camera_info = None
        self._warned_uncalibrated = False
        self._tf = TransformBroadcaster(self)
        self.pub_marker = self.create_publisher(PoseStamped, "/brov/aruco/marker_pose", 10)
        self.pub_robot = self.create_publisher(PoseStamped, "/brov/aruco/robot_pose", 10)
        self.pub_visible = self.create_publisher(Bool, "/brov/aruco/visible", 10)
        self.pub_debug = self.create_publisher(Image, "/brov/aruco/debug_image", qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, "/brov/camera/camera_info", self._on_info, qos_profile_sensor_data
        )
        self.create_subscription(Image, "/brov/camera/image_raw", self._on_image, qos_profile_sensor_data)
        self.get_logger().info(
            f"ArUco 대기: {dictionary_name}, id={self._marker_id}, length={self._marker_length:.3f} m"
        )

    def _on_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _publish_tf(self, parent: str, child: str, pose: PoseStamped) -> None:
        tf = TransformStamped()
        tf.header = pose.header
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = pose.pose.position.x
        tf.transform.translation.y = pose.pose.position.y
        tf.transform.translation.z = pose.pose.position.z
        tf.transform.rotation = pose.pose.orientation
        self._tf.sendTransform(tf)

    def _on_image(self, msg: Image) -> None:
        if self._camera_info is None or self._camera_info.k[0] <= 0.0:
            if not self._warned_uncalibrated:
                self._warned_uncalibrated = True
                self.get_logger().warn("유효한 camera intrinsic이 없어 ArUco metric pose를 계산하지 않음")
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self._dictionary, parameters=self._detector_parameters
        )
        visible = ids is not None and self._marker_id in ids.flatten().tolist()
        self.pub_visible.publish(Bool(data=visible))
        if not visible:
            return

        index = ids.flatten().tolist().index(self._marker_id)
        selected = [corners[index]]
        k = np.asarray(self._camera_info.k, dtype=np.float64).reshape(3, 3)
        d = np.asarray(self._camera_info.d, dtype=np.float64)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            selected, self._marker_length, k, d
        )
        r_camera_marker, _ = cv2.Rodrigues(rvecs[0, 0])
        t_camera_marker = np.eye(4)
        t_camera_marker[:3, :3] = r_camera_marker
        t_camera_marker[:3, 3] = tvecs[0, 0]

        marker_pose = PoseStamped()
        marker_pose.header = msg.header
        _fill_pose(marker_pose, t_camera_marker)
        self.pub_marker.publish(marker_pose)
        self._publish_tf(msg.header.frame_id, self._marker_frame, marker_pose)

        if self._publish_robot:
            # T_M_B = inv(T_C_M) @ inv(T_B_C)
            t_marker_base = np.linalg.inv(t_camera_marker) @ np.linalg.inv(self._t_base_camera)
            robot_pose = PoseStamped()
            robot_pose.header.stamp = msg.header.stamp
            robot_pose.header.frame_id = self._marker_frame
            _fill_pose(robot_pose, t_marker_base)
            self.pub_robot.publish(robot_pose)
            self._publish_tf(self._marker_frame, self._base_frame, robot_pose)

        cv2.aruco.drawDetectedMarkers(frame, selected, np.array([[self._marker_id]]))
        cv2.drawFrameAxes(frame, k, d, rvecs[0, 0], tvecs[0, 0], self._marker_length * 0.5)
        debug = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug.header = msg.header
        self.pub_debug.publish(debug)


def main() -> None:
    rclpy.init()
    node = ArucoPoseNode()
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

"""Gazebo rgbd_camera -> brov_ros2 카메라 계약 + 정책 관측. 2026-09-10.

세 가지를 한다.

1. **camera_info 를 우리가 낸다.** gz 의 `rgbd_camera` 는 이미지를 SDF 대로 640x480 으로
   렌더하면서도 CameraInfo 의 intrinsics 는 기본값(fx=277, cx=160, cy=120 — 320x240 기준)
   으로 남긴다(2026-09-10 실측). `aruco_pose_node` 가 이걸로 pose 를 풀면 크게 틀어진다.
   렌더의 실효 fx 는 태그로 교정해 **465.518** 임을 확인했다(1.0~4.0 m 5점, 잔차는 거리와
   무관한 고정 -0.626 px 코너 편향이고 비율 오차가 아니다).

2. **수중 감쇠를 입힌다.** Gazebo 는 수중 광학을 모사하지 않는다. depth 가 있으므로
   Isaac 과 같은 식을 픽셀별로 적용해 이미지 도메인 갭을 닫는다(uw_render.py).
   이 결과가 "카메라가 실제로 보는 것"이므로 **ArUco 도 이 이미지를 쓴다** — 감쇠로
   줄어든 대비가 검출률에 반영돼야 측정이 정직하다.

3. **정책 관측을 만든다.** 실기 FOV(69.0x54.6) -> sim FOV(47.2x36.3) 중앙 crop 407x305
   -> 84x84 그레이스케일. crop 은 정책 입력에만 적용하고 ArUco 는 원본 640x480 을 쓴다
   (가시 범위를 좁힐 이유가 없다).
"""
from __future__ import annotations

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from uw_render import uw_render_bgr

FX_EFFECTIVE = 465.5181034880913     # 태그 교정으로 확인한 렌더 실효 초점거리

# cv_bridge 를 쓰지 않는다: 이 컨테이너의 cv_bridge 는 OpenCV 4 기준으로 빌드됐는데
# 설치된 cv2 는 5.0 이라 `cv2_to_imgmsg` 가 KeyError: 16 (CV_8UC3) 로 죽는다
# (2026-09-10 실측). 변환은 몇 줄이라 직접 하는 편이 의존성보다 싸고 확실하다.
_ENC = {"bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3),
        "mono8": (np.uint8, 1), "32FC1": (np.float32, 1)}


def to_imgmsg(arr: np.ndarray, encoding: str, header) -> Image:
    dtype, ch = _ENC[encoding]
    a = np.ascontiguousarray(arr, dtype=dtype)
    msg = Image()
    msg.header = header
    msg.height, msg.width = int(a.shape[0]), int(a.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(a.shape[1] * ch * a.dtype.itemsize)
    msg.data = a.tobytes()
    return msg


def from_imgmsg(msg: Image) -> np.ndarray:
    if msg.encoding not in _ENC:
        raise ValueError(f"지원하지 않는 인코딩 {msg.encoding!r}")
    dtype, ch = _ENC[msg.encoding]
    a = np.frombuffer(msg.data, dtype=dtype)
    a = a.reshape(msg.height, msg.step // (dtype().itemsize))
    a = a[:, : msg.width * ch]
    return a.reshape(msg.height, msg.width, ch) if ch > 1 else a.reshape(msg.height, msg.width)


class NbvCameraPipeline(Node):
    def __init__(self) -> None:
        super().__init__("nbv_camera_pipeline")
        self.declare_parameter("rgb_topic", "/brov/sim/camera_rgb")
        self.declare_parameter("depth_topic", "/brov/sim/camera_depth")
        self.declare_parameter("image_topic", "/brov/camera/image_raw")
        self.declare_parameter("info_topic", "/brov/camera/camera_info")
        self.declare_parameter("obs_topic", "/brov/nbv/image_obs")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("water", "IB")
        self.declare_parameter("obs_size", 84)
        self.declare_parameter("max_depth_skew_s", 0.05)

        g = lambda n: self.get_parameter(n).value
        self._frame_id = str(g("frame_id"))
        self._water = str(g("water"))
        self._obs = int(g("obs_size"))
        self._skew = float(g("max_depth_skew_s"))
        self._depth = None
        self._depth_t = None
        self._dropped = 0

        self.pub_img = self.create_publisher(Image, str(g("image_topic")),
                                             qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, str(g("info_topic")),
                                              qos_profile_sensor_data)
        self.pub_obs = self.create_publisher(Image, str(g("obs_topic")),
                                             qos_profile_sensor_data)
        self.create_subscription(Image, str(g("depth_topic")), self._on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, str(g("rgb_topic")), self._on_rgb,
                                 qos_profile_sensor_data)
        self.create_timer(5.0, self._report)
        self.get_logger().info(
            f"water={self._water} fx={FX_EFFECTIVE:.3f} obs={self._obs}x{self._obs}")

    @staticmethod
    def _stamp_s(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_depth(self, msg: Image) -> None:
        self._depth = from_imgmsg(msg)
        self._depth_t = self._stamp_s(msg)

    def _camera_info(self, stamp, w: int, h: int) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._frame_id
        info.width, info.height = w, h
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5                     # gz 렌더에는 왜곡이 없다
        cx, cy = w / 2.0, h / 2.0              # gz 는 주점이 정확히 중앙
        info.k = [FX_EFFECTIVE, 0.0, cx, 0.0, FX_EFFECTIVE, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [FX_EFFECTIVE, 0.0, cx, 0.0, 0.0, FX_EFFECTIVE, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def _on_rgb(self, msg: Image) -> None:
        bgr = from_imgmsg(msg)
        if msg.encoding == "rgb8":
            bgr = bgr[:, :, ::-1]
        h, w = bgr.shape[:2]

        if self._water != "none":
            if self._depth is None or self._depth.shape[:2] != (h, w):
                self._dropped += 1
                return
            if self._depth_t is not None \
                    and abs(self._stamp_s(msg) - self._depth_t) > self._skew:
                # depth 와 RGB 가 다른 시각이면 감쇠가 엉뚱한 거리로 걸린다.
                # 조용히 섞느니 버린다 — 조용한 손상이 크래시보다 나쁘다(§3 렌더 교훈).
                self._dropped += 1
                return
            bgr = uw_render_bgr(bgr, self._depth, self._water)

        header = msg.header
        header.frame_id = self._frame_id
        out = to_imgmsg(bgr, "bgr8", header)
        self.pub_img.publish(out)
        self.pub_info.publish(self._camera_info(msg.header.stamp, w, h))

        # 정책 입력은 전체 화면 — 학습 카메라가 실기와 같은 화각(69.0x54.6)이 됐다
        # (정본 §15.5 (나)). 구 체크포인트용 crop 은 uw_render.crop_to_sim_fov 에 남아 있다.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self._obs, self._obs), interpolation=cv2.INTER_AREA)
        self.pub_obs.publish(to_imgmsg(small, "mono8", header))

    def _report(self) -> None:
        if self._dropped:
            self.get_logger().warning(
                f"depth 부재/시각 불일치로 버린 프레임 {self._dropped}장")
            self._dropped = 0


def main() -> None:
    rclpy.init()
    node = NbvCameraPipeline()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

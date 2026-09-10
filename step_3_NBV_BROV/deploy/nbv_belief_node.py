"""NBV 믿음(belief) 노드 — depth + pose 를 TSDF 볼륨에 융합하고 정책 관측을 낸다.

**융합 코드를 복사하지 않는다** — `envs/tsdf_fusion.py` 를 import 한다(계획서 §6).
sim 과 배포가 같은 함수를 쓰는 것이 이 노드의 존재 이유다.

트랙 A(2026-09-10 사용자 결정): pose 를 **Gazebo GT** 에서 받는다. §12.8 이 밝힌
EKF 붕괴와 파이프라인 구축을 분리하기 위해서다(step_2 의 `gazebo_truth` 와 같은 변수
격리). 추정기가 서면 `pose_source` 만 바꾼다.

기하 규약 (DEPLOY_3WEEK_PLAN §12.0):
    구면 중심 = 물체 바닥면 원점 = pool 원점 = gz world (0, 0, -2.7)
    볼륨 원점 = 물체 bbox 중심 - 1.0 m = 구면 중심 기준 (-1.0007, -1.0002, -0.5335)
    (theta, phi, psi) 는 **base_link** 를 가리킨다. 카메라는 body (0.1575, 0.0053, 0.0678).

깊이: Gazebo `rgbd_camera` 의 `depth_image` 는 **z-depth** 라 변환이 필요 없다
      (Isaac 은 유클리드를 내므로 `env_reward` 쪽에서 변환한다 — §15.2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from std_srvs.srv import Trigger

for _root in (Path(__file__).resolve().parents[1], Path("/tmp/nbv_src")):
    if (_root / "envs" / "tsdf_fusion.py").exists():
        sys.path.insert(0, str(_root))
        break
from envs import tsdf_fusion as tf  # noqa: E402

import uw_render as uw  # noqa: E402

_ENC = {"32FC1": (np.float32, 1), "mono8": (np.uint8, 1)}


def depth_from_msg(msg: Image) -> np.ndarray:
    if msg.encoding not in _ENC:
        raise ValueError(f"지원하지 않는 depth 인코딩 {msg.encoding!r}")
    dtype, _ = _ENC[msg.encoding]
    a = np.frombuffer(msg.data, dtype=dtype)
    return a.reshape(msg.height, msg.step // dtype().itemsize)[:, :msg.width]


def pack(name: str, arr: np.ndarray) -> Float32MultiArray:
    m = Float32MultiArray()
    stride = int(arr.size)
    for k, n in enumerate(arr.shape):
        stride //= int(n)
        d = MultiArrayDimension()
        d.label = f"{name}{k}"
        d.size = int(n)
        d.stride = int(stride) * int(n)
        m.layout.dim.append(d)
    m.data = arr.astype(np.float32).ravel().tolist()
    return m


class NbvBeliefNode(Node):
    def __init__(self) -> None:
        super().__init__("nbv_belief")
        p = self.declare_parameter
        p("depth_topic", "/brov/sim/camera_depth")
        p("info_topic", "/brov/camera/camera_info")
        p("pose_topic", "/brov/sim/gazebo_odometry_raw")
        p("sphere_center_world", [0.0, 0.0, -2.7])
        p("vol_origin_offset", [-1.0007, -1.0002, -0.5335])
        p("camera_offset_body", [0.15751251578330994, 0.0052856863476336,
                                 0.06784216314554214])
        p("voxel_size", 0.10)
        p("trunc_margin", 0.10)
        p("vol_dim", [20, 20, 20])
        p("max_pose_age_s", 0.30)
        p("surface_mask_path", "")   # GT 표면 voxel 마스크(.npy). 있으면 coverage 채점
        # 융합을 **학습 화각**으로 자를지. 실기 카메라(69.0x54.6)는 Isaac 학습
        # 카메라(47.2x36.3)의 **2.10배** 입체각을 덮는다 — 그대로 융합하면 한 결정에
        # 학습 때보다 훨씬 많이 얻어, 정책이 배운 "가면 얼마나 얻는가" 가 어긋난다.
        # 기본 True(학습과 정합). False 로 두면 데이터를 다 쓰되 눈금이 달라진다.
        p("crop_to_sim_fov", True)
        p("sim_hfov_deg", 47.2)
        p("sim_vfov_deg", 36.3)

        g = lambda n: self.get_parameter(n).value
        self._center = torch.tensor(list(g("sphere_center_world")), dtype=torch.float32)
        self._vol_dim = tuple(int(v) for v in g("vol_dim"))
        self._vox = float(g("voxel_size"))
        self._trunc = float(g("trunc_margin"))
        self._cam_off = torch.tensor(list(g("camera_offset_body")), dtype=torch.float32)
        self._vol_origin = (self._center
                            + torch.tensor(list(g("vol_origin_offset")),
                                           dtype=torch.float32)).unsqueeze(0)
        self._max_age = float(g("max_pose_age_s"))
        self._crop = bool(g("crop_to_sim_fov"))
        self._sim_fov = (float(g("sim_hfov_deg")), float(g("sim_vfov_deg")))
        self._window = None

        # GT 표면 마스크가 있으면 Isaac 과 **같은 눈금**으로 coverage 를 낸다
        # (`deploy/make_gt_surface.py` 가 만든다; 실측 467 voxel = Isaac 455~475 범위).
        self._surf = None
        mask_path = str(g("surface_mask_path")).strip()
        if mask_path:
            m = np.load(mask_path)
            if tuple(m.shape) != tuple(int(v) for v in g("vol_dim")):
                raise ValueError(f"마스크 shape {m.shape} != vol_dim {g('vol_dim')}")
            self._surf = torch.from_numpy(m.astype(bool)).unsqueeze(0)

        self._tsdf = torch.ones(1, *self._vol_dim)
        self._weight = torch.zeros(1, *self._vol_dim)
        self._grid = tf.build_voxel_grid(self._vol_dim, self._vox,
                                         device=torch.device("cpu"))
        self._K = None
        self._depth = None
        self._depth_t = None
        self._pose = None
        self._pose_t = None
        self._fused = 0

        self.pub_vox = self.create_publisher(Float32MultiArray, "/brov/nbv/vox_actor", 1)
        self.pub_sph = self.create_publisher(Float32MultiArray, "/brov/nbv/spherical", 1)
        self.pub_st = self.create_publisher(String, "/brov/nbv/belief_status", 1)
        self.create_subscription(CameraInfo, str(g("info_topic")), self._on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, str(g("depth_topic")), self._on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, str(g("pose_topic")), self._on_pose,
                                 qos_profile_sensor_data)
        self.create_service(Trigger, "/brov/nbv/capture", self._on_capture)
        self.create_service(Trigger, "/brov/nbv/reset_volume", self._on_reset)
        self.create_timer(1.0, self._publish_state)
        self.get_logger().info(
            f"belief 준비: vol {self._vol_dim} @ {self._vox} m, "
            f"원점(world) {self._vol_origin[0].tolist()}")

    # ── 입력 ──
    @staticmethod
    def _stamp(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_info(self, msg: CameraInfo) -> None:
        k = torch.tensor(msg.k, dtype=torch.float32).reshape(1, 3, 3)
        self._K = k

    def _on_depth(self, msg: Image) -> None:
        self._depth = torch.from_numpy(depth_from_msg(msg).astype(np.float32)).unsqueeze(0)
        self._depth_t = self._stamp(msg)

    def _on_pose(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self._pose = (
            torch.tensor([[msg.pose.pose.position.x, msg.pose.pose.position.y,
                           msg.pose.pose.position.z]], dtype=torch.float32),
            torch.tensor([[q.w, q.x, q.y, q.z]], dtype=torch.float32))
        self._pose_t = self._stamp(msg)

    # ── 융합 ──
    def _camera_world_pose(self):
        base_p, base_q = self._pose
        r = tf.quat_wxyz_to_rot(base_q)
        cam_p = base_p + (r @ self._cam_off.view(1, 3, 1)).squeeze(-1)
        return cam_p, base_q          # 카메라 회전 오프셋은 항등 (scene_cfg 와 동일)

    def _on_capture(self, _req, resp):
        missing = [n for n, v in (("camera_info", self._K), ("depth", self._depth),
                                  ("pose", self._pose)) if v is None]
        if missing:
            resp.success = False
            resp.message = f"입력 없음: {', '.join(missing)}"
            return resp
        now = self.get_clock().now().nanoseconds * 1e-9
        age = now - self._pose_t
        if age > self._max_age:
            # pose 가 낡으면 voxel 을 엉뚱한 자리에 기입한다 — 조용히 섞지 않는다.
            resp.success = False
            resp.message = f"pose 가 {age:.3f} s 낡음 (> {self._max_age})"
            return resp

        depth = torch.nan_to_num(self._depth, nan=0.0, posinf=0.0, neginf=0.0)
        K = self._K
        if self._crop:
            depth, K = self._crop_fov(depth, K)
        cam_p, cam_q = self._camera_world_pose()
        pose = tf.camera_extrinsic(cam_p, cam_q)
        self._tsdf, self._weight = tf.fuse_depth(
            depth, pose, self._tsdf, self._weight, intrinsics=K,
            vol_origin=self._vol_origin, voxel_size=self._vox,
            trunc_margin=self._trunc, vol_dim=self._vol_dim, vox_local=self._grid)
        self._fused += 1
        obs = int((self._weight > 0).sum())
        occ = int(((self._weight > 0) & (self._tsdf <= 0)).sum())
        self._publish_state()
        resp.success = True
        resp.message = (f"융합 {self._fused}회, 관측 voxel {obs}, occupied {occ}, "
                        f"cov_bin {self._coverage_bin():.4f}")
        return resp

    def _coverage_bin(self) -> float:
        """관측한 GT 표면 voxel 비율. Isaac `coverage_binary` 와 같은 정의."""
        if self._surf is None:
            return float("nan")
        seen = (self._weight > 0) & self._surf
        return float(seen.sum()) / float(self._surf.sum())

    def _crop_fov(self, depth, K):
        """학습 화각으로 중앙 crop + 주점 이동. 창은 한 번만 계산해 캐시한다."""
        h, w = int(depth.shape[1]), int(depth.shape[2])
        if self._window is None:
            self._window = uw.sim_fov_window(float(K[0, 0, 0]), w, h, *self._sim_fov)
            self.get_logger().info(
                f"융합 crop {self._window[2]}x{self._window[3]} "
                f"(원본 {w}x{h}, 학습 화각 {self._sim_fov[0]}x{self._sim_fov[1]} deg)")
        x0, y0, cw, ch = self._window
        k = K.clone()
        k[:, 0, 2] -= x0
        k[:, 1, 2] -= y0
        return depth[:, y0:y0 + ch, x0:x0 + cw].contiguous(), k

    def _on_reset(self, _req, resp):
        self._tsdf = torch.ones(1, *self._vol_dim)
        self._weight = torch.zeros(1, *self._vol_dim)
        self._fused = 0
        self._publish_state()
        resp.success = True
        resp.message = "볼륨 초기화"
        return resp

    # ── 출력 ──
    def _publish_state(self) -> None:
        ch = tf.vox_actor_channels(self._tsdf, self._weight)
        self.pub_vox.publish(pack("vox", ch[0].numpy()))
        if self._pose is not None:
            base_p, _ = self._pose
            th, ph, ps = tf.spherical_from_offset(base_p - self._center.unsqueeze(0))
            self.pub_sph.publish(pack("sph", np.array(
                [float(th[0]), float(ph[0]), float(ps[0])], dtype=np.float32)))
        obs = int((self._weight > 0).sum())
        self.pub_st.publish(String(data=(
            f"fused={self._fused} observed={obs}/{int(np.prod(self._vol_dim))} "
            f"occupied={int(((self._weight > 0) & (self._tsdf <= 0)).sum())} "
            f"cov_bin={self._coverage_bin():.4f}")))


def main() -> None:
    rclpy.init()
    node = NbvBeliefNode()
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

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

import time

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
        # 융합을 **구 학습 화각**으로 자를지. 학습 카메라가 실기 카메라와 같은
        # 640x480 / fx 465.5 (69.0x54.6 deg) 로 바뀌었으므로(정본 §15.5 (나), 2026-09-10)
        # 기본은 **자르지 않는다** — 자르면 오히려 정책이 배운 "가면 얼마나 얻는가"
        # 눈금이 어긋난다. True 는 구 카메라(47.2x36.3) 체크포인트를 돌릴 때만.
        p("crop_to_sim_fov", False)
        # 완료 edge(/brov/mission_complete 상승)에서 **스스로** 융합한다: obs_node 는 완료와
        # 동시에 neutral/disarm 하므로 그 뒤 service 로 찍으면 자세가 풀린 프레임을 얻는다
        # (run #9: 완료 후 0.5 s 만에 6–17°). 학습 env 도 hold 의 마지막 프레임을 쓴다.
        # 그 뒤 auto_capture_reuse_s 안에 오는 /brov/nbv/capture 는 그 결과를 돌려준다.
        p("auto_capture_on_mission_complete", True)
        p("auto_capture_reuse_s", 5.0)
        p("sim_hfov_deg", 47.2)   # 구 학습 카메라 — crop_to_sim_fov=True 일 때만 쓰인다
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
        self._auto = bool(g("auto_capture_on_mission_complete"))
        self._auto_reuse_s = float(g("auto_capture_reuse_s"))
        self._auto_last = None       # (t_mono, success, message)
        self._complete_prev = False
        if self._auto:
            from std_msgs.msg import Bool
            self.create_subscription(Bool, "/brov/mission_complete", self._on_complete, 10)
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
        self._depth_rx = time.monotonic()

    def _on_pose(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self._pose = (
            torch.tensor([[msg.pose.pose.position.x, msg.pose.pose.position.y,
                           msg.pose.pose.position.z]], dtype=torch.float32),
            torch.tensor([[q.w, q.x, q.y, q.z]], dtype=torch.float32))
        self._pose_t = self._stamp(msg)
        self._pose_rx = time.monotonic()
        # depth stamp 에 맞는 pose 를 고르기 위한 짧은 이력 (sim time, ~3 s)
        if not hasattr(self, "_pose_hist"):
            from collections import deque
            self._pose_hist = deque(maxlen=200)
        self._pose_hist.append((self._pose_t, self._pose))

    def _pose_at(self, t: float):
        """depth stamp 에 가장 가까운 pose 와 그 시각차. 카메라(5 Hz, 렌더 지연)는 pose(35 Hz)
        보다 sim 시간으로 0.4–1 s 뒤처진다(run #10) — 최신 pose 가 아니라 **그 프레임의 pose** 를 써야
        voxel 이 제자리에 기입된다."""
        hist = getattr(self, "_pose_hist", None)
        if not hist:
            return self._pose, abs(self._pose_t - t)
        best = min(hist, key=lambda e: abs(e[0] - t))
        return best[1], abs(best[0] - t)

    # ── 융합 ──
    def _camera_world_pose(self, pose=None):
        base_p, base_q = self._pose if pose is None else pose
        r = tf.quat_wxyz_to_rot(base_q)
        cam_p = base_p + (r @ self._cam_off.view(1, 3, 1)).squeeze(-1)
        return cam_p, base_q          # 카메라 회전 오프셋은 항등 (scene_cfg 와 동일)

    def _on_complete(self, msg) -> None:
        rising = bool(msg.data) and not self._complete_prev
        self._complete_prev = bool(msg.data)
        if rising:
            ok, message = self._fuse_latest()
            self._auto_last = (time.monotonic(), ok, message)
            self.get_logger().info(f"완료 edge 융합: {ok} {message}")

    def _on_capture(self, _req, resp):
        if self._auto and self._auto_last is not None \
                and time.monotonic() - self._auto_last[0] < self._auto_reuse_s:
            _, resp.success, resp.message = self._auto_last
            resp.message = "[edge] " + resp.message
            self._auto_last = None
            return resp
        resp.success, resp.message = self._fuse_latest()
        return resp

    def _fuse_latest(self) -> tuple[bool, str]:
        missing = [n for n, v in (("camera_info", self._K), ("depth", self._depth),
                                  ("pose", self._pose)) if v is None]
        if missing:
            return False, f"입력 없음: {', '.join(missing)}"
        # 신선도는 **수신 시각**으로 잰다. 브리지된 gz 메시지의 header stamp 는 sim time
        # 이라 이 노드의 wall clock 과 비교하면 항상 "낡음" 이 된다(폐루프 run #2 실측:
        # capture 6/6 실패). Phase 3 프로브는 wall stamp 로 재발행해서 드러나지 않았다.
        now = time.monotonic()
        age = now - self._pose_rx
        if age > self._max_age:
            # pose 가 낡으면 voxel 을 엉뚱한 자리에 기입한다 — 조용히 섞지 않는다.
            return False, f"pose 가 {age:.3f} s 낡음 (> {self._max_age})"
        # depth 와 pose 는 같은 시계(sim time): depth 프레임 시각의 pose 를 이력에서 고른다.
        pose_for_depth, skew = self._pose_at(self._depth_t)
        if skew > 0.15:
            return False, f"depth 프레임 시각의 pose 없음 (가장 가까운 pose 와 {skew:.3f} s)"

        depth = torch.nan_to_num(self._depth, nan=0.0, posinf=0.0, neginf=0.0)
        K = self._K
        if self._crop:
            depth, K = self._crop_fov(depth, K)
        cam_p, cam_q = self._camera_world_pose(pose_for_depth)
        pose = tf.camera_extrinsic(cam_p, cam_q)
        self._tsdf, self._weight = tf.fuse_depth(
            depth, pose, self._tsdf, self._weight, intrinsics=K,
            vol_origin=self._vol_origin, voxel_size=self._vox,
            trunc_margin=self._trunc, vol_dim=self._vol_dim, vox_local=self._grid)
        self._fused += 1
        obs = int((self._weight > 0).sum())
        occ = int(((self._weight > 0) & (self._tsdf <= 0)).sum())
        self._publish_state()
        raw = self._depth
        finite = torch.isfinite(raw) & (raw > 0)
        frac = float(finite.float().mean())
        dmin = float(raw[finite].min()) if bool(finite.any()) else float("nan")
        dmax = float(raw[finite].max()) if bool(finite.any()) else float("nan")
        cp = cam_p[0].tolist()
        return True, (f"융합 {self._fused}회, 관측 voxel {obs}, occupied {occ}, "
                      f"cov_bin {self._coverage_bin():.4f}; depth 유효 {frac:.2f} "
                      f"[{dmin:.2f},{dmax:.2f}] m, cam ({cp[0]:.2f},{cp[1]:.2f},{cp[2]:.2f}), "
                      f"fx {float(K[0,0,0]):.1f} {tuple(int(v) for v in depth.shape[1:])}")

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

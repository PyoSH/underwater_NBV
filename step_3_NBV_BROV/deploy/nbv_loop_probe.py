"""NBV 결정 루프를 **제어 없이** Gazebo 에서 돌린다 (트랙 A, Phase 4 검증).

왜 제어 없이 하는가: 재는 것은 belief/정책 배선이지 컨트롤러 성능이 아니다. 정적 프로브를
`set_pose` 로 정확한 시점에 놓으면 pose 재현이 완벽해서, coverage 곡선을 Isaac
`evaluate_nbv.py` 와 **같은 조건**으로 비교할 수 있다. 컨트롤러가 붙은 폐루프는 별도다.

파이프라인:
    베이스라인 정책(`envs/nbv_baselines`) -> (theta,phi,psi) -> geofence 사영
    -> 프로브 카메라를 그 시점으로 teleport -> depth/pose 발행
    -> /brov/nbv/capture -> belief 노드가 `envs/tsdf_fusion` 으로 융합
    -> coverage 기록

정책·융합·기하 전부 Isaac 과 **같은 모듈**을 쓴다. 이 스크립트가 하는 일은 배선뿐이다.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

for _root in (Path(__file__).resolve().parents[1], Path("/tmp/nbv_src")):
    if (_root / "envs" / "tsdf_fusion.py").exists():
        sys.path.insert(0, str(_root))
        break
from envs import tsdf_fusion as tf          # noqa: E402
from envs.nbv_baselines import BaselinePolicy, step_spherical  # noqa: E402

import viewpoint_geofence as gf             # noqa: E402

SPHERE_CENTER = np.array([0.0, 0.0, -2.7])          # gz world
CAM_OFFSET = np.array([0.15751251578330994, 0.0052856863476336, 0.06784216314554214])
FX = 465.5181034880913


def rpy_from_rot(r):
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 1.0 - 1e-6:
        return math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])
    return math.atan2(-r[1, 2], r[1, 1]), pitch, 0.0


def quat_xyzw_from_rot(r):
    q = tf.rot_to_quat_wxyz(torch.tensor(r, dtype=torch.float32).unsqueeze(0))[0]
    return float(q[1]), float(q[2]), float(q[3]), float(q[0])


class LoopHarness(Node):
    def __init__(self, world: str, probe_depth: str, w: int, h: int):
        super().__init__("nbv_loop_probe")
        self.world = world
        self._w, self._h = w, h
        self._depth_seen = 0
        self.pub_depth = self.create_publisher(Image, "/brov/sim/camera_depth",
                                               qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, "/brov/camera/camera_info",
                                              qos_profile_sensor_data)
        self.pub_odom = self.create_publisher(Odometry, "/brov/sim/gazebo_odometry_raw",
                                              qos_profile_sensor_data)
        self.create_subscription(Image, probe_depth, self._on_depth,
                                 qos_profile_sensor_data)
        self.cli_capture = self.create_client(Trigger, "/brov/nbv/capture")
        self.cli_reset = self.create_client(Trigger, "/brov/nbv/reset_volume")
        self._latest = None
        self._prev = None

    def _on_depth(self, msg: Image) -> None:
        self._prev = self._latest
        self._latest = msg
        self._depth_seen += 1

    def spin(self, dt: float) -> None:
        rclpy.spin_once(self, timeout_sec=dt)

    def wait_first(self, timeout=20.0) -> bool:
        t0 = time.time()
        while rclpy.ok() and self._latest is None and time.time() - t0 < timeout:
            self.spin(0.05)
        return self._latest is not None

    @staticmethod
    def _same(a: Image, b: Image) -> bool:
        return a is not None and b is not None and a.data == b.data

    def settle_after_move(self, ref: Image, timeout=6.0) -> str:
        """이동 전 프레임과 달라질 때까지, 그 뒤 연속 두 장이 같아질 때까지.

        "N 프레임 대기" 로는 이전 pose 의 프레임을 받는다 — 2026-09-10 실측.
        """
        t0 = time.time()
        changed = ref is None
        while rclpy.ok() and time.time() - t0 < timeout:
            self.spin(0.02)
            if self._latest is None:
                continue
            if not changed:
                if not self._same(self._latest, ref):
                    changed = True
                continue
            if self._same(self._latest, self._prev):
                return "ok"
        return "timeout_changed" if changed else "timeout_nochange"

    def set_probe(self, p, r) -> bool:
        x, y, z, w = quat_xyzw_from_rot(r)
        req = (f'name: "nbv_probe_camera", position: {{x: {p[0]:.6f}, y: {p[1]:.6f}, '
               f'z: {p[2]:.6f}}}, orientation: {{x: {x:.8f}, y: {y:.8f}, z: {z:.8f}, '
               f'w: {w:.8f}}}')
        out = subprocess.run(
            ["gz", "service", "-s", f"/world/{self.world}/set_pose",
             "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
             "--timeout", "3000", "--req", req], capture_output=True, text=True)
        return "data: true" in out.stdout

    def publish_frame(self, base_p, base_r) -> None:
        """belief 노드가 기대하는 계약으로 depth/info/odom 을 낸다."""
        now = self.get_clock().now().to_msg()
        d = Image()
        d.header.stamp = now
        d.header.frame_id = "camera_optical_frame"
        d.height, d.width = self._latest.height, self._latest.width
        d.encoding, d.is_bigendian, d.step = self._latest.encoding, 0, self._latest.step
        d.data = self._latest.data
        self.pub_depth.publish(d)

        info = CameraInfo()
        info.header.stamp = now
        info.header.frame_id = "camera_optical_frame"
        info.width, info.height = d.width, d.height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        cx, cy = d.width / 2.0, d.height / 2.0
        info.k = [FX, 0.0, cx, 0.0, FX, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [FX, 0.0, cx, 0.0, 0.0, FX, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)

        o = Odometry()
        o.header.stamp = now
        o.header.frame_id = "odom"
        o.child_frame_id = "base_link"
        o.pose.pose.position.x = float(base_p[0])
        o.pose.pose.position.y = float(base_p[1])
        o.pose.pose.position.z = float(base_p[2])
        qx, qy, qz, qw = quat_xyzw_from_rot(base_r)
        o.pose.pose.orientation.x, o.pose.pose.orientation.y = qx, qy
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = qz, qw
        self.pub_odom.publish(o)

    def call(self, cli, timeout=10.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(Trigger.Request())
        t0 = time.time()
        while rclpy.ok() and not fut.done() and time.time() - t0 < timeout:
            self.spin(0.05)
        return fut.result() if fut.done() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="pool_sweep")
    ap.add_argument("--probe-depth", default="/nbv/probe/depth_image")
    ap.add_argument("--policy", default="random",
                    choices=["hold", "approach", "sweep", "orbit", "random"])
    ap.add_argument("--decisions", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psi-min", type=float, default=1.4)
    ap.add_argument("--psi-max", type=float, default=2.0)
    ap.add_argument("--phi-min-deg", type=float, default=10.0)
    ap.add_argument("--phi-max-deg", type=float, default=80.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = LoopHarness(args.world, args.probe_depth, 640, 480)
    if not node.wait_first():
        raise SystemExit(f"프로브 depth 토픽 {args.probe_depth} 에서 프레임이 오지 않는다")
    if node.call(node.cli_reset) is None:
        raise SystemExit("belief 노드의 /brov/nbv/reset_volume 응답 없음")

    rng = np.random.default_rng(args.seed)
    theta = torch.tensor([float(rng.uniform(0, 2 * math.pi))])
    phi = torch.tensor([float(rng.uniform(math.radians(args.phi_min_deg),
                                          math.radians(args.phi_max_deg)))])
    psi = torch.tensor([float(rng.uniform(args.psi_min, args.psi_max))])
    pol = BaselinePolicy(args.policy, seed=args.seed)

    rows = []
    for k in range(args.decisions):
        th, ph, ps = float(theta[0]), float(phi[0]), float(psi[0])
        _, _, ps_proj, reason = gf.project(th, ph, ps, args.psi_min, args.psi_max)
        psi = torch.tensor([ps_proj])

        off = tf.offset_from_spherical(theta, phi, psi)[0].numpy()
        base_p = SPHERE_CENTER + off
        base_r = tf.quat_wxyz_to_rot(tf.look_at_quat(
            torch.tensor(base_p, dtype=torch.float32).unsqueeze(0),
            torch.tensor(SPHERE_CENTER, dtype=torch.float32).unsqueeze(0)))[0].numpy()
        cam_p = base_p + base_r @ CAM_OFFSET

        ref = node._latest
        moved = node.set_probe(cam_p, base_r)
        why = node.settle_after_move(ref) if moved else "set_pose_failed"
        node.publish_frame(base_p, base_r)
        for _ in range(6):
            node.spin(0.02)
        res = node.call(node.cli_capture)
        if res is not None and not res.success and "depth" in res.message:
            # 첫 결정에서 belief 노드가 아직 depth 를 못 받은 경우 — 한 번 더 준다.
            for _ in range(20):
                node.publish_frame(base_p, base_r)
                node.spin(0.05)
            res = node.call(node.cli_capture)
        msg = res.message if res is not None else "no-response"
        ok = bool(res.success) if res is not None else False
        obs = occ = -1
        cov = float("nan")
        if ok and "관측 voxel" in msg:
            try:
                obs = int(msg.split("관측 voxel")[1].split(",")[0])
                occ = int(msg.split("occupied")[1].split(",")[0])
                cov = float(msg.split("cov_bin")[1])
            except (IndexError, ValueError):
                pass
        rows.append(dict(decision=k + 1, theta_deg=round(math.degrees(th), 2),
                         phi_deg=round(math.degrees(ph), 2), psi=round(ps_proj, 4),
                         projected=reason, frame=why, capture_ok=int(ok),
                         observed=obs, occupied=occ,
                         cov_bin=round(cov, 4)))
        print(f"  [{k+1:3d}/{args.decisions}] phi={math.degrees(ph):5.1f} "
              f"psi={ps_proj:.2f} {msg}", flush=True)

        theta, phi, psi = step_spherical(
            theta, phi, psi, pol.act(1), max_rate_theta=math.radians(30),
            max_rate_phi=math.radians(30), max_rate_psi=0.5,
            phi_min=math.radians(args.phi_min_deg), phi_max=math.radians(args.phi_max_deg),
            psi_min=args.psi_min, psi_max=args.psi_max)

    with open(args.out / f"loop_{args.policy}.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    good = [r for r in rows if r["observed"] >= 0]
    if good:
        print(f"\n[loop] {args.policy}: 관측 voxel {good[0]['observed']} -> "
              f"{good[-1]['observed']},  cov_bin {good[0]['cov_bin']} -> "
              f"{good[-1]['cov_bin']}")
    print(f"[loop] {args.out / f'loop_{args.policy}.csv'}")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

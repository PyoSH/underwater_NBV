"""마커 가시성·pose 정확도 (theta, phi, psi) 스윕. 2026-09-10.

푸는 문제: DEPLOY_3WEEK_PLAN.md §12.1 에서 마커 fix 가용률이 입사각 한계 가정에 따라
7.7%(<=65deg) ~ 32.1%(<=75deg) 로 흔들렸다. 그 한계값을 **렌더로 측정**해 없앤다.
같은 실행에서 marker frame 축(= aruco.yaml 의 pool_to_marker_quaternion)도 확정한다.

방법: 정적 프로브 카메라를 set_pose 로 각 시점에 옮기고 한 프레임씩 검출한다.
ROV 를 움직이지 않는 이유 — 물리/제어가 끼면 pose 가 정확히 재현되지 않아 오차의
출처가 섞인다. 여기서 재는 것은 **광학·기하 한계**이지 제어 성능이 아니다.

시점 규약 (§12.0): (theta, phi, psi) 는 **base_link**, 구면 중심은 물체 바닥면.
카메라는 base_link 에서 body (0.15751, 0.00529, 0.06784), 자세는 base_link 와 동일.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from uw_render import uw_render_bgr

# ── 상수 (§12.0) ──
POOL_Z_IN_WORLD = -2.7            # pool 원점(수조 바닥 중앙)의 gz world z
MARKER_Z_POOL = 0.9415            # 물체 로컬/pool 기준 태그면 높이
MARKER_YAW_DEG = 45.0             # 물체와 같은 yaw 로 놓인 거치판
MARKER_BLACK_EDGE_M = 0.420       # aruco.yaml marker_length_m
CAM_OFFSET_BODY = np.array([0.15751251578330994, 0.0052856863476336, 0.06784216314554214])
IMG_W, IMG_H = 640, 480


def look_at_quat(from_p: np.ndarray, to_p: np.ndarray) -> np.ndarray:
    """envs/env_utils.py::_look_at_quat 와 동일한 FLU 회전행렬을 반환 (R, 3x3).

    R 의 열 = [forward, -right, up_ortho],  right = forward x up_world.
    """
    fwd = to_p - from_p
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(fwd @ up)) > 1.0 - 1e-6:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-8
    up_o = np.cross(right, fwd)
    up_o /= np.linalg.norm(up_o) + 1e-8
    return np.stack([fwd, -right, up_o], axis=-1)


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([x, y, z, w])


def rpy_from_rot(R: np.ndarray) -> tuple[float, float, float]:
    """gz set_pose 가 받는 roll-pitch-yaw (Z-Y-X 고정축)."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 1.0 - 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


class ProbeGrabber(Node):
    """set_pose 이후 **실제로 그 pose 를 담은** 프레임을 반환한다.

    단순히 "N 프레임 기다리기"로는 안 된다 — 실측(2026-09-10): 4.0 m 로 옮긴 뒤 받은
    프레임의 태그 크기가 직전 3.0 m 시점의 값이었다. gz 렌더/브리지 파이프라인이
    pose 변경보다 몇 프레임 늦기 때문이다. 그래서 두 조건을 쓴다:
      (1) 이동 전 프레임과 **달라질 때까지** 기다린다 (렌더가 반영됐다)
      (2) 그 뒤 **연속 두 프레임이 같아질 때까지** 기다린다 (정착했다)
    gz 렌더는 잡음이 없어 정적 장면의 연속 프레임은 픽셀 단위로 동일하다.
    """

    def __init__(self, topic: str, depth_topic: str | None = None):
        super().__init__("nbv_probe_grabber")
        self._br = CvBridge()
        self._latest = None
        self._prev = None
        self._depth = None
        self._seq = 0
        self.create_subscription(Image, topic, self._cb, 1)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self._cb_depth, 1)

    def _cb(self, msg: Image) -> None:
        self._prev = self._latest
        self._latest = self._br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._seq += 1

    def _cb_depth(self, msg: Image) -> None:
        self._depth = self._br.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _spin(self, dt: float) -> None:
        rclpy.spin_once(self, timeout_sec=dt)

    def prime(self, timeout_s: float = 5.0) -> np.ndarray | None:
        t0 = time.time()
        while rclpy.ok() and self._latest is None and time.time() - t0 < timeout_s:
            self._spin(0.05)
        return self._latest

    def grab_settled(self, ref: np.ndarray | None, timeout_s: float = 6.0,
                     change_thresh: float = 1.0) -> tuple[np.ndarray | None, str]:
        t0 = time.time()
        changed = ref is None
        while rclpy.ok() and time.time() - t0 < timeout_s:
            self._spin(0.02)
            if self._latest is None:
                continue
            if not changed:
                if float(np.abs(self._latest.astype(np.int16)
                                - ref.astype(np.int16)).max()) > change_thresh:
                    changed = True
                continue
            if self._prev is not None and self._latest.shape == self._prev.shape:
                if float(np.abs(self._latest.astype(np.int16)
                                - self._prev.astype(np.int16)).max()) <= change_thresh:
                    return self._latest, "ok"
        return (self._latest, "timeout_changed" if changed else "timeout_nochange")


def set_pose(world: str, model: str, p: np.ndarray, rpy: tuple[float, float, float]) -> bool:
    q = rot_to_quat_xyzw(euler_to_rot(*rpy))
    req = (f'name: "{model}", position: {{x: {p[0]:.6f}, y: {p[1]:.6f}, z: {p[2]:.6f}}}, '
           f'orientation: {{x: {q[0]:.8f}, y: {q[1]:.8f}, z: {q[2]:.8f}, w: {q[3]:.8f}}}')
    r = subprocess.run(
        ["gz", "service", "-s", f"/world/{world}/set_pose",
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", "2000", "--req", req],
        capture_output=True, text=True)
    return "data: true" in r.stdout


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def marker_object_points(edge: float) -> np.ndarray:
    """cv2.aruco 규약: corner0 좌상, +X 우, +Y 상, +Z 마커 바깥."""
    h = edge / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="pool_sweep")
    ap.add_argument("--topic", default="/nbv/probe/image")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fx", type=float, default=465.5181034880913)
    ap.add_argument("--n-theta", type=int, default=8)
    ap.add_argument("--phi-deg", default="10,15,20,25,30,35,40,45,50,55,60,65,70,75,80")
    ap.add_argument("--psi", default="1.4,1.7,2.0")
    ap.add_argument("--depth-topic", default="/nbv/probe/depth_image")
    ap.add_argument("--plate-z", type=float, default=0.933,
                    help="거치판 바닥을 놓을 물체 로컬 높이 [m]. 시작 시 set_pose 로 옮긴다. "
                         "0.933=rim 위(기본/A), 0.833=트레이 바닥(A'). "
                         "0.745(판 높이)는 트레이에 가려 검출 불가 — 2026-09-10 렌더로 확인")
    ap.add_argument("--water", default="IB",
                    help="Jerlov 프리셋. none 이면 수중 감쇠를 적용하지 않는다")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fx = fy = args.fx
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    dist = np.zeros(5)

    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(adict, params)
    obj_pts = marker_object_points(MARKER_BLACK_EDGE_M)

    # 진실 마커 pose (gz world). 태그면 = 거치판 바닥 + 판 두께 8 mm + 0.5 mm 띄움
    marker_z_pool = args.plate_z + 0.0085
    marker_p_w = np.array([0.0, 0.0, POOL_Z_IN_WORLD + marker_z_pool])
    marker_R_w = euler_to_rot(0.0, 0.0, math.radians(MARKER_YAW_DEG))
    sphere_c_w = np.array([0.0, 0.0, POOL_Z_IN_WORLD])

    rclpy.init()
    node = ProbeGrabber(args.topic,
                        None if args.water == "none" else args.depth_topic)
    if node.prime() is None:
        raise SystemExit(f"프로브 토픽 {args.topic} 에서 프레임이 오지 않는다")

    # 거치판을 요청한 높이로 옮기고 렌더가 정착할 때까지 기다린다.
    ref0 = node._latest
    if not set_pose(args.world, "apriltag_16h5_id2",
                    np.array([0.0, 0.0, POOL_Z_IN_WORLD + args.plate_z]),
                    (0.0, 0.0, math.radians(MARKER_YAW_DEG))):
        raise SystemExit("거치판 set_pose 실패")
    node.grab_settled(ref0)
    print(f"[sweep] 거치판 바닥 z={args.plate_z:.4f} -> 태그면 z={marker_z_pool:.4f} (물체 로컬)")

    phis = [math.radians(float(v)) for v in args.phi_deg.split(",")]
    psis = [float(v) for v in args.psi.split(",")]
    thetas = [2 * math.pi * k / args.n_theta for k in range(args.n_theta)]

    rows = []
    total = len(phis) * len(psis) * len(thetas)
    done = 0
    for psi in psis:
        for phi in phis:
            for theta in thetas:
                base_p = sphere_c_w + psi * np.array([
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi)])
                R_wb = look_at_quat(base_p, sphere_c_w)
                cam_p = base_p + R_wb @ CAM_OFFSET_BODY
                rpy = rpy_from_rot(R_wb)

                ref = node._latest
                ok = set_pose(args.world, "nbv_probe_camera", cam_p, rpy)
                img, why = node.grab_settled(ref) if ok else (None, "set_pose_failed")
                rec_why = why

                # 기하 진실값
                v_cm = marker_p_w - cam_p
                slant = float(np.linalg.norm(v_cm))
                incidence = math.degrees(math.acos(
                    max(-1.0, min(1.0, float((-v_cm / slant) @ (marker_R_w[:, 2]))))))
                px_edge = fx * MARKER_BLACK_EDGE_M / slant

                rec = dict(theta_deg=round(math.degrees(theta), 2),
                           phi_deg=round(math.degrees(phi), 2), psi=psi,
                           slant_m=round(slant, 4),
                           incidence_deg=round(incidence, 2),
                           px_edge_pred=round(px_edge, 1),
                           detected=0, pos_err_m="", rot_err_deg="", px_edge_meas="",
                           frame=rec_why)
                if img is not None:
                    if args.water != "none" and node._depth is not None \
                            and node._depth.shape[:2] == img.shape[:2]:
                        img = uw_render_bgr(img, node._depth, args.water)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = detector.detectMarkers(gray)
                    if ids is not None and 2 in ids.ravel().tolist():
                        k = ids.ravel().tolist().index(2)
                        q = corners[k].reshape(4, 2).astype(np.float64)
                        rec["detected"] = 1
                        rec["px_edge_meas"] = round(float(np.mean(
                            [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])), 2)
                        good, rvec, tvec = cv2.solvePnP(
                            obj_pts, q, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if good:
                            R_cm, _ = cv2.Rodrigues(rvec)
                            t_cm = tvec.reshape(3)
                            # 카메라 optical -> gz 센서(FLU) : Zopt=+X, Xopt=-Y, Yopt=-Z
                            R_flu_opt = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
                            R_wm_est = R_wb @ R_flu_opt @ R_cm
                            p_wm_est = cam_p + R_wb @ R_flu_opt @ t_cm
                            rec["pos_err_m"] = round(float(
                                np.linalg.norm(p_wm_est - marker_p_w)), 4)
                            dR = marker_R_w.T @ R_wm_est
                            rec["rot_err_deg"] = round(float(math.degrees(math.acos(
                                max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))), 3)
                            rec["q_wm_xyzw"] = " ".join(
                                f"{v:.5f}" for v in rot_to_quat_xyzw(R_wm_est))
                rows.append(rec)
                done += 1
                if done % 40 == 0:
                    det = sum(r["detected"] for r in rows)
                    print(f"  {done}/{total}  검출 {det} ({100*det/done:.1f}%)", flush=True)

    rclpy.shutdown()
    keys = ["theta_deg", "phi_deg", "psi", "slant_m", "incidence_deg",
            "px_edge_pred", "px_edge_meas", "detected", "pos_err_m", "rot_err_deg",
            "q_wm_xyzw", "frame"]
    with open(args.out / "marker_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[sweep] {len(rows)} 시점 -> {args.out/'marker_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""NBV 폐루프 정책 노드 (Phase 4 ②, 2026-09-10) — 결정 → v3 pose 미션 → 추종 → 촬영.

학습 env 의 역할 분담을 그대로 옮긴다 (정본 §16.2; env.py:339-357, 444-455):
  이 노드   = `_pre_physics_step` + `_get_observations`
              (Δθ,Δφ,Δψ) 를 **명령 상태**에 적분·clamp → 절대 pose(구면→직교 + look_at).
              관측용 (θ,φ,ψ)_actual 은 **측정 pose**(pool) 에서 되읽는다.
  guidance/PID = DP 추종 5 s: 2점(호 분할이면 N점) v3 pose 미션. 홉마다 mission manager 를
              subprocess 로 띄워 validate→commit (한 프로세스 = 한 immutable 미션 계약 유지).
  결정당 1 프레임 = 도착+dwell 종료(`/brov/mission_complete` 상승 edge) 에서 capture.

정책·기하 전부 Isaac 과 같은 모듈: envs.nbv_baselines / envs.tsdf_fusion.look_at_quat /
viewpoint_geofence. 이 파일이 하는 일은 배선과 순서뿐이다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from brov_interfaces.msg import ResolvedMission
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

for _root in (Path(__file__).resolve().parents[1], Path("/tmp/nbv_src")):
    if (_root / "envs" / "tsdf_fusion.py").exists():
        sys.path.insert(0, str(_root))
        break
from envs import tsdf_fusion as tf                          # noqa: E402
from envs.nbv_baselines import BaselinePolicy, step_spherical  # noqa: E402

import viewpoint_geofence as gf                              # noqa: E402

LATCHED = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def spherical_from_pool(p) -> tuple[float, float, float]:
    """env.py:446-448 과 같은 역변환 (구 중심 = pool 원점)."""
    x, y, z = (float(v) for v in p)
    psi = max(math.sqrt(x * x + y * y + z * z), 1e-6)
    phi = math.acos(max(-1.0, min(1.0, z / psi)))
    theta = math.atan2(y, x) % (2.0 * math.pi)
    return theta, phi, psi


# 학습 카메라 오프셋 (scene_cfg / 실기 calibration). SITL 의 Edo 선체 mesh 는 이 자리를 감싸
# depth 가 전부 -inf 라(run #10), 카메라를 body +X 로 cam_x_shift 만큼 빼고 base_link 목표를
# 같은 만큼 look 축 뒤로 물린다: p_base_gz = p_base_train - R*(Δ,0,0). 그러면 카메라는
# p_base_train + R*TRAIN_CAM 에 **정확히** 놓이고 자세도 그대로다. 실기는 Δ=0.
TRAIN_CAM = (0.15751251578330994, 0.0052856863476336, 0.06784216314554214)


def rot_from_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def base_gz_from_train(p_train, q_wxyz, shift: float) -> np.ndarray:
    return np.asarray(p_train, dtype=float) - rot_from_wxyz(q_wxyz) @ np.array([shift, 0.0, 0.0])


def base_train_from_measured(p_meas, q_xyzw, shift: float) -> np.ndarray:
    """측정 base_link → 학습 등가 base_link (관측 (θ,φ,ψ)_actual 은 이것으로 되읽는다)."""
    q = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
    return np.asarray(p_meas, dtype=float) + rot_from_wxyz(q) @ np.array([shift, 0.0, 0.0])


def look_at_wxyz(p_pool) -> tuple[float, float, float, float]:
    q = tf.look_at_quat(torch.tensor([p_pool], dtype=torch.float32),
                        torch.zeros(1, 3, dtype=torch.float32))[0]
    return tuple(float(v) for v in q)


class PolicyLoop(Node):
    def __init__(self, a) -> None:
        super().__init__("nbv_policy")
        self.a = a
        self._pool_pose = None            # (t_mono, p(3), q_xyzw(4))
        self._complete = False
        self._complete_edges = 0
        self._control_active = False
        self._resolved = None
        self._manager_status = None
        self._manager = None
        self._edge_pose = None
        self.create_subscription(Odometry, "/brov/localization/odometry_pool",
                                 self._on_pool, 20)
        self.create_subscription(Bool, "/brov/mission_complete", self._on_complete, 10)
        self.create_subscription(Bool, "/brov/control_active", self._on_active, 10)
        self.create_subscription(ResolvedMission, "/brov/mission/resolved",
                                 self._on_resolved, LATCHED)
        self.create_subscription(String, "/brov/mission/status",
                                 self._on_manager_status, LATCHED)
        self.pub_draft = self.create_publisher(PathMsg, "/brov/mission/draft_path", 1)
        # RViz 용 (nbv_viz_node): 정책이 낸 현재 목표 pose 와 이번 홉의 waypoint 열 (pool 프레임)
        self.pub_target = self.create_publisher(PoseStamped, "/brov/nbv/target_pose", LATCHED)
        self.pub_hop = self.create_publisher(PoseArray, "/brov/nbv/hop_waypoints", LATCHED)
        self.cli = {n: self.create_client(Trigger, t) for n, t in {
            "validate": "/brov/mission/validate", "commit": "/brov/mission/commit",
            "prepare": "/brov/prepare_control", "arm": "/brov/arm_control",
            "start": "/brov/start_control", "stop": "/brov/stop_control",
            "disarm": "/brov/disarm_control",
            "model_start": "/brov/model_based/start", "model_stop": "/brov/model_based/stop",
            "capture": "/brov/nbv/capture", "reset_volume": "/brov/nbv/reset_volume",
        }.items()}

    # ── 콜백 ──
    def _on_pool(self, m: Odometry) -> None:
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self._pool_pose = (time.monotonic(), np.array([p.x, p.y, p.z]), (q.x, q.y, q.z, q.w))

    def _on_complete(self, m: Bool) -> None:
        if m.data and not self._complete:
            self._complete_edges += 1
        self._complete = bool(m.data)

    def _on_active(self, m: Bool) -> None:
        self._control_active = bool(m.data)

    def _on_resolved(self, m: ResolvedMission) -> None:
        self._resolved = m

    def _on_manager_status(self, m: String) -> None:
        try:
            self._manager_status = json.loads(m.data)
        except ValueError:
            self._manager_status = {"state": m.data}

    # ── 유틸 ──
    def spin(self, dt=0.02) -> None:
        rclpy.spin_once(self, timeout_sec=dt)

    def wait_until(self, pred, timeout: float, what: str, quiet: bool = False) -> bool:
        t0 = time.monotonic()
        while rclpy.ok() and time.monotonic() - t0 < timeout:
            self.spin()
            if pred():
                return True
        if not quiet:
            self.get_logger().error(f"timeout ({timeout:.0f}s) waiting for {what}")
        return False

    def kill_manager(self) -> None:
        proc = self._manager
        self._manager = None
        if proc is None or proc.poll() is not None:
            return
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)

    def call(self, name: str, timeout=10.0, retries=1, retry_sleep=0.2):
        cli = self.cli[name]
        if not cli.wait_for_service(timeout_sec=timeout):
            return False, f"service {name} unavailable"
        for _ in range(retries):
            fut = cli.call_async(Trigger.Request())
            t0 = time.monotonic()
            while rclpy.ok() and not fut.done() and time.monotonic() - t0 < timeout:
                self.spin()
            if fut.done() and fut.result() is not None:
                r = fut.result()
                if r.success:
                    return True, r.message
                last = r.message
            else:
                last = "no response"
            t1 = time.monotonic()
            while time.monotonic() - t1 < retry_sleep:
                self.spin()
        return False, last

    def publish_viz_target(self, target_pool, q_wxyz, hop_wps=None, hop_atts=None) -> None:
        """정책 출력(base_link 목표 + look-at 자세)을 RViz 노드에 알린다. 제어에는 쓰이지 않는다."""
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg(); ps.header.frame_id = "pool"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in target_pool)
        w, x, y, z = q_wxyz
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = float(x), float(y), float(z), float(w)
        self.pub_target.publish(ps)
        pa = PoseArray(); pa.header = ps.header
        for p_, q_ in zip(hop_wps or [], hop_atts or []):
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in p_)
            w, x, y, z = q_
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = float(x), float(y), float(z), float(w)
            pa.poses.append(pose)
        self.pub_hop.publish(pa)

    def fresh_pool_pose(self, max_age=0.5):
        if self._pool_pose is None or time.monotonic() - self._pool_pose[0] > max_age:
            return None
        return self._pool_pose[1], self._pool_pose[2]

    # ── 홉 1회: 미션 계약 ──
    def commit_hop(self, waypoints_pool, attitudes_wxyz, manager_params: str):
        """mission manager 를 홉 전용 subprocess 로 띄워 draft→validate→commit. ResolvedMission 반환.

        manager 는 여기서 죽이지 않는다: obs_node 의 권한 게이트는 prepare/arm/start 내내
        `/brov/mission/resolved` 발행자가 **정확히 하나** 살아 있기를 요구한다(run #2 실측).
        홉이 끝난 뒤 `kill_manager()` 로 정리하고, 다음 홉이 새 프로세스를 띄운다
        (한 프로세스 = 한 immutable 미션 계약은 그대로).
        """
        before = self._resolved.mission_id if self._resolved is not None else ""
        self.kill_manager()
        proc = subprocess.Popen(
            ["ros2", "run", "brov_mission", "mission_manager_node", "--ros-args",
             "--params-file", manager_params],
            stdout=open(self.a.out / f"manager_hop{self._hop:03d}.log", "w"),
            stderr=subprocess.STDOUT, start_new_session=True)
        self._manager = proc
        try:
            if not self.wait_until(lambda: self.cli["validate"].service_is_ready()
                                   and self.cli["commit"].service_is_ready()
                                   and self.pub_draft.get_subscription_count() >= 1, 15.0,
                                   "mission manager services/draft subscriber"):
                self.kill_manager()
                return None, "manager did not come up"
            path = PathMsg()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = "pool"
            for p, q in zip(waypoints_pool, attitudes_wxyz):
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in p)
                w, x, y, z = q
                ps.pose.orientation.x, ps.pose.orientation.y = float(x), float(y)
                ps.pose.orientation.z, ps.pose.orientation.w = float(z), float(w)
                path.poses.append(ps)
            self._manager_status = None
            for _ in range(30):
                self.pub_draft.publish(path)
                if self.wait_until(lambda: (self._manager_status or {}).get("state") == "DRAFT",
                                   0.3, "draft ack", quiet=True):
                    break
            else:
                self.kill_manager()
                return None, "manager never acknowledged the draft"
            # 새 manager 는 latched status(5 Hz)·aligned odometry 를 받은 뒤에야 validate 한다
            ok, msg = self.call("validate", retries=12, retry_sleep=0.5)
            if not ok:
                self.kill_manager()
                return None, f"validate: {msg}"
            ok, msg = self.call("commit", retries=3)
            if not ok:
                self.kill_manager()
                return None, f"commit: {msg}"
            if not self.wait_until(lambda: self._resolved is not None
                                   and self._resolved.mission_id != before, 10.0,
                                   "resolved mission"):
                self.kill_manager()
                return None, "no ResolvedMission after commit"
            return self._resolved, msg
        except Exception:
            self.kill_manager()
            raise

    # ── 홉 1회: 제어 ──
    def fly_hop(self, hop_timeout_s: float) -> tuple[bool, str, float]:
        for attempt in range(3):
            ok, msg = self.call("prepare", retries=5, retry_sleep=0.3)
            if not ok:
                return False, f"prepare: {msg}", 0.0
            ok, msg = self.call("arm", retries=20, retry_sleep=0.1)
            if ok:
                break
            # 이전 홉의 disarm 잔향으로 arm 이 revoke 되면 prepared 계약도 지워진다 → 다시 prepare
            if "not prepared" not in msg and "prepare" not in msg:
                return False, f"arm: {msg}", 0.0
            self.get_logger().warning(f"arm attempt {attempt + 1} revoked ({msg}); re-preparing")
        else:
            return False, f"arm: {msg}", 0.0
        edges = self._complete_edges
        t0 = time.monotonic()
        ok, msg = self.call("start", retries=10, retry_sleep=0.1)
        if not ok:
            self.call("disarm")
            return False, f"start: {msg}", 0.0
        ok, msg = self.call("model_start", retries=30, retry_sleep=0.1)
        if not ok:
            self.call("stop"); self.call("disarm")
            return False, f"model_based/start: {msg}", 0.0
        done = self.wait_until(lambda: self._complete_edges > edges, hop_timeout_s,
                               "mission_complete")
        dt = time.monotonic() - t0
        self._edge_pose = self.fresh_pool_pose(max_age=1.0)   # disarm 으로 자세가 풀리기 전
        if not done:
            self.call("model_stop"); self.call("stop"); self.call("disarm")
            return False, "hop timeout", dt
        return True, "complete", dt


def parse_capture(msg: str):
    obs = occ = -1
    cov = float("nan")
    if "관측 voxel" in msg:
        try:
            obs = int(msg.split("관측 voxel")[1].split(",")[0])
            occ = int(msg.split("occupied")[1].split(",")[0])
            cov = float(msg.split("cov_bin")[1].split(";")[0])
        except (IndexError, ValueError):
            pass
    return obs, occ, cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="random",
                    choices=["hold", "approach", "sweep", "orbit", "random"])
    ap.add_argument("--decisions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psi-min", type=float, default=1.4)
    ap.add_argument("--psi-max", type=float, default=2.0)
    ap.add_argument("--phi-min-deg", type=float, default=10.0)
    ap.add_argument("--phi-max-deg", type=float, default=80.0)
    ap.add_argument("--max-duration-s", type=float, default=60.0,
                    help="미션 계약의 max_duration (sim 시간) — 정보용")
    ap.add_argument("--hop-timeout-s", type=float, default=150.0,
                    help="완료 edge 를 기다리는 **wall** 시간. Gazebo RTF 0.5 에서 sim 60 s = wall 120 s")
    ap.add_argument("--att-tol-deg", type=float, default=10.0,
                    help="완료 edge 의 자세 오차가 이보다 크면 settled_timeout 으로 기록 (계약 tolerance 와 맞춘다)")
    ap.add_argument("--min-hop-m", type=float, default=0.06,
                    help="이보다 짧은 홉은 미션 없이 제자리 촬영 (manager min_segment 0.05)")
    ap.add_argument("--manager-params", required=True)
    ap.add_argument("--cam-x-shift", type=float, default=0.0,
                    help="SITL 카메라 x 오프셋 - 학습 오프셋 [m]. base_link 목표를 look 축 뒤로 이만큼 물린다")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    phi_min, phi_max = math.radians(a.phi_min_deg), math.radians(a.phi_max_deg)

    rclpy.init()
    node = PolicyLoop(a)
    node._hop = 0
    log = node.get_logger()
    if not node.wait_until(lambda: node.fresh_pool_pose() is not None, 30.0, "pool pose"):
        return 1
    ok, msg = node.call("reset_volume")
    log.info(f"belief reset: {ok} {msg}")

    # 명령 상태 초기화 = 측정 pose 를 학습 범위로 clamp (첫 홉이 구면 위로 데려간다)
    p0_meas, q0_meas = node.fresh_pool_pose()
    p0 = base_train_from_measured(p0_meas, q0_meas, a.cam_x_shift)
    th, ph, ps = spherical_from_pool(p0)
    theta = torch.tensor([th]); phi = torch.tensor([min(max(ph, phi_min), phi_max)])
    psi = torch.tensor([min(max(ps, a.psi_min), a.psi_max)])
    pol = BaselinePolicy(a.policy, seed=a.seed)
    log.info(f"start pool={np.round(p0, 3).tolist()} sph_actual=({math.degrees(th):.1f}, "
             f"{math.degrees(ph):.1f}, {ps:.2f}) -> cmd=({math.degrees(float(theta)):.1f}, "
             f"{math.degrees(float(phi)):.1f}, {float(psi):.2f})")

    rows = []
    for k in range(a.decisions):
        node._hop = k + 1
        t_hop = time.monotonic()
        th_c, ph_c, ps_c = float(theta[0]), float(phi[0]), float(psi[0])
        _, _, ps_c, proj = gf.project(th_c, ph_c, ps_c, a.psi_min, a.psi_max)
        psi = torch.tensor([ps_c])
        target = np.array(gf.view_position(th_c, ph_c, ps_c))

        cur = node.fresh_pool_pose()
        if cur is None:
            log.error("pool pose stale; abort"); break
        p_cur, q_cur_xyzw = cur
        p_cur_train = base_train_from_measured(p_cur, q_cur_xyzw, a.cam_x_shift)
        sph_cur = spherical_from_pool(p_cur_train)
        hop_len = float(np.linalg.norm(target - p_cur_train))
        status = "ok"; n_wp = 0; mission_id = ""; fly_s = 0.0
        q_target = look_at_wxyz(target)
        node.publish_viz_target(base_gz_from_train(target, q_target, a.cam_x_shift), q_target)
        if hop_len < a.min_hop_m:
            status = "hold"     # 정책이 제자리를 골랐다 — sim 의 PID 유지와 같다
        else:
            pts, pieces = gf.arc_split(sph_cur, (th_c, ph_c, ps_c))
            if pts is None:
                status = "reject_chord"
            else:
                wps_train = [np.array(gf.view_position(*s)) for s in pts]
                atts = [look_at_wxyz(w) for w in wps_train]
                # 미션에는 gz base_link (학습 base 를 look 축 뒤로 Δ) 를 싣는다; wp0 = 측정 pose 그대로
                q0 = (q_cur_xyzw[3], q_cur_xyzw[0], q_cur_xyzw[1], q_cur_xyzw[2])
                wps = [p_cur] + [base_gz_from_train(w, q, a.cam_x_shift) for w, q in zip(wps_train, atts)]
                atts = [q0] + atts
                n_wp = len(wps)
                node.publish_viz_target(wps[-1], atts[-1], wps, atts)
                resolved, msg = node.commit_hop(wps, atts, a.manager_params)
                if resolved is None:
                    status = f"commit_fail:{msg}"
                else:
                    mission_id = resolved.mission_id
                    ok, why, fly_s = node.fly_hop(a.hop_timeout_s)
                    if not ok:
                        status = f"fly_fail:{why}"
                    node.kill_manager()
        # 촬영 (도착 직후; 실패 홉도 현재 pose 에서 찍어 기록한다)
        for _ in range(5):
            node.spin()
        node._edge_pose = None if status != "ok" else node._edge_pose
        ok, msg = node.call("capture", retries=3, retry_sleep=0.3)
        log.info(f"capture: {ok} {msg}")
        obs, occ, cov = parse_capture(msg if ok else "")
        after = node._edge_pose if node._edge_pose is not None else node.fresh_pool_pose(max_age=1.0)
        if after is not None:
            p_a_meas, q_a = after
            p_a = base_train_from_measured(p_a_meas, q_a, a.cam_x_shift)
            sph_a = spherical_from_pool(p_a)
            pos_err = float(np.linalg.norm(p_a - target))
            q_t = look_at_wxyz(target)
            q_a_wxyz = (q_a[3], q_a[0], q_a[1], q_a[2])
            d = abs(sum(u * v for u, v in zip(q_t, q_a_wxyz)))
            att_err = math.degrees(2.0 * math.acos(min(1.0, d)))
        else:
            sph_a = (float("nan"),) * 3; pos_err = att_err = float("nan")
        if status == "ok" and (pos_err > 0.15 + 0.05 or att_err > a.att_tol_deg + 1.0):
            # 완료 edge 는 max_duration 종료에서도 뜬다 — 오차가 계약 tolerance 를 넘으면 정착 실패다
            status = "settled_timeout"
        rows.append(dict(
            decision=k + 1, status=status, mission=mission_id[:8], waypoints=n_wp,
            theta_cmd=round(math.degrees(th_c), 2), phi_cmd=round(math.degrees(ph_c), 2),
            psi_cmd=round(ps_c, 3), projected=proj,
            theta_act=round(math.degrees(sph_a[0]), 2), phi_act=round(math.degrees(sph_a[1]), 2),
            psi_act=round(sph_a[2], 3), pos_err_m=round(pos_err, 4), att_err_deg=round(att_err, 2),
            hop_len_m=round(hop_len, 3), fly_s=round(fly_s, 1),
            hop_s=round(time.monotonic() - t_hop, 1), capture_ok=int(ok), observed=obs,
            occupied=occ, cov_bin=round(cov, 4)))
        log.info(f"[{k + 1:2d}/{a.decisions}] {status} wp={n_wp} cmd=({math.degrees(th_c):.0f},"
                 f"{math.degrees(ph_c):.0f},{ps_c:.2f}) err={pos_err:.3f}m/{att_err:.1f}deg "
                 f"fly={fly_s:.0f}s cov={cov:.3f}")
        # 다음 결정 — 관측은 측정값(sph_a)이지만 baseline 은 관측을 쓰지 않는다
        theta, phi, psi = step_spherical(
            theta, phi, psi, pol.act(1), max_rate_theta=math.radians(30),
            max_rate_phi=math.radians(30), max_rate_psi=0.5,
            phi_min=phi_min, phi_max=phi_max, psi_min=a.psi_min, psi_max=a.psi_max)

    node.call("model_stop"); node.call("stop"); node.call("disarm")
    node.kill_manager()
    with open(a.out / f"closed_loop_{a.policy}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    good = [r for r in rows if r["status"] == "ok"]
    print(f"[closed-loop] {a.policy}: hops ok {len(good)}/{len(rows)}, "
          f"pos_err median {np.median([r['pos_err_m'] for r in good]) if good else float('nan'):.3f} m, "
          f"att_err median {np.median([r['att_err_deg'] for r in good]) if good else float('nan'):.1f} deg, "
          f"cov {rows[0]['cov_bin']} -> {rows[-1]['cov_bin']}")
    print(f"[closed-loop] {a.out / f'closed_loop_{a.policy}.csv'}")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

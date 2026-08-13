#!/usr/bin/env python3
"""
관측(observation) + 액추에이션 ROS2 노드 — 정책 추론은 policy_node.py가 분리 담당.
MAVLink 연결을 이 노드 하나만 갖는다(이유는 ros2_nodes/policy_node.py 상단 주석
참조 — 이 SITL에서 두 번째 독립 연결이 동작하지 않아서, 연결을 하나로 유지하고
policy_node.py가 발행한 `/brov/thruster_pwm`을 구독해 대신 송신하는 구조로 정리함).
obs 자체는 여전히 정책과 완전히 분리되어 있어 `ros2 topic echo /brov/observation`
으로 독립 검증 가능하다.

실행(컨테이너 안, ROS2 소싱 후):
    python3 obs_node.py --ros-args \
        -p connection:=udpin:0.0.0.0:14550 \
        -p waypoints:="0,0,0;3,0,0" \
        -p cruise_speed:=0.3 \
        -p heading_mode:=straight \
        -p waypoint_frame:=start_heading \
        -p loop:=false \
        -p send_pwm:=true -p arm:=true

`loop`(기본 false): 마지막 웨이포인트 도달 후 처음으로 되돌아가 반복할지(true,
sim의 test_policy.py 평가 스크립트처럼 계속 왕복/순환하고 싶을 때) 여부. false면
마지막 웨이포인트 도달 이력으로 `/brov/mission_complete`가 True가 되며, 이후에도
최종 waypoint position hold는 계속 동작함.

`waypoint_frame`: `start_heading`(기본)은 start_control 순간 위치와 yaw를 각각
원점과 +X 전방으로 정의하고, `ned`는 시작 위치만 0으로 만든다.
`heading_mode`: `straight`는 시작 yaw 유지, `align`은 LOS 방향 정렬,
`upright`는 선택 frame의 yaw=0을 목표로 한다.

발행 토픽:
    /brov/observation      (Float32MultiArray, 16) — envs/vel_env.py._get_observations()와 동일 규약
    /brov/debug/pos_ned    (Float32MultiArray, 3)  — 원시 LOCAL_POSITION_NED (검증용)
    /brov/debug/vel_ned    (Float32MultiArray, 3)  — 원시 LOCAL_POSITION_NED 속도 (검증용)
    /brov/debug/att_quat_ned (Float32MultiArray, 4) — 원시 ATTITUDE_QUATERNION [w,x,y,z] (검증용)
    /brov/target_waypoint  (Float32MultiArray, 3)  — 목표 웨이포인트(선택 frame, m)
    /brov/waypoint_idx     (Int32)                  — 현재 세그먼트 시작 인덱스(0-base)
    /brov/mission_complete (Bool)                   — loop:=false일 때만 의미 있음, 마지막 웨이포인트 도달 시 True
    /brov/control_active   (Bool)                   — 적분 및 PWM 허용 상태
    /brov/camera_tilt/commanded (Float32)           — 마지막으로 보낸 정규화 tilt [-1,1]

구독 토픽:
    /brov/thruster_pwm (8,) [-1,1] — policy_node.py가 발행. `send_pwm:=true`일 때만
    실제 RC_CHANNELS_OVERRIDE로 송신(기본 false — obs만 보고 싶을 때 안전하게 끌 수 있음).
    /brov/camera_tilt/command (Float32) [-1,1] — RC8과 분리된 MAVLink mount pitch 명령.
    -1=최대 아래, 0=중앙, +1=최대 위. 각도/속도 제한은 ROS parameter로 설정한다.
    /brov/estop (std_msgs/Empty) — 사용자 입력에 의한 즉시 정지. 아무 메시지나 오면
    트립: 중립(1500us) 즉시 송신 + disarm + 이후 /brov/thruster_pwm 영구 무시(자동
    재개 없음 — 재개하려면 노드를 재시작해야 함, 발산 중 자동 재무장으로 반복 트립되는
    것을 막기 위한 의도적 설계). 트리거 예시:
        ros2 topic pub -1 /brov/estop std_msgs/msg/Empty {}
    자동 발산 감지(다중 기준 트리거)는 아직 미구현 — 지금은 사람이 직접 판단해서
    이 토픽을 쏘는 수동 정지만 있다.

제어 lifecycle 서비스:
    ros2 service call /brov/start_control std_srvs/srv/Trigger {}
        healthy telemetry 확인 후 적분항을 0으로 reset하고 적분/PWM을 허용한다.
    ros2 service call /brov/stop_control std_srvs/srv/Trigger {}
        적분을 동결하고 PWM neutral을 송신한다.
    ros2 service call /brov/reset_integrator std_srvs/srv/Trigger {}
        fault latch와 적분항을 reset하되 control은 frozen 상태로 유지한다.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))   # deploy의 부모 — deploy 패키지 하나만 있으면 됨(vendoring)

import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray, Empty, Int32, Bool
from std_srvs.srv import Trigger

from deploy.real_robot_interface import RealRobotInterface
from deploy.obs_builder import ObservationBuilder
from deploy.guidance_standalone import LOSGuidance
from deploy import math_utils as mu


def _parse_waypoints(s: str) -> torch.Tensor:
    pts = [[float(v) for v in wp.split(",")] for wp in s.split(";")]
    return torch.tensor(pts).unsqueeze(0)


class ObsNode(Node):
    def __init__(self):
        super().__init__("brov_obs_node")
        self.declare_parameter("connection", "udpin:0.0.0.0:14550")
        self.declare_parameter("waypoints", "0,0,0;3,0,0")
        self.declare_parameter("cruise_speed", 0.2)
        self.declare_parameter("heading_mode", "straight")
        self.declare_parameter("waypoint_frame", "start_heading")
        self.declare_parameter("reach_threshold", 0.15)
        self.declare_parameter("lookahead_dist", 1.0)
        self.declare_parameter("depth_hold_kp", 0.8)
        self.declare_parameter("depth_speed_limit", 0.1)
        self.declare_parameter("terminal_hold_kp", 0.5)
        self.declare_parameter("terminal_speed_limit", 0.1)
        self.declare_parameter("send_pwm", False)
        self.declare_parameter("arm", False)
        self.declare_parameter("loop", False)
        self.declare_parameter("att_max_age_s", 0.2)
        self.declare_parameter("pos_max_age_s", 0.5)
        self.declare_parameter("ekf_max_age_s", 0.5)
        self.declare_parameter("max_integration_dt_s", 0.15)
        self.declare_parameter("integral_vel_limit", 5.0)
        self.declare_parameter("integral_att_limit", 5.0)
        self.declare_parameter("quat_norm_tolerance", 0.1)
        self.declare_parameter("camera_tilt_min_deg", -45.0)
        self.declare_parameter("camera_tilt_max_deg", 45.0)
        self.declare_parameter("camera_tilt_max_rate_deg_s", 30.0)

        conn = self.get_parameter("connection").value
        waypoints = _parse_waypoints(self.get_parameter("waypoints").value)
        waypoint_frame = str(self.get_parameter("waypoint_frame").value)
        heading_mode = str(self.get_parameter("heading_mode").value)
        if waypoint_frame not in {"ned", "start_heading"}:
            raise ValueError("waypoint_frame은 'ned' 또는 'start_heading'이어야 함")
        valid_heading_modes = {"align", "upright", "straight", "random_at_waypoint"}
        if heading_mode not in valid_heading_modes:
            raise ValueError(
                f"heading_mode={heading_mode!r} invalid; expected {sorted(valid_heading_modes)}"
            )
        self._camera_tilt_min_deg = float(self.get_parameter("camera_tilt_min_deg").value)
        self._camera_tilt_max_deg = float(self.get_parameter("camera_tilt_max_deg").value)
        self._camera_tilt_max_rate_deg_s = float(
            self.get_parameter("camera_tilt_max_rate_deg_s").value
        )
        if self._camera_tilt_min_deg >= 0.0 or self._camera_tilt_max_deg <= 0.0:
            raise ValueError("camera tilt 범위는 min < 0 < max여야 함")
        if self._camera_tilt_max_rate_deg_s <= 0.0:
            raise ValueError("camera_tilt_max_rate_deg_s는 양수여야 함")

        self.interface = RealRobotInterface(conn)
        self.interface.connect()

        self._send_pwm = bool(self.get_parameter("send_pwm").value)
        if self._send_pwm:
            self.interface.enable_passthrough()
            self.get_logger().info(
                "RC7/RC8 camera option 격리 + RCPassThru 전환 완료 — "
                "/brov/thruster_pwm 수신 시 실제 송신"
            )
            if bool(self.get_parameter("arm").value):
                if self.interface.arm():
                    self.get_logger().info("arm 완료")
                else:
                    self.get_logger().error("arm 실패 — send_pwm은 계속되지만 무장 안 된 상태")

        self.sub_pwm = self.create_subscription(
            Float32MultiArray, "/brov/thruster_pwm", self._on_pwm, 10
        )
        self._estopped = False
        self.sub_estop = self.create_subscription(Empty, "/brov/estop", self._on_estop, 10)
        self.sub_camera_tilt = self.create_subscription(
            Float32, "/brov/camera_tilt/command", self._on_camera_tilt, 10
        )

        self.obs_builder = ObservationBuilder(
            integral_vel_limit=float(self.get_parameter("integral_vel_limit").value),
            integral_att_limit=float(self.get_parameter("integral_att_limit").value),
            waypoint_frame=waypoint_frame,
        )
        self.guidance = LOSGuidance(
            waypoints, "cpu",
            cruise_speed=self.get_parameter("cruise_speed").value,
            lookahead_dist=self.get_parameter("lookahead_dist").value,
            heading_mode=heading_mode,
            reach_threshold=self.get_parameter("reach_threshold").value,
            loop=bool(self.get_parameter("loop").value),
            depth_hold_kp=self.get_parameter("depth_hold_kp").value,
            depth_speed_limit=self.get_parameter("depth_speed_limit").value,
            terminal_hold_kp=self.get_parameter("terminal_hold_kp").value,
            terminal_speed_limit=self.get_parameter("terminal_speed_limit").value,
        )

        self.pub_obs  = self.create_publisher(Float32MultiArray, "/brov/observation", 10)
        self.pub_pos  = self.create_publisher(Float32MultiArray, "/brov/debug/pos_ned", 10)
        self.pub_vel  = self.create_publisher(Float32MultiArray, "/brov/debug/vel_ned", 10)
        self.pub_quat = self.create_publisher(Float32MultiArray, "/brov/debug/att_quat_ned", 10)
        self.pub_pos_mission = self.create_publisher(
            Float32MultiArray, "/brov/debug/pos_mission", 10
        )
        self.pub_v_body_zup = self.create_publisher(
            Float32MultiArray, "/brov/debug/v_body_zup", 10
        )
        self.pub_v_desired_body_zup = self.create_publisher(
            Float32MultiArray, "/brov/debug/v_desired_body_zup", 10
        )
        self.pub_servo_output = self.create_publisher(
            Int32MultiArray, "/brov/debug/servo_output_us", 10
        )
        self.pub_target_wp = self.create_publisher(Float32MultiArray, "/brov/target_waypoint", 10)
        self.pub_wp_idx     = self.create_publisher(Int32, "/brov/waypoint_idx", 10)
        self.pub_mission_complete = self.create_publisher(Bool, "/brov/mission_complete", 10)
        self.pub_control_active = self.create_publisher(Bool, "/brov/control_active", 10)
        self.pub_camera_tilt_commanded = self.create_publisher(
            Float32, "/brov/camera_tilt/commanded", 10
        )

        self.srv_start_control = self.create_service(
            Trigger, "/brov/start_control", self._on_start_control
        )
        self.srv_stop_control = self.create_service(
            Trigger, "/brov/stop_control", self._on_stop_control
        )
        self.srv_reset_integrator = self.create_service(
            Trigger, "/brov/reset_integrator", self._on_reset_integrator
        )

        self._ready = False
        self._control_active = False
        # start_control 직후 policy가 이전(frozen) observation으로 계산한 PWM을 보내는
        # 경합을 막는다. 첫 active observation을 발행한 뒤에만 PWM을 허용한다.
        self._active_obs_published = False
        self._faulted = False
        self._last_sample_key = None
        self._last_sample_time = None
        self._last_wp_idx = -1
        self._logged_complete = False
        self._logged_integrator_clamp = False
        self._last_wait_reason = None
        self._last_no_snapshot_log = 0.0
        self._camera_tilt_target_deg = 0.0
        self._camera_tilt_commanded_deg = 0.0
        self._camera_tilt_has_command = False
        self._camera_tilt_last_update = time.monotonic()
        self.timer = self.create_timer(0.04, self._tick)   # 25Hz — envs/vel_env_cfg.py policy_dt와 동일
        self.camera_tilt_timer = self.create_timer(0.05, self._camera_tilt_tick)  # 20Hz
        self.get_logger().info(
            f"연결: {conn}, waypoints({waypoint_frame}, m): {waypoints.tolist()}, "
            f"heading_mode={heading_mode} — "
            "적분/PWM은 /brov/start_control 호출 전까지 동결"
        )
        self.get_logger().info(
            "camera tilt: /brov/camera_tilt/command [-1,1] → "
            f"[{self._camera_tilt_min_deg:.1f}, {self._camera_tilt_max_deg:.1f}]deg, "
            f"rate≤{self._camera_tilt_max_rate_deg_s:.1f}deg/s"
        )

    def _telemetry_valid(self, snap: dict) -> tuple[bool, str]:
        if snap["att_age_s"] >= float(self.get_parameter("att_max_age_s").value):
            return False, f"attitude stale ({snap['att_age_s']:.3f}s)"
        if snap["pos_age_s"] >= float(self.get_parameter("pos_max_age_s").value):
            return False, f"position stale ({snap['pos_age_s']:.3f}s)"
        if snap["ekf_age_s"] >= float(self.get_parameter("ekf_max_age_s").value):
            return False, f"EKF status stale ({snap['ekf_age_s']:.3f}s)"
        if not self.interface.is_ekf_healthy(snap):
            return False, f"EKF unhealthy (velocity_variance={snap['ekf_vel_variance']})"

        tensors = (
            snap["att_quat_ned"], snap["body_rates_ned"],
            snap["pos_ned"], snap["vel_ned"],
        )
        if not all(torch.isfinite(value).all() for value in tensors):
            return False, "telemetry NaN/Inf"
        q_norm = float(snap["att_quat_ned"].norm())
        q_tol = float(self.get_parameter("quat_norm_tolerance").value)
        if abs(q_norm - 1.0) > q_tol:
            return False, f"quaternion norm invalid ({q_norm:.4f})"
        return True, "ok"

    def _publish_raw(self, snap: dict) -> None:
        self.pub_pos.publish(Float32MultiArray(data=snap["pos_ned"].tolist()))
        self.pub_vel.publish(Float32MultiArray(data=snap["vel_ned"].tolist()))
        self.pub_quat.publish(Float32MultiArray(data=snap["att_quat_ned"].tolist()))
        control = self.interface.control_snapshot()
        if control["servo_output_us"] is not None:
            self.pub_servo_output.publish(
                Int32MultiArray(data=control["servo_output_us"].tolist())
            )

    def _trip_fault(self, reason: str) -> None:
        if self._faulted:
            return
        self._faulted = True
        self._control_active = False
        self._active_obs_published = False
        self._camera_tilt_has_command = False
        self.get_logger().error(f"CONTROL FAULT — {reason}; 적분 중지/PWM neutral")
        if self._send_pwm:
            self.interface.neutral_stop()
            self.interface.disarm()

    def _on_start_control(self, _request, response):
        snap = self.interface.snapshot()
        if self._estopped:
            response.success, response.message = False, "estop latched"
            return response
        if self._faulted:
            response.success, response.message = False, "fault latched; reset_integrator 필요"
            return response
        if self._control_active:
            response.success, response.message = True, "control already active"
            return response
        if not self._ready or snap is None:
            response.success, response.message = False, "telemetry not ready"
            return response
        valid, reason = self._telemetry_valid(snap)
        if not valid:
            response.success, response.message = False, reason
            return response
        # 노드 기동 시점이 아니라 실제 제어 시작 시점을 mission-relative (0,0,0)으로
        # 정의한다. 대기 중 기체가 이동해도 경로와 적분항은 여기서 함께 새로 시작한다.
        self.obs_builder.reset(snap["pos_ned"], snap["att_quat_ned"])
        initial_quat = self.obs_builder.attitude_in_waypoint_frame(
            snap["att_quat_ned"]
        ).unsqueeze(0)
        self.guidance.reset(torch.zeros(1, dtype=torch.long), initial_quat=initial_quat)
        initial_yaw_ned_deg = float(
            torch.rad2deg(mu.yaw_from_quat(snap["att_quat_ned"]))
        )
        self._last_wp_idx = -1
        self._logged_complete = False
        self._last_sample_time = max(snap["att_rx_time"], snap["pos_rx_time"])
        self._control_active = True
        self._active_obs_published = False
        self._logged_integrator_clamp = False
        self.get_logger().info(
            "CONTROL ACTIVE — z_v/z_q reset 후 적분/PWM 허용; "
            f"frame={self.obs_builder.waypoint_frame}, initial_yaw_ned={initial_yaw_ned_deg:.1f}deg"
        )
        response.success, response.message = True, "control active; integrators reset"
        return response

    def _on_stop_control(self, _request, response):
        self._control_active = False
        self._active_obs_published = False
        if self._send_pwm:
            self.interface.neutral_stop()
        self.get_logger().info("CONTROL STOPPED — 적분 동결/PWM neutral")
        response.success, response.message = True, "control stopped; integrators frozen"
        return response

    def _on_reset_integrator(self, _request, response):
        self._control_active = False
        self._active_obs_published = False
        self._faulted = False
        self.obs_builder.reset_integrators()
        snap = self.interface.snapshot()
        self._last_sample_time = (
            None if snap is None else max(snap["att_rx_time"], snap["pos_rx_time"])
        )
        if self._send_pwm:
            self.interface.neutral_stop()
        self.get_logger().info("integrator/fault reset — control은 frozen 상태")
        response.success, response.message = True, "integrators reset; control frozen"
        return response

    def _tick(self):
        snap = self.interface.snapshot()
        if snap is None:
            now = time.monotonic()
            if now - self._last_no_snapshot_log >= 2.0:
                self.get_logger().warn(
                    "telemetry 대기 중 — ATTITUDE_QUATERNION/LOCAL_POSITION_NED 미수신"
                )
                self._last_no_snapshot_log = now
            return

        # 새 패킷이 없어도 매 timer tick에서 age를 다시 계산한 snapshot으로 stale을
        # 검사해야, MAVLink가 완전히 끊겼을 때도 active control을 확실히 fault 처리한다.
        self.pub_control_active.publish(Bool(data=self._control_active))
        valid, reason = self._telemetry_valid(snap)
        if not valid:
            reason_kind = reason.split(" (", 1)[0]
            if reason_kind != self._last_wait_reason:
                self.get_logger().warn(f"observation gated: {reason}")
                self._last_wait_reason = reason_kind
            if self._control_active:
                self._trip_fault(reason)
            return
        self._last_wait_reason = None

        sample_key = (snap["att_seq"], snap["pos_seq"])
        if sample_key == self._last_sample_key:
            return   # 같은 MAVLink snapshot을 timer가 다시 읽어도 재적분/재발행하지 않음
        self._last_sample_key = sample_key
        self._publish_raw(snap)

        if not self._ready:
            self.obs_builder.reset(snap["pos_ned"], snap["att_quat_ned"])
            initial_quat = self.obs_builder.attitude_in_waypoint_frame(
                snap["att_quat_ned"]
            ).unsqueeze(0)
            self.guidance.reset(torch.zeros(1, dtype=torch.long), initial_quat=initial_quat)
            self._ready = True
            self._last_sample_time = max(snap["att_rx_time"], snap["pos_rx_time"])
            self.get_logger().info("첫 healthy telemetry 확보 — frozen obs 발행 시작")

        sample_time = max(snap["att_rx_time"], snap["pos_rx_time"])
        dt = 0.0 if self._last_sample_time is None else sample_time - self._last_sample_time
        self._last_sample_time = sample_time
        if dt < 0.0:
            if self._control_active:
                self._trip_fault(f"negative telemetry dt ({dt:.6f}s)")
            return
        max_dt = float(self.get_parameter("max_integration_dt_s").value)
        if self._control_active and dt > max_dt:
            self._trip_fault(f"telemetry dt too large ({dt:.6f}s > {max_dt:.6f}s)")
            return

        obs, debug = self.obs_builder.build(
            snap["att_quat_ned"], snap["body_rates_ned"],
            snap["pos_ned"], snap["vel_ned"], self.guidance,
            dt if self._control_active else 0.0,
            integrate=self._control_active,
            advance_waypoint=self._control_active,
        )
        if obs.shape != (16,) or not torch.isfinite(obs).all():
            if self._control_active:
                self._trip_fault("invalid observation shape or NaN/Inf")
            return
        if debug["integrator_clamped"] and not self._logged_integrator_clamp:
            self._logged_integrator_clamp = True
            self.get_logger().warn("integrator clamp 도달 — 학습 범위 경계에서 유지")
        self.pub_obs.publish(Float32MultiArray(data=obs.tolist()))
        self.pub_pos_mission.publish(Float32MultiArray(data=debug["pos_env"].tolist()))
        self.pub_v_body_zup.publish(
            Float32MultiArray(data=debug["v_body_zup"].tolist())
        )
        self.pub_v_desired_body_zup.publish(
            Float32MultiArray(data=debug["v_d_b_zup"].tolist())
        )
        if self._control_active:
            self._active_obs_published = True

        # 개선 1: 지금 추종 중인 웨이포인트 발행 — guidance.compute()가 위 build() 안에서
        # 이미 _wp_idx를 갱신했으므로 여기서 그 결과를 그대로 읽기만 하면 됨.
        idx = int(self.guidance._wp_idx[0].item())
        target_wp = self.guidance._wp[0, (idx + 1) % self.guidance.num_wp]
        self.pub_target_wp.publish(Float32MultiArray(data=target_wp.tolist()))
        self.pub_wp_idx.publish(Int32(data=idx))
        self.pub_mission_complete.publish(Bool(data=bool(self.guidance.mission_complete[0])))

        if idx != self._last_wp_idx:
            self._last_wp_idx = idx
            self.get_logger().info(f"웨이포인트 전환: idx={idx} → target={target_wp.tolist()}")
        if bool(self.guidance.mission_complete[0]) and not self._logged_complete:
            self._logged_complete = True
            self.get_logger().info(
                "미션 완료 — 마지막 웨이포인트 도달, terminal position hold 유지 중"
            )

    def _on_pwm(self, msg: Float32MultiArray) -> None:
        if (self._estopped or self._faulted or not self._send_pwm
                or not self._control_active or not self._active_obs_published):
            return
        if len(msg.data) != 8:
            self.get_logger().warn(f"pwm 차원 {len(msg.data)} != 8 — 무시")
            return
        pwm = torch.tensor(msg.data, dtype=torch.float32)
        if not torch.isfinite(pwm).all() or bool((pwm.abs() > 1.0).any()):
            self._trip_fault("invalid PWM command (NaN/Inf or outside [-1,1])")
            return
        snap = self.interface.snapshot()
        if snap is None:
            self._trip_fault("PWM received without telemetry")
            return
        valid, reason = self._telemetry_valid(snap)
        if not valid:
            self._trip_fault(f"PWM blocked: {reason}")
            return
        self.interface.send_pwm(pwm)

    def _on_camera_tilt(self, msg: Float32) -> None:
        """정규화된 사용자/RL tilt 목표를 각도 목표로 변환한다."""
        value = float(msg.data)
        if self._estopped or self._faulted:
            self.get_logger().warn("camera tilt 명령 무시 — estop/fault latched")
            return
        if not torch.isfinite(torch.tensor(value)) or not -1.0 <= value <= 1.0:
            self.get_logger().warn(f"camera tilt {value} outside [-1,1] — 무시")
            return
        self._camera_tilt_target_deg = (
            value * self._camera_tilt_max_deg if value >= 0.0
            else -value * self._camera_tilt_min_deg
        )
        self._camera_tilt_has_command = True

    def _camera_tilt_tick(self) -> None:
        """카메라 목표각에 rate limit을 적용해 20Hz mount 명령으로 보낸다."""
        now = time.monotonic()
        dt = min(max(now - self._camera_tilt_last_update, 0.0), 0.2)
        self._camera_tilt_last_update = now
        if not self._camera_tilt_has_command or self._estopped or self._faulted:
            return

        error = self._camera_tilt_target_deg - self._camera_tilt_commanded_deg
        max_step = self._camera_tilt_max_rate_deg_s * dt
        step = max(-max_step, min(max_step, error))
        self._camera_tilt_commanded_deg += step
        normalized = (
            self._camera_tilt_commanded_deg / self._camera_tilt_max_deg
            if self._camera_tilt_commanded_deg >= 0.0
            else -self._camera_tilt_commanded_deg / self._camera_tilt_min_deg
        )
        self.interface.set_camera_tilt(
            normalized,
            min_pitch_deg=self._camera_tilt_min_deg,
            max_pitch_deg=self._camera_tilt_max_deg,
        )
        self.pub_camera_tilt_commanded.publish(Float32(data=float(normalized)))
        if abs(error) <= max_step:
            self._camera_tilt_has_command = False

    def _on_estop(self, _msg: Empty) -> None:
        if self._estopped:
            return   # 이미 트립됨 — 중복 처리 불필요
        self._estopped = True
        self._control_active = False
        self._active_obs_published = False
        self._camera_tilt_has_command = False
        self.get_logger().error("ESTOP 수신 — 즉시 중립 정지 + disarm, 재시작 전까지 재개 안 함")
        if self._send_pwm:
            self.interface.neutral_stop()
            self.interface.disarm()


def main():
    rclpy.init()
    node = ObsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._send_pwm and bool(node.get_parameter("arm").value):
            node.interface.disarm()
        node.interface.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

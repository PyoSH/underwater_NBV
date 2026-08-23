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
        -p heading_mode:=align \
        -p loop:=false \
        -p send_pwm:=true -p arm:=true

`loop`(기본 false): 마지막 웨이포인트 도달 후 처음으로 되돌아가 반복할지(true,
sim의 test_policy.py 평가 스크립트처럼 계속 왕복/순환하고 싶을 때) 여부. false면
마지막 웨이포인트에서 정지(목표속도 0)하고 `/brov/mission_complete`가 True가 됨.

발행 토픽:
    /brov/observation      (Float32MultiArray, 16) — envs/vel_env.py._get_observations()와 동일 규약
    /brov/debug/pos_ned    (Float32MultiArray, 3)  — 원시 LOCAL_POSITION_NED (검증용)
    /brov/debug/vel_ned    (Float32MultiArray, 3)  — 원시 LOCAL_POSITION_NED 속도 (검증용)
    /brov/debug/att_quat_ned (Float32MultiArray, 4) — 원시 ATTITUDE_QUATERNION [w,x,y,z] (검증용)
    /brov/target_waypoint  (Float32MultiArray, 3)  — 지금 추종 중인 목표 웨이포인트(NED, m)
    /brov/waypoint_idx     (Int32)                  — 현재 세그먼트 시작 인덱스(0-base)
    /brov/mission_complete (Bool)                   — loop:=false일 때만 의미 있음, 마지막 웨이포인트 도달 시 True

구독 토픽:
    /brov/thruster_pwm (8,) [-1,1] — policy_node.py가 발행. `send_pwm:=true`일 때만
    실제 RC_CHANNELS_OVERRIDE로 송신(기본 false — obs만 보고 싶을 때 안전하게 끌 수 있음).
    /brov/estop (std_msgs/Empty) — 사용자 입력에 의한 즉시 정지. 아무 메시지나 오면
    트립: 중립(1500us) 즉시 송신 + disarm + 이후 /brov/thruster_pwm 영구 무시(자동
    재개 없음 — 재개하려면 노드를 재시작해야 함, 발산 중 자동 재무장으로 반복 트립되는
    것을 막기 위한 의도적 설계). 트리거 예시:
        ros2 topic pub -1 /brov/estop std_msgs/msg/Empty {}
    자동 발산 감지(다중 기준 트리거)는 아직 미구현 — 지금은 사람이 직접 판단해서
    이 토픽을 쏘는 수동 정지만 있다.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))   # deploy의 부모 — deploy 패키지 하나만 있으면 됨(vendoring)

import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty, Int32, Bool

from deploy.real_robot_interface import RealRobotInterface
from deploy.obs_builder import ObservationBuilder
from deploy.guidance_standalone import LOSGuidance


def _parse_waypoints(s: str) -> torch.Tensor:
    pts = [[float(v) for v in wp.split(",")] for wp in s.split(";")]
    return torch.tensor(pts).unsqueeze(0)


class ObsNode(Node):
    def __init__(self):
        super().__init__("brov_obs_node")
        self.declare_parameter("connection", "udpin:0.0.0.0:14550")
        self.declare_parameter("waypoints", "0,0,0;3,0,0")
        self.declare_parameter("cruise_speed", 0.3)
        self.declare_parameter("heading_mode", "align")
        self.declare_parameter("reach_threshold", 0.5)
        self.declare_parameter("send_pwm", False)
        self.declare_parameter("arm", False)
        self.declare_parameter("loop", False)

        conn = self.get_parameter("connection").value
        waypoints = _parse_waypoints(self.get_parameter("waypoints").value)

        self.interface = RealRobotInterface(conn)
        self.interface.connect()

        self._send_pwm = bool(self.get_parameter("send_pwm").value)
        if self._send_pwm:
            self.interface.enable_passthrough()
            self.get_logger().info("RCPassThru 전환 완료 — /brov/thruster_pwm 수신 시 실제 송신")
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

        self.obs_builder = ObservationBuilder()
        self.guidance = LOSGuidance(
            waypoints, "cpu",
            cruise_speed=self.get_parameter("cruise_speed").value,
            heading_mode=self.get_parameter("heading_mode").value,
            reach_threshold=self.get_parameter("reach_threshold").value,
            loop=bool(self.get_parameter("loop").value),
        )

        self.pub_obs  = self.create_publisher(Float32MultiArray, "/brov/observation", 10)
        self.pub_pos  = self.create_publisher(Float32MultiArray, "/brov/debug/pos_ned", 10)
        self.pub_vel  = self.create_publisher(Float32MultiArray, "/brov/debug/vel_ned", 10)
        self.pub_quat = self.create_publisher(Float32MultiArray, "/brov/debug/att_quat_ned", 10)
        self.pub_target_wp = self.create_publisher(Float32MultiArray, "/brov/target_waypoint", 10)
        self.pub_wp_idx     = self.create_publisher(Int32, "/brov/waypoint_idx", 10)
        self.pub_mission_complete = self.create_publisher(Bool, "/brov/mission_complete", 10)

        self._ready = False
        self._last_t = None
        self._last_wp_idx = -1
        self._logged_complete = False
        self.timer = self.create_timer(0.04, self._tick)   # 25Hz — envs/vel_env_cfg.py policy_dt와 동일
        self.get_logger().info(f"연결: {conn}, waypoints(NED, m): {waypoints.tolist()}")

    def _tick(self):
        snap = self.interface.snapshot()
        if snap is None:
            return

        if not self._ready:
            self.obs_builder.reset(snap["pos_ned"])
            self.guidance.reset(torch.zeros(1, dtype=torch.long))
            self._ready = True
            self._last_t = time.monotonic()
            self.get_logger().info("첫 텔레메트리 확보 — obs 발행 시작")
            return

        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now

        obs, _debug = self.obs_builder.build(
            snap["att_quat_ned"], snap["body_rates_ned"],
            snap["pos_ned"], snap["vel_ned"], self.guidance, dt,
        )
        self.pub_obs.publish(Float32MultiArray(data=obs.tolist()))
        self.pub_pos.publish(Float32MultiArray(data=snap["pos_ned"].tolist()))
        self.pub_vel.publish(Float32MultiArray(data=snap["vel_ned"].tolist()))
        self.pub_quat.publish(Float32MultiArray(data=snap["att_quat_ned"].tolist()))

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
            self.get_logger().info("미션 완료 — 마지막 웨이포인트 도달, 제자리 정지(v_d=0) 명령 중")

    def _on_pwm(self, msg: Float32MultiArray) -> None:
        if self._estopped or not self._send_pwm:
            return
        if len(msg.data) != 8:
            self.get_logger().warn(f"pwm 차원 {len(msg.data)} != 8 — 무시")
            return
        self.interface.send_pwm(torch.tensor(msg.data, dtype=torch.float32))

    def _on_estop(self, _msg: Empty) -> None:
        if self._estopped:
            return   # 이미 트립됨 — 중복 처리 불필요
        self._estopped = True
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

"""ArduSub EKF 의 로컬 pose 를 **읽기만** 해서 ROS 로 흘린다. 2026-09-10.

왜 brov_base 를 안 쓰는가: Phase 3 이 재는 것은 DVL 이 끊겼을 때의 **INS 표류**이지
제어 성능이 아니다. 제어 스택을 띄우면 (a) 무엇이 표류를 만드는지 섞이고 (b) 실수로
구동할 여지가 생긴다. 이 노드는 MAVLink 를 **수신 전용**으로 열고 아무것도 보내지 않는다
(arm/PWM/mode 명령 경로가 아예 없다).

발행:
  /brov/sim/ekf_local_ned   nav_msgs/Odometry   (frame_id=ekf_ned, child=base_link_frd)
    position/velocity 는 LOCAL_POSITION_NED 그대로 — **NED 이고 FRD 다**. ENU 로 바꾸지
    않는다: 비교 상대인 Gazebo GT 는 ENU 라, 변환은 분석 쪽에서 한 번만 하는 편이
    실수를 줄인다(여기서 돌려두면 어느 쪽이 이미 변환됐는지 헷갈린다).
  /brov/sim/ekf_status      std_msgs/String     EKF_STATUS_REPORT 요약
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MAVLINK20", "1")

import rclpy
from nav_msgs.msg import Odometry
from pymavlink import mavutil
from rclpy.node import Node
from std_msgs.msg import String


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--connection", default="udpin:0.0.0.0:14552")
    ap.add_argument("--confirm-sitl", action="store_true", required=True,
                    help="이 엔드포인트가 SITL 임을 명시 확인")
    args, _ = ap.parse_known_args()
    if not args.connection.startswith("udpin:"):
        raise SystemExit("수신 전용 udpin: 엔드포인트만 허용한다")

    rclpy.init()
    node = Node("nbv_mavlink_ekf_probe")
    pub = node.create_publisher(Odometry, "/brov/sim/ekf_local_ned", 20)
    pub_st = node.create_publisher(String, "/brov/sim/ekf_status", 5)
    link = mavutil.mavlink_connection(args.connection, source_system=200,
                                      source_component=200, input=True)
    node.get_logger().info(f"수신 전용 {args.connection} — 아무것도 보내지 않는다")

    seen = 0
    try:
        while rclpy.ok():
            msg = link.recv_match(
                type=["LOCAL_POSITION_NED", "EKF_STATUS_REPORT"], blocking=True,
                timeout=1.0)
            if not rclpy.ok():
                break
            if msg is None:
                continue
            if msg.get_type() == "LOCAL_POSITION_NED":
                o = Odometry()
                o.header.stamp = node.get_clock().now().to_msg()
                o.header.frame_id = "ekf_ned"
                o.child_frame_id = "base_link_frd"
                o.pose.pose.position.x = float(msg.x)
                o.pose.pose.position.y = float(msg.y)
                o.pose.pose.position.z = float(msg.z)
                o.pose.pose.orientation.w = 1.0
                o.twist.twist.linear.x = float(msg.vx)
                o.twist.twist.linear.y = float(msg.vy)
                o.twist.twist.linear.z = float(msg.vz)
                pub.publish(o)
                seen += 1
                if seen % 250 == 1:
                    node.get_logger().info(
                        f"LOCAL_POSITION_NED #{seen}  "
                        f"n={msg.x:+.3f} e={msg.y:+.3f} d={msg.z:+.3f}")
            else:
                pub_st.publish(String(data=(
                    f"flags={msg.flags} vel={msg.velocity_variance:.3f} "
                    f"posh={msg.pos_horiz_variance:.3f} posv={msg.pos_vert_variance:.3f} "
                    f"compass={msg.compass_variance:.3f}")))
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 신호와 spin 이 겹치면 rclpy 가 "context is not valid" 로 죽는다.
        # 측정 로그에 가짜 트레이스백을 남기지 않도록 조용히 닫는다.
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

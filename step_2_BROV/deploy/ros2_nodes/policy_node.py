#!/usr/bin/env python3
"""
정책 추론 전용 ROS2 노드 — obs_node.py와 분리, MAVLink 접속 없음(순수 계산 노드).
`/brov/observation`을 구독해 정책 추론(TorchScript) + action→PWM 변환까지만
수행하고 결과를 발행한다.

MAVLink 송신(액추에이션)은 이 노드가 아니라 obs_node.py가 담당한다 —
`udpout:127.0.0.1:14550`으로 이 노드가 별도 연결을 시도했더니, 이 SITL의
MAVProxy가 GCS 쪽에서 먼저 `udpin`으로 bind/listen하는 걸 전제로 하는 구성이라
(obs_node가 이미 그렇게 붙어 있음), 두 번째 `udpout` 연결은 응답을 전혀 못
받고 걸렸다(실측 확인). 그래서 obs_node가 `/brov/thruster_pwm`을 구독해 자기
연결로 대신 보내는 구조로 정리했다(ros2_nodes/obs_node.py 참조) — obs/action/pwm
3개 토픽은 이 구조에서도 전부 그대로 `ros2 topic echo`로 확인 가능하다.

실행(컨테이너 안, ROS2 소싱 후):
    python3 policy_node.py --ros-args -p policy_path:=/path/to/policy.pt

발행 토픽 (전부 std_msgs/Float32MultiArray):
    /brov/action         (6,)  — [-1,1], surge/sway/heave/roll/pitch/yaw
    /brov/thruster_pwm   (8,)  — [-1,1] (1500±400us로 변환 전 정규화 값)

action은 매 obs 콜백(보통 25Hz)마다 나오는데 `ros2 topic echo`로는 숫자만 빠르게
스크롤돼서 읽기 어렵다 — `vis_rate_hz`(기본 2Hz)마다 6축을 막대그래프로 터미널에
로그 출력한다(`get_logger().info()`, ROS 로그와 동일 경로라 별도 설정 불필요).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))   # deploy의 부모 — deploy 패키지 하나만 있으면 됨(vendoring)

import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from deploy.policy_runner import PolicyRunner
from deploy.vendor.thruster import BROV2ThrusterModel, build_allocation_matrix
from deploy.vendor.params import load_brov2_yaml, thruster_pos_dir_ned

# envs/vel_env_cfg.py의 f_max와 동일하게 유지 (deploy 코드가 IsaacLab 의존성을 가질 수 없어 부득이하게 복제 — CLAUDE.md 참조)
_F_MAX = torch.tensor([85.0, 85.0, 120.0, 26.0, 14.0, 22.0])

_ACTION_LABELS = ["surge", "sway ", "heave", "roll ", "pitch", "yaw  "]
_BAR_WIDTH = 20


def _action_bar(label: str, value: float, width: int = _BAR_WIDTH) -> str:
    """[-1,1] 값을 가운데(0) 기준 막대그래프 문자열로. 예: 'surge [---------##--------] +0.21'"""
    v = max(-1.0, min(1.0, value))
    mid = width // 2
    filled = int(round(abs(v) * mid))
    chars = ["-"] * width
    chars[mid] = "|"
    if v >= 0:
        for i in range(filled):
            chars[mid + i] = "#"
    else:
        for i in range(filled):
            chars[mid - 1 - i] = "#"
    return f"{label} [{''.join(chars)}] {value:+.2f}"


class PolicyNode(Node):
    def __init__(self):
        super().__init__("brov_policy_node")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("obs_rate_hz", 25.0)   # obs_node의 timer 주기와 맞출 것 — 시각화 스로틀 계산용
        self.declare_parameter("vis_rate_hz", 2.0)
        policy_path = self.get_parameter("policy_path").value
        if not policy_path:
            raise ValueError("policy_path 파라미터 필요 (export_policy.py가 만든 policy.pt)")

        obs_rate_hz = float(self.get_parameter("obs_rate_hz").value)
        vis_rate_hz = float(self.get_parameter("vis_rate_hz").value)
        self._vis_every_n = max(1, round(obs_rate_hz / vis_rate_hz))
        self._obs_count = 0

        self.policy = PolicyRunner(policy_path, device="cpu")

        yaml_params = load_brov2_yaml()
        thr_pos, thr_dir = thruster_pos_dir_ned(yaml_params)
        self.thruster = BROV2ThrusterModel(num_envs=1, dt=0.04, device="cpu", pos=thr_pos, dir=thr_dir)
        B = build_allocation_matrix(self.thruster._pos, self.thruster._dir)
        self.B_pinv = torch.linalg.pinv(B)

        self.pub_action = self.create_publisher(Float32MultiArray, "/brov/action", 10)
        self.pub_pwm    = self.create_publisher(Float32MultiArray, "/brov/thruster_pwm", 10)
        self.sub_obs = self.create_subscription(Float32MultiArray, "/brov/observation", self._on_obs, 10)
        self.get_logger().info(f"정책 로드 완료: {policy_path}")

    def _on_obs(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != 16:
            self.get_logger().warn(f"obs 차원 {len(msg.data)} != 16 — 무시")
            return
        obs = torch.tensor(msg.data, dtype=torch.float32)

        action = self.policy.act(obs)                                       # (6,), [-1,1]
        tau_cmd = _F_MAX * action
        f_desired = (self.B_pinv @ tau_cmd.unsqueeze(-1)).squeeze(-1)        # (8,)
        pwm = self.thruster.inverse_thrust(f_desired.unsqueeze(0)).squeeze(0)  # (8,), [-1,1]

        self.pub_action.publish(Float32MultiArray(data=action.tolist()))
        self.pub_pwm.publish(Float32MultiArray(data=pwm.tolist()))

        # 개선 2: 터미널에서 action을 막대그래프로 시각화 (vis_rate_hz로 스로틀 — 매 콜백마다
        # 찍으면 25Hz로 터미널이 흘러가버려서 못 읽음)
        self._obs_count += 1
        if self._obs_count % self._vis_every_n == 0:
            lines = [_action_bar(lbl, v.item()) for lbl, v in zip(_ACTION_LABELS, action)]
            self.get_logger().info("action:\n" + "\n".join(lines))


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

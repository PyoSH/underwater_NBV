#!/usr/bin/env python3
"""정책을 우회하는 waypoint 추종용 model-based controller ROS2 노드.

``obs_node.py``가 발행하는 16차원 observation을 받아 명시적 PI/PD wrench와
thruster PWM을 계산한다. 기본적으로 preview만 발행하며, 다음 두 조건이 만족된
뒤 ``/brov/model_based/start`` 서비스를 호출해야 실제 ``/brov/thruster_pwm``을
발행한다.

1. ``/brov/control_active``가 True (obs_node start_control 완료)
2. policy_node 등 다른 PWM publisher가 없음

토픽:
  /brov/model_based/wrench_zup          (6,) [N, Nm]
  /brov/model_based/wrench_sname        (6,) [N, Nm]
  /brov/model_based/estimated_wrench_zup (6,) [N, Nm]
  /brov/model_based/action              (6,) wrench/f_max
  /brov/model_based/thruster_force      (8,) [N]
  /brov/model_based/thruster_pwm_preview (8,) [-1,1]
  /brov/model_based/enabled             Bool
  /brov/thruster_pwm                    (8,) 실제 출력, enabled일 때만
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import rclpy
import torch
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import Trigger

from deploy.model_based_controller import ModelBasedController
from deploy.vendor.params import load_brov2_yaml, thruster_pos_dir_ned


def _array3(node: Node, name: str) -> list[float]:
    value = list(node.get_parameter(name).value)
    if len(value) != 3:
        raise ValueError(f"{name}은 3개 값이어야 함")
    return [float(item) for item in value]


class ModelBasedControllerNode(Node):
    def __init__(self):
        super().__init__("brov_model_based_controller")
        self.declare_parameter("linear_kp", [25.0, 25.0, 35.0])
        self.declare_parameter("linear_ki", [0.0, 0.0, 0.0])
        self.declare_parameter("attitude_kp", [3.0, 3.0, 3.0])
        self.declare_parameter("attitude_ki", [0.0, 0.0, 0.0])
        self.declare_parameter("angular_kd", [1.5, 1.5, 1.0])
        self.declare_parameter("force_limit", [15.0, 15.0, 20.0])
        self.declare_parameter("torque_limit", [3.0, 3.0, 3.0])
        self.declare_parameter("minimum_active_pwm", 0.10)
        self.declare_parameter("thruster_force_activation", 0.25)
        self.declare_parameter("observation_timeout_s", 0.25)

        params = load_brov2_yaml()
        thruster_pos, thruster_dir = thruster_pos_dir_ned(params)
        self.controller = ModelBasedController(
            thruster_pos,
            thruster_dir,
            linear_kp=_array3(self, "linear_kp"),
            linear_ki=_array3(self, "linear_ki"),
            attitude_kp=_array3(self, "attitude_kp"),
            attitude_ki=_array3(self, "attitude_ki"),
            angular_kd=_array3(self, "angular_kd"),
            force_limit=_array3(self, "force_limit"),
            torque_limit=_array3(self, "torque_limit"),
            minimum_active_pwm=float(self.get_parameter("minimum_active_pwm").value),
            thruster_force_activation=float(
                self.get_parameter("thruster_force_activation").value
            ),
        )
        self._timeout = float(self.get_parameter("observation_timeout_s").value)
        if self._timeout <= 0.0:
            raise ValueError("observation_timeout_s는 양수여야 함")

        self._enabled = False
        self._control_active = False
        self._last_obs_time: float | None = None
        self._last_output = None

        self.pub_wrench_zup = self.create_publisher(
            Float32MultiArray, "/brov/model_based/wrench_zup", 10
        )
        self.pub_wrench_sname = self.create_publisher(
            Float32MultiArray, "/brov/model_based/wrench_sname", 10
        )
        self.pub_action = self.create_publisher(
            Float32MultiArray, "/brov/model_based/action", 10
        )
        self.pub_estimated_wrench = self.create_publisher(
            Float32MultiArray, "/brov/model_based/estimated_wrench_zup", 10
        )
        self.pub_force = self.create_publisher(
            Float32MultiArray, "/brov/model_based/thruster_force", 10
        )
        self.pub_preview = self.create_publisher(
            Float32MultiArray, "/brov/model_based/thruster_pwm_preview", 10
        )
        self.pub_enabled = self.create_publisher(Bool, "/brov/model_based/enabled", 10)
        self.pub_pwm = self.create_publisher(Float32MultiArray, "/brov/thruster_pwm", 10)

        self.sub_obs = self.create_subscription(
            Float32MultiArray, "/brov/observation", self._on_observation, 10
        )
        self.sub_active = self.create_subscription(
            Bool, "/brov/control_active", self._on_control_active, 10
        )
        self.srv_start = self.create_service(
            Trigger, "/brov/model_based/start", self._on_start
        )
        self.srv_stop = self.create_service(
            Trigger, "/brov/model_based/stop", self._on_stop
        )
        self.timer = self.create_timer(0.05, self._safety_tick)
        self.get_logger().info(
            "model-based controller ready — preview only; "
            "/brov/start_control 후 /brov/model_based/start 필요"
        )

    def _other_pwm_publishers(self) -> list[str]:
        others = []
        for info in self.get_publishers_info_by_topic("/brov/thruster_pwm"):
            if info.node_name != self.get_name() or info.node_namespace != self.get_namespace():
                others.append(f"{info.node_namespace.rstrip('/')}/{info.node_name}")
        return others

    def _publish_enabled(self) -> None:
        self.pub_enabled.publish(Bool(data=self._enabled))

    def _publish_neutral(self) -> None:
        self.pub_pwm.publish(Float32MultiArray(data=[0.0] * 8))

    def _disable(self, reason: str, *, warn: bool = True) -> None:
        was_enabled = self._enabled
        self._enabled = False
        if was_enabled:
            self._publish_neutral()
            message = f"MODEL CONTROL STOPPED — {reason}; neutral published"
            # rclpy Humble은 같은 소스 위치에서 logger severity가 바뀌면 예외를 낸다.
            if warn:
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)
        self._publish_enabled()

    def _on_control_active(self, msg: Bool) -> None:
        self._control_active = bool(msg.data)
        if self._enabled and not self._control_active:
            self._disable("obs control inactive")

    def _on_observation(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != 16:
            if self._enabled:
                self._disable(f"invalid observation dimension {len(msg.data)}")
            return
        try:
            output = self.controller.compute(torch.tensor(msg.data, dtype=torch.float32))
        except ValueError as exc:
            if self._enabled:
                self._disable(str(exc))
            return
        self._last_obs_time = time.monotonic()
        self._last_output = output
        self.pub_wrench_zup.publish(Float32MultiArray(data=output.wrench_zup.tolist()))
        self.pub_wrench_sname.publish(Float32MultiArray(data=output.wrench_sname.tolist()))
        self.pub_action.publish(Float32MultiArray(data=output.normalized_action_zup.tolist()))
        self.pub_estimated_wrench.publish(
            Float32MultiArray(data=output.estimated_wrench_zup.tolist())
        )
        self.pub_force.publish(Float32MultiArray(data=output.thruster_force.tolist()))
        self.pub_preview.publish(Float32MultiArray(data=output.pwm.tolist()))
        if self._enabled:
            self.pub_pwm.publish(Float32MultiArray(data=output.pwm.tolist()))

    def _on_start(self, _request, response):
        if self._enabled:
            response.success, response.message = True, "model control already active"
            return response
        if not self._control_active:
            response.success, response.message = False, "obs control is not active"
            return response
        if self._last_obs_time is None or time.monotonic() - self._last_obs_time >= self._timeout:
            response.success, response.message = False, "fresh observation unavailable"
            return response
        others = self._other_pwm_publishers()
        if others:
            response.success = False
            response.message = f"other PWM publisher exists: {', '.join(others)}"
            return response
        self._enabled = True
        self._publish_enabled()
        self.get_logger().info("MODEL CONTROL ACTIVE — explicit PI/PD wrench → PWM")
        response.success, response.message = True, "model control active"
        return response

    def _on_stop(self, _request, response):
        self._disable("stop service", warn=False)
        response.success, response.message = True, "model control stopped; neutral published"
        return response

    def _safety_tick(self) -> None:
        self._publish_enabled()
        if not self._enabled:
            return
        if self._last_obs_time is None or time.monotonic() - self._last_obs_time >= self._timeout:
            self._disable("observation watchdog timeout")
            return
        others = self._other_pwm_publishers()
        if others:
            self._disable(f"competing PWM publisher: {', '.join(others)}")

    def shutdown(self) -> None:
        was_enabled = self._enabled
        self._disable("node shutdown", warn=False)
        if was_enabled:
            # Ctrl+C 직후 publisher를 바로 파괴하면 마지막 DDS sample 전달 전에
            # 프로세스가 끝날 수 있으므로 neutral을 짧게 반복한다.
            for _ in range(2):
                time.sleep(0.05)
                self._publish_neutral()


def main():
    # Humble 기본 SIGINT handler는 spin()을 빠져나오기 전에 context를 shutdown해
    # finally의 neutral publish를 불가능하게 한다. Python의 KeyboardInterrupt를
    # 사용해 context가 살아 있는 동안 neutral을 먼저 발행한다.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ModelBasedControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

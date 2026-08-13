# Model-based waypoint control

`model_based_controller_node.py`는 학습 정책을 사용하지 않는다. `obs_node.py`의
16차원 observation에 명시적 PI/PD 음의 피드백을 적용하고, BlueROV2 allocation
matrix와 T200 역모델로 PWM을 계산한다.

## 안전한 preview

`policy_node.py`를 실행하지 않는다. 첫 터미널에서 실제 출력을 끈 obs node를 실행한다.

```bash
cd /workspace/deploy/ros2_nodes
python3 obs_node.py --ros-args \
  -p connection:=udpout:192.168.2.2:14550 \
  -p waypoints:="0,0,0;0.5,0,0" \
  -p waypoint_frame:=start_heading \
  -p heading_mode:=straight \
  -p loop:=false \
  -p cruise_speed:=0.1 \
  -p send_pwm:=false \
  -p arm:=false
```

두 번째 터미널에서 controller를 실행한다.

```bash
cd /workspace/deploy/ros2_nodes
python3 model_based_controller_node.py
```

아직 `/brov/model_based/start`를 호출하지 않아도 preview는 발행된다.

```bash
ros2 topic echo /brov/model_based/wrench_zup
ros2 topic echo /brov/model_based/thruster_pwm_preview
```

## 실제 저속 시험

로봇을 수중에서 안전하게 고정한 뒤 obs node의 `send_pwm`, `arm`을 true로 바꾼다.
`policy_node.py`는 반드시 종료된 상태여야 한다.

```bash
ros2 service call /brov/start_control std_srvs/srv/Trigger "{}"
ros2 service call /brov/model_based/start std_srvs/srv/Trigger "{}"
```

정상 정지 순서:

```bash
ros2 service call /brov/model_based/stop std_srvs/srv/Trigger "{}"
ros2 service call /brov/stop_control std_srvs/srv/Trigger "{}"
```

비상 정지:

```bash
ros2 topic pub --once /brov/estop std_msgs/msg/Empty "{}"
```

기본 gain과 wrench limit은 보수적으로 설정되어 있으며 ROS parameter로 변경할 수
있다. 첫 실제 시험에서는 integral gain을 기본값 0으로 유지한다.

```text
linear_kp    = [25, 25, 35] N/(m/s)
attitude_kp  = [3, 3, 3] Nm/rad
angular_kd   = [1.5, 1.5, 1.0] Nm/(rad/s)
force_limit  = [15, 15, 20] N
torque_limit = [3, 3, 3] Nm
linear_ki = attitude_ki = 0
minimum_active_pwm = 0.10
thruster_force_activation = 0.25 N
```

T200 모델은 `|PWM|<=0.075`에서 추력이 0이다. 작은 요구 추력을 deadband 내부
PWM으로 보내는 것을 방지하기 위해, 절댓값 0.25 N 이상의 thruster 요구는 최소
`|PWM|=0.10`으로 보상한다. `/brov/model_based/estimated_wrench_zup`에서 이
deadband 보상과 1차 thruster lag를 반영한 예상 wrench를 확인할 수 있다.

노드는 stale observation, inactive obs control 또는 경쟁 PWM publisher를 발견하면
model control을 중지하고 neutral PWM을 한 번 발행한다.

`obs_node.py`의 guidance는 수평 LOS와 depth hold를 분리한다. 각 구간에서 다음
waypoint의 Z를 `depth_hold_kp` position outer-loop로 계속 추종하며, 마지막
waypoint 도달 후에도 `terminal_hold_kp`로 3D 위치를 유지한다.

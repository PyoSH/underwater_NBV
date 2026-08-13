# BlueROV2 Sim2Real 데모 실행 절차

## 1. 데모 범위

권장 데모 순서:

1. BlueOS MAVLink 및 카메라 연결 확인
2. ROS 2 camera image 발행
3. DVL/EKF observation과 waypoint 시각화
4. Model-based waypoint 추종
5. 정상 정지 및 설정 복원 확인

RL policy 제어는 별도 실험 단계다. `policy_node`와
`model_based_controller_node`를 동시에 실행하면 안 된다.

## 2. 실험 전 안전 조건

- 로봇이 수중에 있고 프로펠러 주변이 비어 있어야 한다.
- tether가 추진기와 얽히지 않고 상승을 방해하지 않아야 한다.
- 즉시 전원 차단 또는 `/brov/estop`을 실행할 담당자가 있어야 한다.
- QGroundControl과 이 deploy pipeline을 동시에 MAVLink GCS로 사용하지 않는다.
- ArduSub mode는 `MANUAL(19)`이어야 한다.
- DVL ExternalNav와 `LOCAL_POSITION_NED`가 정상이어야 한다.
- BlueOS MAVLink server는 `192.168.2.2:14550`에서 접근 가능해야 한다.
- camera endpoint는 Mac의 실제 tether IP와 UDP 5600을 가리켜야 한다.

비상정지 명령은 모든 터미널에 준비해 둔다.

```bash
ros2 topic pub --once /brov/estop std_msgs/msg/Empty "{}"
```

## 3. Docker 및 네트워크

Docker Desktop 설정:

```text
Enable host networking:       ON
Use kernel networking for UDP: OFF
```

Mac 프로젝트 디렉터리에서:

```bash
make ros-build
docker compose ps
make shell
```

컨테이너 환경 검사:

```bash
cd /workspace/deploy
python3 docker/check_environment.py
```

## 4. 카메라 실행

BlueOS 기준 설정:

```text
source:   /dev/video2
format:   640×480 @ 30 fps
encoding: H264/RTP
endpoint: udp://<Mac tether IP>:5600
```

터미널 A에서는 package launch를 사용한다.

```bash
ros2 launch brov_bringup camera.launch.py \
  udp_port:=5600 \
  camera_info_path:=/workspace/deploy/config/camera_intrinsics.yaml
```

확인:

```bash
ros2 topic hz /brov/camera/image_raw
rqt_image_view /brov/camera/image_raw
```

현재 `config/camera_intrinsics.yaml`은 미보정 상태다. ArUco metric pose 데모 전에는
실제 하우징·수중·640×480 조건에서 intrinsic calibration을 완료해야 한다.

## 5. Model-based waypoint 데모

### 5.1 Obs/MAVLink + controller bringup

터미널 B에서 두 노드를 함께 실행한다. launch만으로는 제어 service를 시작하지 않는다.

```bash
ros2 launch brov_bringup model_demo.launch.py \
  connection:=udpout:192.168.2.2:14550 \
  send_pwm:=true \
  arm:=true
```

기본 경로와 gain은 각각 `mission_demo.yaml`, `model_controller.yaml`에 있다. 다른
실험값은 YAML을 복사해 수정한 뒤 `mission_file:=/절대/경로/mission.yaml` 및
`controller_config:=/절대/경로/controller.yaml`로 전달한다.

이 노드는 시작 시 다음 작업을 수행하므로 수 초가 걸릴 수 있다.

- SERVO1~8 function 백업
- RC7/RC8 camera option 백업 및 격리
- SERVO1~8 RCPassThru 전환/read-back
- arm
- MAVLink telemetry 요청

다음 로그가 나온 뒤 계속한다.

```text
arm 완료
첫 healthy telemetry 확보 — frozen obs 발행 시작
```

### 5.2 Preview 확인

Model controller는 launch 직후 실제 PWM을 발행하지 않고 preview만 계산한다.

```bash
ros2 topic echo --once /brov/model_based/wrench_zup
ros2 topic echo --once /brov/model_based/thruster_pwm_preview
```

### 5.3 제어 시작

별도 터미널 C:

```bash
ros2 service call /brov/start_control std_srvs/srv/Trigger "{}"
ros2 service call /brov/model_based/start std_srvs/srv/Trigger "{}"
```

두 응답 모두 `success=True`여야 한다.

### 5.4 실시간 확인

최소 확인 항목:

```bash
ros2 topic echo /brov/debug/pos_mission
ros2 topic echo /brov/debug/v_desired_body_zup
ros2 topic echo /brov/model_based/wrench_zup
ros2 topic echo /brov/debug/servo_output_us
```

기대 동작:

- 첫 구간에서 `pos_mission.z`가 `0 → -0.5`로 이동한다.
- 수평 구간에서도 독립 depth outer-loop가 `z≈-0.5`를 유지한다.
- 마지막 waypoint 도달 후 terminal position hold가 계속 동작한다.
- `/brov/debug/servo_output_us`가 명령 중 1500에서 벗어나고 정지 시 1500으로 돌아온다.

## 6. RL policy 실험 모드

Model-based controller를 완전히 종료한 뒤에만 실행한다.

```bash
ros2 node list | grep brov_model_based_controller
```

출력이 없어야 한다. RL 전용 launch를 사용한다.

```bash
ros2 launch brov_bringup rl_demo.launch.py \
  policy_path:=/workspace/deploy/exported/policy.pt \
  connection:=udpout:192.168.2.2:14550 \
  send_pwm:=true \
  arm:=true
```

정책은 별도 start service가 없다. `obs_node`의 control gate만 연다.

```bash
ros2 service call /brov/start_control std_srvs/srv/Trigger "{}"
```

주의:

- observation의 `q_e`는 `w>=0` hemisphere로 수정됐지만 실기체 policy 재검증이 남아 있다.
- 기존 policy에서 zero-state sway bias와 높은 pitch gain이 실측됐다.
- 첫 RL 시험은 로봇 고정·낮은 waypoint speed·즉시 estop 조건에서 수행한다.

## 7. 정상 정지

Model-based 제어:

```bash
ros2 service call /brov/model_based/stop std_srvs/srv/Trigger "{}"
ros2 service call /brov/stop_control std_srvs/srv/Trigger "{}"
```

RL 제어:

```bash
ros2 service call /brov/stop_control std_srvs/srv/Trigger "{}"
```

그 다음 controller 노드를 `Ctrl+C`로 종료하고, 마지막으로 `obs_node`를 정상적으로
`Ctrl+C` 종료한다. `obs_node` 정상 종료가 disarm, RC override release,
SERVO function 및 RC7/RC8 option 복원을 담당한다.

종료 확인:

```bash
ros2 node list | grep '^/brov_'
```

비정상 종료나 Docker 재시작이 있었다면 다음 파라미터를 반드시 확인한다.

```text
SERVO1~8_FUNCTION
RC7_OPTION
RC8_OPTION
armed state
```

## 8. 데모 실패 시 빠른 판정

| 증상 | 우선 확인 |
|---|---|
| heartbeat timeout | BlueOS endpoint, Mac tether IP, Docker host network |
| telemetry not ready | DVL ExternalNav, ATTITUDE_QUATERNION, LOCAL_POSITION_NED |
| observation stale/fault | MAVLink 주기와 UDP 지터, `max_integration_dt_s` |
| PWM topic은 있으나 모터 무반응 | `servo_output_us`, RCPassThru, arm, deadband |
| 모터는 돌지만 이동 방향 오류 | T1~T8 채널/반전 mask, tether 구속 |
| 수평 구간에서 하강 | `v_desired_body_zup.z`, depth outer-loop parameter |
| 카메라 영상 없음 | Mac UDP 5600 수신, BlueOS endpoint, camera node 위치 |
| ArUco 거리가 틀림 | intrinsic 파일, 해상도, marker 실제 길이 |

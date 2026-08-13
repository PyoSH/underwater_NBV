# 개발 현황 및 아키텍처

## 1. 시스템 범위

현재 deploy는 다음 환경을 대상으로 한다.

```text
Host:       Apple Silicon MacBook
Runtime:    Docker Desktop, linux/arm64
Container:  Ubuntu 22.04, ROS 2 Humble desktop
Vehicle:    BlueROV2 Heavy + ArduSub/BlueOS + DVL
Camera:     Blue Robotics Low-Light HD USB Camera, H264/RTP
Control:    25 Hz observation/controller
```

Docker bind mount는 Mac의 deploy 디렉터리를 `/workspace/deploy`에 연결하므로 양쪽의
파일 수정은 즉시 같은 파일에 반영된다.

## 2. 데이터 흐름

```text
BlueOS/ArduSub
  ├─ MAVLink heartbeat/attitude/local position/EKF/servo output
  │      ↓
  │  RealRobotInterface
  │      ↓
  │  obs_node ── guidance ── ObservationBuilder ── /brov/observation
  │      ↑                                           │
  │      │                         ┌─────────────────┴─────────────┐
  │      │                         ↓                               ↓
  │      │                  policy_node                 model-based PI/PD
  │      │                         └──────── /brov/thruster_pwm ───┘
  │      └──────── RC_CHANNELS_OVERRIDE ← PWM reversal/deadband/allocation
  │
  └─ H264/RTP UDP 5600
         ↓
     camera_stream_node ── Image/CameraInfo
         ├─ checkerboard calibration
         └─ ArUco pose/TF
```

## 3. 좌표계 계약

### MAVLink 입력

```text
World: NED    X North, Y East, Z Down
Body:  FRD    X Forward, Y Right, Z Down
Quaternion:   [w,x,y,z], body → NED world
```

### Mission frame

`waypoint_frame=start_heading`은 `/brov/start_control` 순간의 위치를 원점으로 하고,
초기 yaw 방향을 mission +X로 정의한다. roll/pitch와 NED Z 방향은 유지한다.

```text
waypoint Z < 0: 상승
waypoint Z > 0: 하강
```

### Policy observation frame

학습 환경에 맞춰 body 값을 FLU/Z-up으로 변환한다.

```text
T3 = diag(1,-1,-1)
```

16차원 observation:

| Index | 값 | 정의 |
|---|---|---|
| 0:4 | `q_e` | desired 대비 current 자세 오차 quaternion |
| 4:7 | `v_e_b` | current body velocity − desired body velocity |
| 7:10 | `omega_b` | body angular velocity |
| 10:13 | `z_v` | velocity error integral |
| 13:16 | `z_q` | quaternion vector error integral |

`q_e`는 동일 회전의 `q/-q` 불연속을 막기 위해 항상 `w>=0` 표현을 사용한다.

### Wrench와 PWM

RL action/model controller wrench는 Z-up body 기준이다. Thruster allocation matrix는
SNAME/FRD이므로 allocation 전에 힘·토크 부호를 변환한다. RCPassThru는 ArduSub의
`MOT_n_DIRECTION`을 우회하므로 최종 송신 단계에서 실측 반전 mask를 적용한다.

```text
T1~T8 reversal = [+1,-1,-1,+1,+1,+1,+1,-1]
```

## 4. 주요 모듈

| 파일 | 역할 |
|---|---|
| `real_robot_interface.py` | MAVLink 연결, telemetry, parameter, arm/disarm, passthrough, PWM, camera tilt |
| `obs_builder.py` | NED/FRD → mission/Z-up 변환, 16차원 observation, 적분 상태 |
| `guidance_standalone.py` | LOS, 독립 depth hold, terminal 3D position hold |
| `ros2_nodes/obs_node.py` | telemetry gate, lifecycle service, debug topic, PWM 송신 |
| `ros2_nodes/policy_node.py` | TorchScript policy inference, action→allocation→PWM |
| `model_based_controller.py` | 명시적 velocity PI + attitude/rate PD, deadband-aware PWM |
| `ros2_nodes/model_based_controller_node.py` | preview/enable/watchdog/publisher 충돌 방지 |
| `vendor/thruster.py` | T200 forward/inverse model과 allocation matrix |
| `diag_thruster_map.py` | neutral, 전기 채널, 실제 운동 방향 진단 |
| `ros2_nodes/camera_stream_node.py` | BlueOS H264/RTP decode, ROS Image/CameraInfo |
| `ros2_nodes/checkerboard_calibration_node.py` | 자동 checkerboard 표본 수집 및 intrinsic 계산 |
| `ros2_nodes/aruco_pose_node.py` | marker pose 및 선택적 robot pose/TF |

### ROS 2 패키지 계층

| 패키지 | 실행 파일/책임 |
|---|---|
| `brov_base` | `obs_node`, MAVLink/observation/guidance 호환 API |
| `brov_control` | `policy_node`, `model_based_controller_node` |
| `brov_perception` | camera, checkerboard calibration, ArUco nodes |
| `brov_bringup` | vehicle/mission/safety/controller/camera YAML과 launch |

패키지 소스는 `ros2_ws/src`에 있다. 현재는 검증된 구현을 중복하지 않기 위해 console
entry point가 기존 `deploy.*` 모듈을 호출하는 1차 이행 구조다. 따라서 이 버전은
Compose의 `/workspace/deploy` bind mount와 `PYTHONPATH=/workspace`가 필요하며, 임의의
독립 ROS 설치 경로로 복사해 사용하는 단계는 아니다.

## 5. ROS 인터페이스

### 핵심 입력/출력

| 토픽 | 타입 | 의미 |
|---|---|---|
| `/brov/observation` | Float32MultiArray(16) | policy/model controller 공통 입력 |
| `/brov/thruster_pwm` | Float32MultiArray(8) | 논리 normalized PWM |
| `/brov/action` | Float32MultiArray(6) | RL action |
| `/brov/target_waypoint` | Float32MultiArray(3) | mission frame 목표점 |
| `/brov/waypoint_idx` | Int32 | 현재 세그먼트 시작 index |
| `/brov/mission_complete` | Bool | 마지막 waypoint 도달 이력 |
| `/brov/control_active` | Bool | obs 적분/PWM gate 상태 |
| `/brov/estop` | Empty | neutral + disarm, latch |

### Debug

| 토픽 | 의미 |
|---|---|
| `/brov/debug/att_quat_ned` | ArduSub 절대 자세 quaternion |
| `/brov/debug/pos_ned` | raw LOCAL_POSITION_NED |
| `/brov/debug/vel_ned` | raw NED velocity |
| `/brov/debug/pos_mission` | start-control 기준 mission 위치 |
| `/brov/debug/v_body_zup` | 실제 body velocity, policy frame |
| `/brov/debug/v_desired_body_zup` | guidance 목표 body velocity |
| `/brov/debug/servo_output_us` | 실제 ArduSub SERVO1~8 output |

### Services

```text
/brov/start_control
/brov/stop_control
/brov/reset_integrator
/brov/model_based/start
/brov/model_based/stop
/brov/camera/set_camera_info
```

## 6. 구현 및 검증 완료

- Apple Silicon arm64 Docker/ROS 2 Humble 환경
- Mac bind mount와 XQuartz GUI 진입 자동화
- BlueOS UDP MAVLink 양방향 통신
- 25 Hz attitude/local-position/EKF telemetry와 stale/fault gate
- DVL ExternalNav 기반 LOCAL_POSITION_NED 사용
- start-heading mission frame
- 16차원 RL observation 포팅
- quaternion `q/-q` canonicalization
- waypoint LOS + 독립 depth outer-loop
- terminal 3D position hold
- RC1~8 RCPassThru 직접 PWM
- RC7/RC8 camera option 격리와 MAVLink mount tilt
- T1~T8 실제 채널·방향 검사
- 실측 reversal mask 적용
- Model-based PI/PD baseline controller
- T200 deadband-aware PWM
- controller publisher 충돌 방지 및 watchdog
- BlueOS camera → ROS Image/CameraInfo
- checkerboard calibration 및 ArUco node 구현
- `brov_base/control/perception/bringup` ament Python 패키지
- 안전 기본값의 base/model/RL/camera/full bringup launch
- Docker shell의 colcon overlay 자동 로딩과 `make ros-build/ros-test`

현재 회귀 테스트:

```text
8 passed
```

## 7. 실기체에서 확인된 사실

### Thruster

```text
Neutral: 8채널 1500 us
Electrical mapping: T1~T8 DIRECT, 교차 출력 없음
Motion direction: T2/T3/T8 reversed, 나머지 normal
```

### Telemetry

- 정상 구간의 observation은 약 25 Hz다.
- 간헐적으로 0.2초 이상, 과거에는 약 2초의 MAVLink gap이 관측됐다.
- telemetry fault 시 적분 중지, neutral, disarm한다.

### RL policy

기존 policy에서 다음 현상이 확인됐다.

```text
zero observation sway action ≈ +0.232
pitch error ±5.14° → pitch action 약 ±0.48
```

또한 observation quaternion이 `-identity` hemisphere로 들어가던 결함을 수정했다.
따라서 기존 실기체 policy 결과는 수정 전 observation의 영향이 포함되어 있으며,
수정 후 재시험이 필요하다.

### Model-based controller

- observation→wrench→allocation→PWM 경로가 동작한다.
- 작은 요구 추력이 T200 deadband 내부 PWM이 되던 문제를 수정했다.
- depth LOS 결합 때문에 수평 구간에서 하강하던 문제를 독립 depth hold로 수정했다.

## 8. 현재 미완료 및 위험

### 높은 우선순위

1. 수정된 quaternion/depth guidance 조건에서 RL policy 실기체 재검증
2. RL policy zero-state bias와 pitch gain의 학습 원인 분석 또는 재학습
3. 실제 ballast/tether 조건에서 model controller gain과 heave feedforward/PI 식별
4. 비정상 종료 후 SERVO function 및 RC7/RC8 option 복원 보장
5. PWM freshness watchdog 추가 — stale controller command 차단
6. start service에서 arm/mode/controller publisher 상태를 더 강하게 검증

### 카메라/인지

1. 실제 하우징·수중·640×480 intrinsic calibration
2. reprojection error와 독립 검증 이미지 평가
3. base_link↔camera optical extrinsic 측정
4. camera tilt 각도를 반영하는 동적 TF
5. ArUco 거리·자세 정확도 검증

현재 `config/camera_intrinsics.yaml`은 0 값의 placeholder이므로 metric ArUco pose는
아직 데모 준비 완료 상태가 아니다.

### 성능/통신

- Docker Desktop은 hard real-time이 아니며 현재 pipeline은 25 Hz다.
- 엄격한 400 Hz 제어는 현재 구조의 목표가 아니다.
- H264 RTP jitterbuffer 지연과 MAVLink UDP gap을 장시간 기록할 logging 도구가 필요하다.
- `obs_node`와 `policy_node`의 CPU 사용률이 높게 관측된 적이 있어 profiling이 필요하다.

## 9. 권장 후속 개발 순서

1. Intrinsic calibration 완료 및 결과 검증
2. Model-based 전체 waypoint 데모를 rosbag으로 기록
3. 실기체 heave/surge/sway 축별 system identification
4. Model controller gain/feedforward 확정
5. 수정된 observation으로 RL shadow-mode 비교
6. RL action과 model wrench를 동일 로그에서 비교
7. RL 재학습 또는 residual/hybrid controller 검토
8. ArUco extrinsic/dynamic TF와 robot pose 통합
9. rosbag 자동 기록 옵션 및 데모 로그 manifest 추가
10. 기존 `deploy.*` 구현을 패키지 내부로 완전히 이관해 독립 설치 가능하게 만들기

## 10. 개발 원칙

- `policy_node`와 model controller를 동시에 실행하지 않는다.
- raw telemetry, transformed observation, wrench, PWM, actual servo를 분리해 기록한다.
- 좌표계 부호를 추정으로 수정하지 않고 축별 bench/water test로 검증한다.
- `send_pwm=false`, `arm=false` shadow mode를 먼저 거친다.
- 실기체 parameter를 쓰는 작업은 백업/read-back/rollback을 유지한다.
- 정상 종료 시 반드시 controller → obs 순서로 종료해 설정 복원을 완료한다.

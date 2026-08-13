# BlueROV2 Sim2Real deploy

Apple Silicon MacBook의 Docker Desktop에서 Ubuntu 22.04/ROS 2 Humble을 실행하고,
BlueOS·ArduSub·DVL·USB 저조도 카메라를 이용해 BlueROV2 Heavy의 waypoint 제어와
영상 기반 위치 추정을 실험하는 배포 코드다.

현재 제공하는 두 제어 경로는 다음과 같다.

```text
                         ┌─ policy_node (TorchScript RL) ───────┐
BlueOS/DVL → obs_node ───┤                                      ├→ thruster PWM → ArduSub
                         └─ model_based_controller (PI/PD) ─────┘

BlueOS H264 → camera_stream_node → Image/CameraInfo → calibration → ArUco pose
```

두 controller는 동시에 실행하지 않는다. 현재 데모 기준 경로는 검증과 해석이 쉬운
model-based controller이며, RL policy는 observation quaternion 수정 후 실기체
재검증이 필요한 실험 경로다.

## 문서

- [Sim2Real 데모 실행 절차](DEMO_RUNBOOK.md)
- [개발 현황·아키텍처·후속 과제](DEVELOPMENT_STATUS.md)
- [Model-based controller 상세](MODEL_BASED_CONTROL.md)
- [카메라·intrinsic calibration·ArUco](docker/CAMERA.md)
- [Docker 환경](docker/README.md)
- [2026-08-13 작업 기록](WORKLOG_2026-08-13.md)

## 빠른 시작

Mac에서:

```bash
cd /Users/pyoseunghyeon/Documents/5.Research/underwater_NBV/step_2_BROV/deploy
make ros-build
make shell
```

`make ros-build`는 네 개의 ROS 2 패키지를 colcon overlay에 설치한다. `make shell`은
ROS Humble과 해당 overlay를 자동으로 source한다. 컨테이너에서 환경을 검사한다.
이 명령은 모터를 작동하지 않는다.

```bash
cd /workspace/deploy
python3 docker/check_environment.py
```

기본 bringup은 항상 `send_pwm=false`, `arm=false`이며 control service도 자동으로
호출하지 않는다.

```bash
ros2 launch brov_bringup sim2real_demo.launch.py \
  controller:=model camera:=true
```

실기체 arm/PWM을 사용하기 전에는 반드시 [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md)의
preflight와 종료 절차를 따른다.

# BlueROV2 RL deploy — macOS Docker 환경

Apple Silicon Mac의 Docker Desktop에서 Ubuntu 22.04, ROS 2 Humble 기반으로
`obs_node.py`와 `policy_node.py`를 실행하는 개발·실험 환경이다. IsaacLab과
RSL-RL은 정책 export에만 필요하므로 이 이미지에는 설치하지 않는다. ROS는
`ros-humble-desktop` 메타패키지를 설치하므로 rqt, RViz2 및 표준 GUI 도구를 포함한다.

## 구조

Mac의 현재 `deploy` 디렉터리는 컨테이너의 `/workspace/deploy`에 bind mount된다.
따라서 VS Code나 컨테이너에서 수정한 내용은 별도 동기화 없이 같은 Mac 파일에
즉시 반영된다. 컨테이너를 삭제해도 소스와 `exported/policy.pt`는 유지된다.

```text
BlueOS -- UDP 14550 --> Mac/Docker Desktop host network --> brov-dev:14550
                                                   |
                            obs_node -- ROS 2 --> policy_node
                                ^                    |
                                +--- thruster_pwm ---+

BlueOS camera -- RTP/H264 UDP 5600 --> Docker host network --> camera_stream_node
```

## 사전 조건

1. Apple Silicon용 Docker Desktop을 설치하고 실행한다.
   Docker Desktop의 Settings → Resources → Network에서 **Enable host
   networking**을 활성화하고 **Use kernel networking for UDP**는 비활성화한 뒤
   Apply & Restart한다. 이 조합이 현재 BlueOS MAVLink/camera 실기체에서 검증됐다.
2. Mac 터미널에서 아래 명령이 성공하는지 확인한다.

   ```bash
   docker --version
   docker compose version
   ```

3. Docker Desktop의 File Sharing에 프로젝트가 있는 `/Users` 경로가 허용되어
   있어야 한다. 기본 설정에서는 `/Users`가 공유된다.
4. BlueOS MAVLink endpoint의 목적지를 `Mac의 BlueROV2 네트워크 IP:14550/UDP`로
   설정한다. Docker 내부 IP가 아니라 Mac의 실제 Ethernet IP를 사용한다.

## 빌드 및 시작

Mac 터미널에서 `deploy` 디렉터리로 이동한다.

```bash
cd /Users/pyoseunghyeon/Documents/5.Research/underwater_NBV/step_2_BROV/deploy
docker compose build
make ros-build
docker compose ps
```

`make ros-build`는 컨테이너를 시작하고 `/workspace/deploy/ros2_ws`에서 네 패키지를
`colcon build --symlink-install`로 빌드한다. 최초 checkout 뒤에는 launch보다 먼저
반드시 한 번 실행한다.

환경, 패키지, 차량 YAML, TorchScript 정책 및 UDP 포트를 점검한다. 이 검사는
MAVLink 메시지를 보내거나 모터를 작동하지 않는다.

```bash
make check
```

## 패키지와 launch 실행

처음에는 반드시 실제 PWM과 arm을 끈 상태로 확인한다.

`make shell`로 진입하면 ROS Humble과 build된 overlay가 자동으로 로드된다. 확인:

```bash
make shell
ros2 pkg executables | grep '^brov_'
```

실제 PWM과 arm을 끈 shadow bringup:

```bash
ros2 launch brov_bringup sim2real_demo.launch.py \
  controller:=model camera:=true send_pwm:=false arm:=false
```

다른 터미널도 `make shell`로 열어 토픽을 검사한다.

```bash
ros2 topic list
ros2 topic hz /brov/observation
ros2 topic echo /brov/action
ros2 topic echo /brov/thruster_pwm
```

`obs_node`가 heartbeat와 정상 telemetry를 받고 observation/action/PWM 값의 크기와
부호를 확인한 뒤에만 `send_pwm:=true`를 사용한다. `arm:=true`는 수조, tether,
물리적 비상정지 수단과 채널 매핑까지 검증한 마지막 단계에서만 사용한다.

현재 `controller`는 `model` 또는 `rl` 중 하나를 명시하며 두 controller를 동시에
생성하지 않는다. launch는 `/brov/start_control`이나 controller start service를
자동 호출하지 않는다.

카메라 토픽, intrinsic calibration 및 ArUco pose 절차는
[`docker/CAMERA.md`](CAMERA.md)를 따른다.

비상정지:

```bash
ros2 topic pub --once /brov/estop std_msgs/msg/Empty '{}'
```

## XQuartz와 컨테이너 셸 자동 진입

Mac에 XQuartz를 한 번 설치한 후에는 프로젝트 디렉터리에서 다음 명령만 실행한다.

```bash
make shell
```

이 명령은 XQuartz의 TCP X11 설정, XQuartz 실행, `xhost +localhost`, Compose
컨테이너 시작, ROS overlay source 및 `brov-dev` 셸 진입을 자동으로 수행한다. 컨테이너 셸에서 나오면
허용했던 localhost X11 접근도 자동으로 회수한다.

어느 디렉터리에서든 `brov` 명령으로 실행하려면 Mac의 `~/.zshrc`에 다음 한 줄을
직접 추가한다.

```bash
alias brov='make -C "/Users/pyoseunghyeon/Documents/5.Research/underwater_NBV/step_2_BROV/deploy" shell'
```

설정을 반영한 뒤 `brov`를 실행한다.

```bash
source ~/.zshrc
brov
```

## UDP 진단

Mac에서 BlueROV2 연결 인터페이스 주소를 확인한다.

```bash
ifconfig
```

컨테이너에서 UDP 14550 수신 여부를 관찰할 수 있다. 이 명령을 실행하는 동안에는
`obs_node`를 동시에 실행하지 않는다. 두 프로세스가 같은 포트를 점유할 수 있기
때문이다.

```bash
docker compose exec brov nc -u -l 14550
```

heartbeat가 도착하지 않으면 다음 순서로 확인한다.

1. BlueOS endpoint 목적지가 Mac IP인지 확인
2. endpoint 포트가 UDP 14550인지 확인
3. macOS 방화벽에서 Docker Desktop 수신 허용
4. 다른 프로그램이 Mac의 UDP 14550을 사용 중인지 확인
5. `docker compose logs brov` 및 `docker compose ps` 확인

## VS Code Dev Container (선택)

VS Code에 Microsoft **Dev Containers** 확장을 설치한 경우 Command Palette에서
`Dev Containers: Reopen in Container`를 선택한다. `.devcontainer` 설정은 같은
Compose 서비스를 재사용하므로, 터미널 방식과 별도의 환경이 생성되지 않는다.

VS Code 창은 Mac에서 실행되지만 Python interpreter와 터미널은 Ubuntu 컨테이너를
사용한다. 편집 대상은 bind-mounted Mac 파일이다.

## 종료와 재빌드

```bash
docker compose down
```

소스 수정만으로는 이미지 재빌드가 필요 없다. `Dockerfile`이나
`docker/requirements.txt`를 수정했을 때만 다시 빌드한다.

```bash
docker compose build --no-cache
docker compose up -d
```

## 주의사항

- Docker Desktop은 hard real-time 환경이 아니다. 현재 25 Hz 정책 실험에는 사용할
  수 있지만 엄격한 400 Hz deadline을 보장하지 않는다.
- 컨테이너나 Docker VM이 비정상 종료되면 Python의 정상 종료 루틴이 실행되지 않을
  수 있다. ArduSub의 RC override/failsafe timeout을 별도로 검증해야 한다.
- Docker Desktop for Mac은 일반적인 USB 장치 직접 passthrough를 지원하지 않는다.
  이 구성은 BlueOS와 Ethernet/UDP로 연결하는 것을 전제로 한다.
- `export_policy.py`는 이 컨테이너에서 실행하지 않는다. IsaacLab 환경에서 export한
  `policy.pt`만 `exported/`에 배치한다.
- 현재 ROS package console entry point는 기존의 검증된 `/workspace/deploy` 구현을
  호출한다. Compose bind mount와 `PYTHONPATH=/workspace`가 배포 계약의 일부다.
- PyTorch는 Apple Silicon 컨테이너에 맞춰 Dockerfile이 CPU wheel index에서 설치한다.
  따라서 이 환경에서는 `rosdep install`보다 Dockerfile과 `requirements.txt`를
  Python runtime dependency의 기준으로 사용한다.

# BlueROV2 camera → ROS 2 → calibration → ArUco

현재 BlueOS 설정을 전제로 한다.

```text
Input:    /dev/video2
Format:   640 x 480 @ 30 fps
Encoding: H264
Endpoint: udp://192.168.2.101:5600
```

이 Compose 구성은 Docker Desktop의 `network_mode: host`를 사용한다. Docker
Desktop Settings → Resources → Network에서 **Enable host networking**을 먼저
활성화해야 하며, `ports:` publish를 함께 사용하지 않는다.

## 1. 이미지 토픽 발행

Docker 이미지를 갱신한 뒤 컨테이너에서 실행한다.

```bash
cd /workspace/deploy/ros2_nodes
python3 camera_stream_node.py --ros-args \
  -p udp_port:=5600 \
  -p frame_id:=camera_optical_frame \
  -p camera_info_path:=/workspace/deploy/config/camera_intrinsics.yaml
```

다른 터미널에서 확인한다.

```bash
ros2 topic hz /brov/camera/image_raw
ros2 topic echo --once /brov/camera/camera_info
```

### macOS에서 rqt_image_view 사용

컨테이너의 Linux GUI를 macOS에 표시하려면 Mac에 XQuartz가 필요하다. XQuartz를
한 번 설치한 뒤 프로젝트의 자동 진입 명령을 사용하는 것을 권장한다.

```bash
make shell
```

이 명령은 XQuartz 시작과 X11 접근 허용을 자동으로 처리하고 컨테이너 셸을 연다.
그 다음 컨테이너에서 실행한다.

```bash
docker compose exec brov rqt_image_view /brov/camera/image_raw
# 또는 전체 rqt UI
docker compose exec brov rqt
```

컨테이너 셸을 종료하면 로컬 X11 접근 허용도 자동으로 회수된다.

`cannot connect to display`가 나오면 XQuartz 재시작 여부와 Compose의
`DISPLAY=host.docker.internal:0` 설정을 확인한다. GUI가 필요하지 않으면
`ros2 topic hz` 및 `/brov/aruco/debug_image` 토픽만으로도 파이프라인을 운용할 수
있다.

초기 `camera_intrinsics.yaml`의 초점거리는 0이므로 보정 전에는 ArUco 거리·pose를
계산하지 않는다.

## 2. Intrinsic calibration (headless 권장)

아래 예시는 checkerboard 내부 코너가 8x6이고 정사각형 한 변이 30 mm인 경우다.
실제 보드 치수와 다르면 반드시 바꾼다.

```bash
cd /workspace/deploy/ros2_nodes
python3 checkerboard_calibration_node.py --ros-args \
  -p columns:=8 \
  -p rows:=6 \
  -p square_size_m:=0.030 \
  -p target_samples:=30
```

같은 해상도, 같은 focus, 실제 방수 하우징 및 실제 운용 매질(수중)에서 보정한다.
보드가 화면의 중앙·네 모서리·다양한 거리·기울기를 충분히 덮도록 움직인다.
서로 다른 표본 30장이 자동 수집되면 보정 결과와 RMS reprojection error가 로그에
출력되고 bind-mounted `config/camera_intrinsics.yaml`에 저장된다. 저장 후
`camera_stream_node.py`를 재시작한다.

표준 ROS `cameracalibrator` GUI도 이미지에 설치되어 있지만 macOS Docker에서
사용하려면 XQuartz 등 별도 X11 구성이 필요하므로 기본 절차에서는 사용하지 않는다.

## 3. ArUco pose

기본값은 `DICT_4X4_50`, ID 0, 검은 외곽선 한 변 0.15 m다. 실제 marker 길이를
정확히 측정해 바꾼다.

```bash
python3 aruco_pose_node.py --ros-args \
  -p dictionary:=DICT_4X4_50 \
  -p marker_id:=0 \
  -p marker_length_m:=0.15
```

출력:

```text
/brov/aruco/visible
/brov/aruco/marker_pose
/brov/aruco/debug_image
TF: camera_optical_frame -> aruco_reference
```

## 4. Marker 기준 로봇 pose

카메라 외부 파라미터를 측정하기 전에는 `publish_robot_pose=false`를 유지한다.
측정 후 ROS `base_link` 기준 카메라 optical frame의 위치(m)와 ZYX에 대응하는
고정 roll/pitch/yaw(rad)를 입력한다.

```bash
python3 aruco_pose_node.py --ros-args \
  -p marker_length_m:=0.15 \
  -p publish_robot_pose:=true \
  -p base_to_camera_xyz:="[0.12, 0.0, 0.03]" \
  -p base_to_camera_rpy:="[0.0, 0.0, 0.0]"
```

추가 출력:

```text
/brov/aruco/robot_pose      # aruco_reference 기준 base_link pose
TF: aruco_reference -> base_link
```

위 xyz/rpy는 예시일 뿐이며 실측 없이 사용하면 안 된다. Tilt gimbal을 사용한다면
고정 외부 파라미터가 아니라 현재 tilt 각을 반영하는 동적 TF가 필요하다.

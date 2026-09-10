# deploy/ 실행 의존성 (실험 laptop 로컬 실행용, 2026-09-11)

이 디렉터리의 노드는 컨테이너 없이 laptop 에서 그대로 돈다는 전제로 정리한다. 아래 목록은 각
파일의 import 를 그대로 옮긴 것이며, 버전은 이 PC 의 SITL 컨테이너(Ubuntu 22.04)에서 검증한 값이다.

## 시스템 (apt)
| 패키지 | 용도 | 비고 |
|---|---|---|
| `ros-humble-desktop` (rclpy, rviz2, tf2_ros, visualization_msgs, nav_msgs, geometry_msgs, sensor_msgs, std_srvs) | 모든 노드, RViz | Humble 고정. RViz/tf2_ros 는 시스템 python(/usr/bin/python3 3.10) 이어야 한다 — conda python 으로 실행하면 rclpy 를 못 찾는다 |
| `ros-humble-cv-bridge` | `aruco_pose_node` (brov_perception) | cv_bridge 는 **시스템 OpenCV(4.x)** 와 짝이어야 한다. pip `opencv-python` 5.x 를 같이 깔면 `cv2_to_imgmsg` 가 `KeyError: 16` 으로 죽는다(이 PC 컨테이너에서 실측) — 그래서 `nbv_camera_pipeline.py` 는 cv_bridge 없이 직접 변환한다 |
| `ros-humble-ros-gz-bridge`, Gazebo **Garden** | SITL 전용 (`run_nbv_sitl.sh`) | 실기에서는 불필요 |
| ArduSub SITL + MAVProxy | SITL 전용 | 실기에서는 실제 autopilot |

## brov_ros2 (colcon 워크스페이스, `--symlink-install` 권장)
`brov_interfaces`, `brov_base`, `brov_mission`, `brov_control`, `brov_localization`, `brov_perception`, `brov_viz`.
NBV 노드가 직접 import 하는 것: `brov_interfaces.msg` (ResolvedMission, OdometrySession, AlignedOdometry, LocalizationStatus).
`nbv_viz_node` 는 brov_interfaces 없이 표준 메시지만 쓴다 — RViz 만 띄우는 PC 에는 워크스페이스 빌드 없이
`/usr/bin/python3 brov_viz/brov_viz/nbv_viz_node.py` 로 실행 가능(`run_rviz_host.sh`).

## Python (pip, 시스템 python 3.10 에)
| 패키지 | 검증 버전 | 쓰는 파일 |
|---|---|---|
| `numpy` | 1.26.4 | 전부 |
| `torch` (CPU 가능) | 컨테이너 기본 | `envs/tsdf_fusion.py`(TSDF 융합), `envs/nbv_baselines.py`, `nbv_belief_node.py`, `nbv_policy_node.py`, `nbv_truth_localization.py` 시험 |
| `scikit-image` | 0.25.2 | `nbv_belief_node.py` 의 mesh 복원(`skimage.measure.marching_cubes`). 없으면 voxel 큐브만 발행하고 경고 |
| `opencv-python` (cv2) | 컨테이너 기본 | `nbv_camera_pipeline.py`(수중 렌더·리사이즈), `aruco_pose_node`(brov_perception) — cv_bridge 짝 주의(위) |
| `pymavlink` | 컨테이너 기본 | `mavlink_ekf_probe.py`, `stage2_sitl_dvl_injector.py` (SITL 전용) |
| `pytest` | | `test_*.py` |

## 공유 모듈 경로
`nbv_policy_node.py`, `nbv_belief_node.py` 는 `envs/tsdf_fusion.py`·`envs/nbv_baselines.py` 를 저장소 상위
(`step_3_NBV_BROV/`) 또는 `/tmp/nbv_src` 에서 찾는다 — laptop 에서는 저장소를 통째로 두면 된다.
Isaac Lab 은 **불필요**(`envs/tsdf_fusion.py` 는 순수 torch).

## 실행 (laptop, 실기)
```bash
# 1) brov_ros2 스택 (실기): pool_localized_demo.launch.py 계열 + mission_manager_nbv_pose.yaml
# 2) belief + 정책 노드:  python3 nbv_belief_node.py ...;  python3 nbv_policy_node.py --manager-params ...
# 3) RViz:               ./run_rviz_host.sh        (BROV_ROS2=<brov_ros2 경로>)
```

## RViz / DDS 주의 (호스트·laptop)
- `run_rviz_host.sh` 는 `ROS_LOCALHOST_ONLY=0` + `FASTRTPS_DEFAULT_PROFILES_FILE=deploy/fastdds_udp_only.xml`(UDPv4 전용) 로
  띄운다. 컨테이너/다른 프로세스 공간의 노드와는 Fast-DDS shared-memory 가 통하지 않아(`/dev/shm` 미공유) discovery 만 되고
  데이터가 안 온다 — 실기 laptop 한 대에서 모두 돌리면 필요 없지만 켜 둬도 무해하다.
- rviz2 는 셸의 `LD_LIBRARY_PATH`(cuda, gazebo-11)·conda 경로를 물려받으면 "Failed to create an OpenGL context" 로 죽는다 →
  스크립트가 깨끗한 환경에서 ROS 만 source 한다. `.bashrc` 의 `ROS_LOCALHOST_ONLY=1` 도 스크립트 안에서 0 으로 덮는다.
- OpenCV: `aruco_pose_node` 는 4.x 함수형 API 와 4.7+/5.x `ArucoDetector`·`solvePnP(IPPE_SQUARE)` 를 모두 지원한다.
  다만 `brov_perception/test` 의 합성 마커 시험 2건은 `cv2.aruco.drawMarker`/`estimatePoseSingleMarkers` 를 써서 5.x 에선 실패한다(시험 코드 한정).

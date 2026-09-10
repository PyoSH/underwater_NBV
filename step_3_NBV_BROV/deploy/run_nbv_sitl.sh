#!/usr/bin/env bash
# NBV 배포 SITL 사이클 (컨테이너 안에서 실행). 2026-09-10.
#
# step_2_BROV/run_attitude_hold_model_based.sh 의 기동 순서를 계승한다:
#   gz sim -> ArduSub -> MAVProxy -> EKF 파라미터 검증 -> ros_gz_bridge -> DVL injector
#   -> 카메라 파이프라인 -> brov bringup -> rosbag
# 바뀐 것은 (1) 수조 world, (2) 카메라 토픽 브리지, (3) DVL 자세 dropout, (4) seabed z.
#
# Phase 3 의 목적은 표류 실측이다. 그래서 이 스크립트는 제어를 시작하지 않는 모드
# (--no-control) 를 기본으로 두고, 미션은 별도 인자로 켠다 — 무엇이 표류를 만드는지
# 섞이면 안 되기 때문이다.

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: run_nbv_sitl.sh <run_dir> [옵션]
  --tilt-deg N        DVL bottom-lock 최대 기울기 [deg] (기본 180 = 비활성)
  --duration-s N      기록 시간 (기본 120)
  --spawn X,Y,Z       ROV 스폰 pose (기본 2.2,0,-0.6)
  --water NAME        Jerlov 프리셋 (기본 IB, none 이면 감쇠 없음)
  --dvl-duration-s N  DVL injector 를 N초 뒤 정지 (0=계속). 그 뒤 구간이 INS 단독이다
  --dvl-restart-after-s N  첫 injector 정지 후 N초 뒤 두 번째 injector 기동 (0=안 함).
                      EKF 가 fix 복귀에 회복하는지 재는 용도
  --belief 0|1        NBV 믿음 노드(TSDF) 기동 (기본 1)
USAGE
  exit 2
}

[[ $# -ge 1 ]] || usage
RUN_DIR=$1; shift
TILT_DEG=180
DURATION_S=120
SPAWN="2.2,0,-0.6"
WATER=IB
DVL_DURATION_S=0
BELIEF=1
DVL_RESTART_AFTER_S=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tilt-deg) TILT_DEG=$2; shift 2 ;;
    --duration-s) DURATION_S=$2; shift 2 ;;
    --spawn) SPAWN=$2; shift 2 ;;
    --water) WATER=$2; shift 2 ;;
    --dvl-duration-s) DVL_DURATION_S=$2; shift 2 ;;
    --dvl-restart-after-s) DVL_RESTART_AFTER_S=$2; shift 2 ;;
    --belief) BELIEF=$2; shift 2 ;;
    *) usage ;;
  esac
done
case "$RUN_DIR" in
  /home/bluerov2_sitl/brov_ros2/runtime/experiments/nbv_*) ;;
  *) echo "RUN_DIR 은 .../runtime/experiments/nbv_* 이어야 한다" >&2; exit 2 ;;
esac
[[ -e "$RUN_DIR" ]] && { echo "기존 run 디렉터리 재사용 거부: $RUN_DIR" >&2; exit 2; }

DEPLOY=/tmp/nbv_deploy
WORLD=$DEPLOY/pool_6x10x3.sdf
PARAMS=/tmp/stage2_waterlinked_default.parm
DVL_INJECTOR=/tmp/stage2_sitl_dvl_injector.py
ORIGIN_HELPER=/tmp/stage2_set_ekf_origin.py
BROV_SOURCE=/home/bluerov2_sitl/brov_ros2
BROV_INSTALL=$BROV_SOURCE/install_mk2
SEABED_Z=-2.7
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}

GZ_PID=; ARDUSUB_PID=; MAVPROXY_PID=; BRIDGE_PID=; DVL_PID=; DVL2_PID=; CAM_PID=; BAG_PID=; EKF_PID=; BELIEF_PID=
stop_group() { local p=${1:-} s=${2:-INT}; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && kill -"$s" -- -"$p" 2>/dev/null || true; }
cleanup() {
  echo "[nbv-sitl] 정리"
  for sig in INT TERM KILL; do
    for p in "$BAG_PID" "$BELIEF_PID" "$EKF_PID" "$CAM_PID" "$DVL2_PID" "$DVL_PID" "$BRIDGE_PID" "$MAVPROXY_PID" "$ARDUSUB_PID" "$GZ_PID"; do
      stop_group "$p" "$sig"
    done
    sleep 1
  done
}
trap 'st=$?; trap - EXIT INT TERM; cleanup; exit $st' EXIT INT TERM

for f in "$WORLD" "$PARAMS" "$DVL_INJECTOR" "$ORIGIN_HELPER" \
         "$DEPLOY/nbv_camera_pipeline.py" "$DEPLOY/mavlink_ekf_probe.py"; do test -f "$f"; done
mkdir -p "$RUN_DIR"

set +u
source /opt/ros/humble/setup.bash
source /home/bluerov2_sitl/colcon_ws/install/setup.bash
[[ -f "$BROV_INSTALL/setup.bash" ]] && source "$BROV_INSTALL/setup.bash"
source /home/bluerov2_sitl/gz_ws/gazebo_exports.sh
set -u
export GZ_SIM_RESOURCE_PATH=$DEPLOY/models:$GZ_SIM_RESOURCE_PATH

# ROV 카메라를 얹은 파생 모델을 상류에서 매번 새로 만든다 (포크 금지)
python3 "$DEPLOY/make_rov_with_camera.py" --out "$DEPLOY/models/bluerov2_heavy" \
  > "$RUN_DIR/rov_camera.txt" 2>&1

{
  echo "world=$WORLD"
  echo "seabed_world_z=$SEABED_Z"
  echo "dvl_bottom_lock_max_tilt_deg=$TILT_DEG"
  echo "water_preset=$WATER"
  echo "belief=$BELIEF"
  echo "spawn=$SPAWN"
  echo "duration_s=$DURATION_S"
  echo "dvl_duration_s=$DVL_DURATION_S"
  echo "dvl_restart_after_s=$DVL_RESTART_AFTER_S"
  sha256sum "$WORLD" "$PARAMS" "$DVL_INJECTOR" "$DEPLOY/nbv_camera_pipeline.py" \
            "$DEPLOY/uw_render.py" "$DEPLOY/viewpoint_geofence.py"
} > "$RUN_DIR/manifest.txt"

setsid gz sim -s -r -v 2 "$WORLD" > "$RUN_DIR/gazebo.log" 2>&1 &
GZ_PID=$!
sleep 8

IFS=, read -r SX SY SZ <<< "$SPAWN"
gz service -s /world/pool_6x10x3/set_pose --reqtype gz.msgs.Pose \
  --reptype gz.msgs.Boolean --timeout 3000 \
  --req "name: \"bluerov2_heavy\", position: {x: $SX, y: $SY, z: $SZ}" \
  > "$RUN_DIR/spawn_set_pose.txt" 2>&1 || true

setsid bash -lc "cd '$RUN_DIR' && exec \
  /home/bluerov2_sitl/ardupilot/build/sitl/bin/ardusub \
  -S -w --model JSON --speedup 1 --slave 0 \
  --defaults '/home/bluerov2_sitl/ardupilot/Tools/autotest/default_params/sub-6dof.parm,$PARAMS' \
  --sim-address=127.0.0.1 -I0 \
  --home 55.99541530863445,-3.301022500491058,0.0,0.0" \
  > "$RUN_DIR/ardusub.stdout.log" 2>&1 &
ARDUSUB_PID=$!

setsid bash -lc "cd '$RUN_DIR' && exec /home/bluerov2_sitl/.local/bin/mavproxy.py \
  --daemon --master=tcp:127.0.0.1:5760 --sitl=127.0.0.1:5501 --streamrate=25 \
  --out=udp:127.0.0.1:14552 --out=udp:127.0.0.1:14554 --out=udp:127.0.0.1:14555" \
  > "$RUN_DIR/mavproxy.log" 2>&1 &
MAVPROXY_PID=$!

for _ in $(seq 1 80); do grep -q "online system 1" "$RUN_DIR/mavproxy.log" 2>/dev/null && break; sleep 0.5; done
grep -q "online system 1" "$RUN_DIR/mavproxy.log"

for _ in $(seq 1 80); do
  [[ -s "$RUN_DIR/mav.parm" ]] && grep -q '^VISO_TYPE' "$RUN_DIR/mav.parm" \
    && grep -q '^RNGFND1_TYPE' "$RUN_DIR/mav.parm" && break
  sleep 0.25
done
for spec in AHRS_EKF_TYPE=3 EK2_ENABLE=0 EK3_ENABLE=1 VISO_TYPE=1 \
            EK3_SRC1_POSXY=6 EK3_SRC1_VELXY=6 EK3_SRC1_POSZ=1 \
            EK3_SRC1_VELZ=0 EK3_SRC1_YAW=1 RNGFND1_TYPE=10 \
            SIM_GPS_DISABLE=1 GPS1_TYPE=0 GPS2_TYPE=0; do
  n=${spec%%=*}; v=${spec#*=}
  awk -v n="$n" -v v="$v" '$1==n{f=1; if (($2+0)!=(v+0)) exit 2} END{if(!f) exit 1}' \
    "$RUN_DIR/mav.parm" || { echo "postboot 파라미터 불일치: $spec" >&2; exit 1; }
done
grep -E '^(AHRS_EKF_TYPE|EK2_|EK3_|GPS[12]_TYPE|RNGFND1_TYPE|SIM_GPS_DISABLE|VISO_)' \
  "$RUN_DIR/mav.parm" > "$RUN_DIR/ekf_params_verified.txt"

setsid ros2 run ros_gz_bridge parameter_bridge \
  '/model/bluerov2_heavy/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
  '/brov/camera/image@sensor_msgs/msg/Image[gz.msgs.Image' \
  '/brov/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image' \
  --ros-args \
  -r /model/bluerov2_heavy/odometry:=/brov/sim/gazebo_odometry_raw \
  -r /brov/camera/image:=/brov/sim/camera_rgb \
  -r /brov/camera/depth_image:=/brov/sim/camera_depth \
  > "$RUN_DIR/gt_bridge.log" 2>&1 &
BRIDGE_PID=$!
for _ in $(seq 1 60); do
  timeout 1 ros2 topic echo --once /brov/sim/gazebo_odometry_raw >/dev/null 2>&1 && break
done
timeout 2 ros2 topic echo --once /brov/sim/gazebo_odometry_raw > "$RUN_DIR/gt_pre_bringup.txt"

setsid env MAVLINK20=1 /usr/bin/python3 "$DVL_INJECTOR" \
  --connection udpin:0.0.0.0:14555 --topic /brov/sim/gazebo_odometry_raw \
  --rate-hz 10 --far-rate-hz 5 --range-transition-m 3 \
  --rangefinder --seabed-world-z "$SEABED_Z" \
  --bottom-lock-min-world-z -2.6 --bottom-lock-max-world-z -0.05 \
  --bottom-lock-max-tilt-deg "$TILT_DEG" \
  --duration-s "$DVL_DURATION_S" --confirm-sitl > "$RUN_DIR/dvl_injector.log" 2>&1 &
DVL_PID=$!

# EKF origin: GPS 없이 EKF3 가 로컬 해를 내려면 원점이 필요하다
/usr/bin/python3 "$ORIGIN_HELPER" --connection udpin:0.0.0.0:14554 \
  > "$RUN_DIR/ekf_origin.log" 2>&1

setsid python3 "$DEPLOY/mavlink_ekf_probe.py" --connection udpin:0.0.0.0:14552 \
  --confirm-sitl > "$RUN_DIR/ekf_probe.log" 2>&1 &
EKF_PID=$!

setsid python3 "$DEPLOY/nbv_camera_pipeline.py" --ros-args \
  -p water:="$WATER" > "$RUN_DIR/camera_pipeline.log" 2>&1 &
CAM_PID=$!

if [[ "$BELIEF" == "1" ]]; then
  setsid python3 "$DEPLOY/nbv_belief_node.py" \
    > "$RUN_DIR/belief.log" 2>&1 &
  BELIEF_PID=$!
fi

setsid ros2 bag record -o "$RUN_DIR/bag" \
  /brov/sim/gazebo_odometry_raw /brov/sim/ekf_local_ned /brov/sim/ekf_status \
  /brov/stage2/dvl_sample /brov/stage2/dvl_valid /brov/stage2/dvl_status \
  /brov/camera/camera_info /brov/nbv/image_obs \
  /brov/nbv/vox_actor /brov/nbv/spherical /brov/nbv/belief_status \
  > "$RUN_DIR/bag.log" 2>&1 &
BAG_PID=$!

start_dvl() {   # $1 = duration-s, $2 = 로그 접미사
  setsid env MAVLINK20=1 /usr/bin/python3 "$DVL_INJECTOR" \
    --connection udpin:0.0.0.0:14555 --topic /brov/sim/gazebo_odometry_raw \
    --rate-hz 10 --far-rate-hz 5 --range-transition-m 3 \
    --rangefinder --seabed-world-z "$SEABED_Z" \
    --bottom-lock-min-world-z -2.6 --bottom-lock-max-world-z -0.05 \
    --bottom-lock-max-tilt-deg "$TILT_DEG" \
    --duration-s "$1" --confirm-sitl > "$RUN_DIR/dvl_injector_$2.log" 2>&1 &
  echo $!
}

echo "[nbv-sitl] 기록 ${DURATION_S}s (tilt gate ${TILT_DEG} deg)"
if [[ "$DVL_RESTART_AFTER_S" != "0" && "$DVL_DURATION_S" != "0" ]]; then
  # 1차 injector 가 스스로 멈추기를 기다렸다가 gap 후 2차를 띄운다.
  sleep "$DVL_DURATION_S"
  echo "[nbv-sitl] DVL 정지 — ${DVL_RESTART_AFTER_S}s 공백 후 재시작"
  sleep "$DVL_RESTART_AFTER_S"
  DVL2_PID=$(start_dvl 0 second)
  echo "[nbv-sitl] DVL 재시작 (PID $DVL2_PID)"
  sleep $(( DURATION_S - DVL_DURATION_S - DVL_RESTART_AFTER_S ))
else
  sleep "$DURATION_S"
fi
echo "[nbv-sitl] 완료: $RUN_DIR"

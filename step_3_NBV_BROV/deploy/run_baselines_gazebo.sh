#!/usr/bin/env bash
# 신 카메라(실기 화각, crop 없음)로 석고틀 베이스라인. 컨테이너 안에서 실행.
#   sweep 100결정 = 관측 가능 표면 ceiling (계획서 §11.7-4 정의)
#   orbit / random 40결정 x 3시드
# 출력: $OUT/loop_<policy>_s<seed>.csv
set -euo pipefail
OUT=${1:-/tmp/nbv_baselines}
mkdir -p "$OUT"
# ROS setup 스크립트는 미정의 변수를 만지므로 set -u 아래에서는 죽는다 (run_nbv_sitl.sh 와 동일 처리)
set +u
source /opt/ros/humble/setup.bash
source /home/bluerov2_sitl/colcon_ws/install/setup.bash
source /home/bluerov2_sitl/gz_ws/gazebo_exports.sh
set -u
export GZ_SIM_RESOURCE_PATH=/tmp/nbv_deploy/models:$GZ_SIM_RESOURCE_PATH
export ROS_DOMAIN_ID=97
cd /tmp/nbv_deploy

setsid gz sim -s -r -v 2 pool_sweep.sdf > "$OUT/gz.log" 2>&1 &
GZ=$!
sleep 14
setsid ros2 run ros_gz_bridge parameter_bridge \
  "/nbv/probe/depth_image@sensor_msgs/msg/Image[gz.msgs.Image" > "$OUT/bridge.log" 2>&1 &
BR=$!
sleep 6
# (나) 카메라 = 실기 화각 그대로 -> crop 끔
setsid python3 nbv_belief_node.py --ros-args \
  -p surface_mask_path:=/tmp/nbv_deploy/surf_vol.npy \
  -p crop_to_sim_fov:=false > "$OUT/belief.log" 2>&1 &
BL=$!
sleep 4

run() {  # policy decisions seed
  python3 nbv_loop_probe.py --policy "$1" --decisions "$2" --seed "$3" --out "$OUT/tmp_$1_$3" \
    > "$OUT/run_$1_s$3.log" 2>&1
  mv "$OUT/tmp_$1_$3/loop_$1.csv" "$OUT/loop_$1_s$3.csv"
  echo "[baselines] $1 seed=$3 done: $(tail -1 "$OUT/loop_$1_s$3.csv" | cut -d, -f1,10)"
}
run sweep 100 0
for s in 1 2 3; do run orbit 40 $s; done
for s in 1 2 3; do run random 40 $s; done

kill -INT -- -$BL -$BR -$GZ 2>/dev/null || true
sleep 2
kill -KILL -- -$BL -$BR -$GZ 2>/dev/null || true
echo "[baselines] 완료: $OUT"

#!/usr/bin/env bash
# Fresh Case-A cycle with the classical model-based controller (no RL policy)
# using Gazebo truth or Water Linked VPD/EKF feedback. Mirrors
# run_mk2_case_c_deploy.sh but swaps policy_node_mk2 for
# model_based_controller_node so the identical plant/allocation/actuator
# chain can be used as a causal-isolation baseline.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /home/bluerov2_sitl/brov_ros2/runtime/experiments/case_c_model_based_<run_id> gazebo_truth|mavlink_ekf" >&2
  exit 2
fi
RUN_DIR=$1
FEEDBACK_SOURCE=$2
case "$RUN_DIR" in
  /home/bluerov2_sitl/brov_ros2/runtime/experiments/case_c_model_based_*) ;;
  *) echo "RUN_DIR must be an explicit persistent case_c_model_based_* path" >&2; exit 2 ;;
esac
case "$FEEDBACK_SOURCE" in
  gazebo_truth|mavlink_ekf) ;;
  *) echo "feedback source must be gazebo_truth or mavlink_ekf" >&2; exit 2 ;;
esac
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to reuse existing run directory: $RUN_DIR" >&2
  exit 2
fi

WORLD=/tmp/stage2_bluerov2_heavy_underwater_8p5m.sdf
PARAMS=/tmp/stage2_waterlinked_default.parm
DVL_INJECTOR=/tmp/stage2_sitl_dvl_injector.py
ORIGIN_HELPER=/tmp/stage2_set_ekf_origin.py
CYCLE_SUPERVISOR=/tmp/stage2_wait_case_c_cycle.py
START_BARRIER=/tmp/stage2_wait_gt_start.py
BROV_SOURCE=/home/bluerov2_sitl/brov_ros2
BROV_INSTALL=$BROV_SOURCE/install_mk2
MISSION=$BROV_SOURCE/brov_bringup/config/mission_sim2sim_mk2_case_c_0p5.yaml
CONTROLLER_CONFIG=$BROV_SOURCE/brov_control/config/model_controller.yaml
ROS_DOMAIN_ID=42
export ROS_DOMAIN_ID

GZ_PID=
ARDUSUB_PID=
MAVPROXY_PID=
BRIDGE_PID=
DVL_PID=
LAUNCH_PID=
BAG_PID=
SUPERVISOR_PID=

stop_group() {
  local pid=${1:-}
  local signal=${2:-INT}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -"$signal" -- -"$pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo "[case-c-model-based-$FEEDBACK_SOURCE] stopping control and task-owned processes"
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    timeout 4 ros2 service call /brov/model_based/stop std_srvs/srv/Trigger "{}" \
      >"$RUN_DIR/cleanup_model_stop.txt" 2>&1 || true
    timeout 4 ros2 service call /brov/stop_control std_srvs/srv/Trigger "{}" \
      >"$RUN_DIR/cleanup_stop.txt" 2>&1 || true
    timeout 4 ros2 service call /brov/disarm_control std_srvs/srv/Trigger "{}" \
      >"$RUN_DIR/cleanup_disarm.txt" 2>&1 || true
  fi
  stop_group "$BAG_PID" INT
  stop_group "$SUPERVISOR_PID" INT
  stop_group "$LAUNCH_PID" INT
  stop_group "$DVL_PID" INT
  stop_group "$BRIDGE_PID" INT
  stop_group "$MAVPROXY_PID" INT
  stop_group "$ARDUSUB_PID" INT
  stop_group "$GZ_PID" INT
  sleep 3
  stop_group "$BAG_PID" TERM
  stop_group "$SUPERVISOR_PID" TERM
  stop_group "$LAUNCH_PID" TERM
  stop_group "$DVL_PID" TERM
  stop_group "$BRIDGE_PID" TERM
  stop_group "$MAVPROXY_PID" TERM
  stop_group "$ARDUSUB_PID" TERM
  stop_group "$GZ_PID" TERM
  sleep 1
  stop_group "$BAG_PID" KILL
  stop_group "$SUPERVISOR_PID" KILL
  stop_group "$LAUNCH_PID" KILL
  stop_group "$DVL_PID" KILL
  stop_group "$BRIDGE_PID" KILL
  stop_group "$MAVPROXY_PID" KILL
  stop_group "$ARDUSUB_PID" KILL
  stop_group "$GZ_PID" KILL
}

finish() {
  local status=$?
  trap - EXIT INT TERM
  cleanup
  exit "$status"
}
trap finish EXIT INT TERM

for required in "$WORLD" "$PARAMS" "$MISSION" "$DVL_INJECTOR" \
  "$ORIGIN_HELPER" "$CYCLE_SUPERVISOR" "$START_BARRIER" "$CONTROLLER_CONFIG"; do
  test -f "$required"
done
test -f "$BROV_INSTALL/setup.bash"

mkdir -p "$RUN_DIR"

set +u
source /opt/ros/humble/setup.bash
source /home/bluerov2_sitl/colcon_ws/install/setup.bash
source "$BROV_INSTALL/setup.bash"
source /home/bluerov2_sitl/gz_ws/gazebo_exports.sh
set -u

if ros2 node list 2>/dev/null | grep -Eq '^/brov_(obs|policy|model_based_controller)_node$|^/brov_model_based_controller$'; then
  echo "stale BROV control nodes are already running" >&2
  exit 1
fi

{
  echo "deployment=model_based_controller"
  echo "required_executable=model_based_controller_node"
  echo "feedback_source=$FEEDBACK_SOURCE"
  echo "mission_contract=direct_relative_case_c_shaped_2m_0p5"
  echo "dvl_contract=water_linked_code_default_aligned_ideal_measurement"
  echo "dvl_orientation_assumption=DVL_DOWN"
  echo "dvl_should_send=POSITION_DELTA"
  echo "dvl_rangefinder=true"
  echo "dvl_auto_rate_hz=10_below_3m__5_above_3m"
  echo "dvl_queue_delay_s=0"
  echo "dvl_velocity_noise_std_mps=0"
  echo "dvl_fom_mps=0"
  echo "dvl_mount_assumption=base_link_origin_VISO_POS_zero"
  echo "brov_install=$BROV_INSTALL"
  echo "brov_control_prefix=$(ros2 pkg prefix brov_control)"
  echo "brov_bringup_prefix=$(ros2 pkg prefix brov_bringup)"
  sha256sum "$WORLD" "$PARAMS" "$MISSION" "$DVL_INJECTOR" \
    "$CYCLE_SUPERVISOR" "$START_BARRIER" "$CONTROLLER_CONFIG" \
    "$BROV_SOURCE/brov_control/brov_control/model_based_controller.py" \
    "$BROV_SOURCE/brov_control/brov_control/model_based_controller_node.py" \
    "$BROV_SOURCE/brov_bringup/launch/sim2sim_case_c_model_based.launch.py"
} >"$RUN_DIR/manifest.txt"
cp "$WORLD" "$PARAMS" "$MISSION" "$DVL_INJECTOR" \
  "$CYCLE_SUPERVISOR" "$START_BARRIER" "$CONTROLLER_CONFIG" "$RUN_DIR/"

setsid gz sim -s -r -v 2 "$WORLD" >"$RUN_DIR/gazebo.log" 2>&1 &
GZ_PID=$!
echo "[case-c-model-based-$FEEDBACK_SOURCE] Gazebo PID=$GZ_PID"
sleep 1

setsid bash -lc "cd '$RUN_DIR' && exec \
  /home/bluerov2_sitl/ardupilot/build/sitl/bin/ardusub \
  -S -w --model JSON --speedup 1 --slave 0 \
  --defaults '/home/bluerov2_sitl/ardupilot/Tools/autotest/default_params/sub-6dof.parm,$PARAMS' \
  --sim-address=127.0.0.1 -I0 \
  --home 55.99541530863445,-3.301022500491058,0.0,0.0" \
  >"$RUN_DIR/ardusub.stdout.log" 2>&1 &
ARDUSUB_PID=$!

setsid bash -lc "cd '$RUN_DIR' && exec \
  /home/bluerov2_sitl/.local/bin/mavproxy.py \
  --daemon \
  --master=tcp:127.0.0.1:5760 --sitl=127.0.0.1:5501 --streamrate=25 \
  --out=udp:127.0.0.1:14552 --out=udp:127.0.0.1:14554 \
  --out=udp:127.0.0.1:14555" >"$RUN_DIR/mavproxy.log" 2>&1 &
MAVPROXY_PID=$!

for _ in $(seq 1 60); do
  if grep -q "online system 1" "$RUN_DIR/mavproxy.log" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
grep -q "online system 1" "$RUN_DIR/mavproxy.log"

PARAMETERS_READY=false
for _ in $(seq 1 60); do
  if kill -0 "$MAVPROXY_PID" 2>/dev/null \
      && test -s "$RUN_DIR/mav.parm" \
      && grep -q '^VISO_TYPE' "$RUN_DIR/mav.parm" \
      && grep -q '^RNGFND1_TYPE' "$RUN_DIR/mav.parm"; then
    PARAMETERS_READY=true
    break
  fi
  sleep 0.25
done
if [[ "$PARAMETERS_READY" != true ]]; then
  echo "MAVProxy did not finish the parameter snapshot" >&2
  exit 1
fi

for specification in \
  AHRS_EKF_TYPE=3 EK2_ENABLE=0 EK3_ENABLE=1 VISO_TYPE=1 \
  EK3_SRC1_POSXY=6 EK3_SRC1_VELXY=6 EK3_SRC1_POSZ=1 \
  EK3_SRC1_VELZ=0 EK3_SRC1_YAW=1 VISO_DELAY_MS=10 VISO_ORIENT=0 \
  VISO_POS_X=0 VISO_POS_Y=0 VISO_POS_Z=0 RNGFND1_TYPE=10 \
  SIM_GPS_DISABLE=1 GPS1_TYPE=0 GPS2_TYPE=0; do
  name=${specification%%=*}
  expected=${specification#*=}
  if ! awk -v name="$name" -v expected="$expected" '
      $1 == name { found=1; if (($2 + 0) != (expected + 0)) exit 2 }
      END { if (!found) exit 1 }
    ' "$RUN_DIR/mav.parm"; then
    echo "postboot parameter mismatch: $specification" >&2
    exit 1
  fi
done
grep -E '^(AHRS_EKF_TYPE|EK2_ENABLE|EK3_ENABLE|EK3_SRC1_|GPS1_TYPE|GPS2_TYPE|RNGFND1_TYPE|SIM_GPS_DISABLE|VISO_)' \
  "$RUN_DIR/mav.parm" >"$RUN_DIR/ekf_params_verified.txt"

setsid ros2 run ros_gz_bridge parameter_bridge \
  '/model/bluerov2_heavy/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
  '/observer_camera_c/image@sensor_msgs/msg/Image[gz.msgs.Image' \
  --ros-args \
  -r /model/bluerov2_heavy/odometry:=/brov/sim/gazebo_odometry_raw \
  -r /observer_camera_c/image:=/brov/sim/observer_camera_c \
  >"$RUN_DIR/gt_bridge.log" 2>&1 &
BRIDGE_PID=$!

for _ in $(seq 1 40); do
  if timeout 1 ros2 topic echo --once /brov/sim/gazebo_odometry_raw \
      >/dev/null 2>&1; then
    break
  fi
done
timeout 2 ros2 topic echo --once /brov/sim/gazebo_odometry_raw \
  >"$RUN_DIR/gt_pre_bringup.txt"

setsid env MAVLINK20=1 /usr/bin/python3 "$DVL_INJECTOR" \
  --connection udpin:0.0.0.0:14555 \
  --topic /brov/sim/gazebo_odometry_raw \
  --rate-hz 10 --far-rate-hz 5 --range-transition-m 3 \
  --delay-s 0 --velocity-noise-std 0 --fom-mps 0 \
  --rangefinder --seabed-world-z -10 --rangefinder-max-m 50 \
  --bottom-lock-min-world-z -9.7 --bottom-lock-max-world-z -0.5 \
  --confirm-sitl >"$RUN_DIR/dvl_injector.log" 2>&1 &
DVL_PID=$!

env MAVLINK20=1 /usr/bin/python3 "$ORIGIN_HELPER" \
  --connection udpin:0.0.0.0:14554 --timeout-s 20 \
  >"$RUN_DIR/origin.log" 2>&1

setsid ros2 bag record -o "$RUN_DIR/bag" \
  /clock /brov/sim/gazebo_odometry_raw /brov/sim/observer_camera_c \
  /brov/stage2/dvl_sample /brov/stage2/dvl_schema \
  /brov/stage2/dvl_status /brov/stage2/dvl_valid \
  /brov/observation \
  /brov/model_based/wrench_zup /brov/model_based/wrench_sname \
  /brov/model_based/action /brov/model_based/estimated_wrench_zup \
  /brov/model_based/thruster_force \
  /brov/model_based/thruster_pwm_preview /brov/model_based/enabled \
  /brov/thruster_pwm \
  /brov/control_active /brov/mission_complete /brov/waypoint_idx \
  /brov/target_waypoint \
  /brov/debug/feedback_source /brov/debug/feedback_timing \
  /brov/debug/feedback_timing_schema \
  /brov/debug/feedback_pos_ned /brov/debug/feedback_vel_ned \
  /brov/debug/feedback_att_quat_ned \
  /brov/debug/feedback_body_rates_frd \
  /brov/debug/gazebo_truth_pos_ned /brov/debug/gazebo_truth_vel_ned \
  /brov/debug/gazebo_truth_att_quat_ned \
  /brov/debug/pos_ned /brov/debug/vel_ned /brov/debug/att_quat_ned \
  /brov/debug/pos_mission /brov/debug/v_body_zup \
  /brov/debug/v_desired_body_zup /brov/debug/q_desired_zup \
  /brov/debug/servo_output_us /brov/odometry/local \
  /brov/odometry/local_with_session /brov/odometry/session_id \
  /rosout /parameter_events >"$RUN_DIR/rosbag.log" 2>&1 &
BAG_PID=$!

setsid ros2 launch brov_bringup sim2sim_case_c_model_based.launch.py \
  connection:=udpin:0.0.0.0:14552 \
  feedback_source:="$FEEDBACK_SOURCE" \
  start_gazebo_truth_bridge:=false \
  send_pwm:=true arm:=true \
  >"$RUN_DIR/brov_launch.log" 2>&1 &
LAUNCH_PID=$!

for _ in $(seq 1 60); do
  if timeout 1 ros2 topic echo --once /brov/stage2/dvl_valid 2>/dev/null \
      | grep -q "data: true"; then
    break
  fi
done
timeout 2 ros2 topic echo --once /brov/stage2/dvl_valid | grep -q "data: true"

for _ in $(seq 1 60); do
  if timeout 1 ros2 topic echo --once /brov/debug/feedback_vel_ned \
      >/dev/null 2>&1; then
    break
  fi
done
timeout 2 ros2 topic echo --once /brov/debug/feedback_vel_ned >/dev/null

ros2 param get /brov_obs_node feedback_source >"$RUN_DIR/feedback_source.txt"
ros2 param get /brov_obs_node cruise_speed >"$RUN_DIR/cruise_speed.txt"
ros2 param get /brov_obs_node waypoints >"$RUN_DIR/waypoints.txt"
ros2 param get /brov_obs_node heading_mode >"$RUN_DIR/heading_mode.txt"
ros2 param get /brov_obs_node loop >"$RUN_DIR/loop.txt"
ros2 param dump /brov_obs_node >"$RUN_DIR/obs_params.yaml"
ros2 param dump /brov_model_based_controller >"$RUN_DIR/model_controller_params.yaml"
grep -q "$FEEDBACK_SOURCE" "$RUN_DIR/feedback_source.txt"
grep -Eq "0\.5(0*)?$" "$RUN_DIR/cruise_speed.txt"
grep -q "random_at_waypoint" "$RUN_DIR/heading_mode.txt"
grep -qi "true" "$RUN_DIR/loop.txt"
grep -q "5.0,0,0.20" "$RUN_DIR/waypoints.txt"
grep -q "5.0,5.0,0.20" "$RUN_DIR/waypoints.txt"

test "$(ros2 node list 2>/dev/null | grep -c '^/brov_obs_node$' || true)" -eq 1
test "$(ros2 node list 2>/dev/null | grep -c '^/brov_model_based_controller$' || true)" -eq 1
ros2 topic info -v /brov/thruster_pwm >"$RUN_DIR/thruster_pwm_authority.txt"
grep -q "Publisher count: 1" "$RUN_DIR/thruster_pwm_authority.txt"
ros2 topic info -v /brov/sim/gazebo_odometry_raw >"$RUN_DIR/gt_authority.txt"
grep -q "Publisher count: 1" "$RUN_DIR/gt_authority.txt"

timeout 2 ros2 topic echo --once /brov/sim/gazebo_odometry_raw \
  >"$RUN_DIR/gt_before_arm.txt"
timeout 2 ros2 topic echo --once /brov/debug/vel_ned \
  >"$RUN_DIR/ekf_before_arm.txt"

setsid /usr/bin/python3 "$CYCLE_SUPERVISOR" \
  --timeout-s 120 --debounce-samples 3 \
  >"$RUN_DIR/cycle_supervisor.json" &
SUPERVISOR_PID=$!
sleep 0.5
kill -0 "$SUPERVISOR_PID"

echo "[case-c-model-based-$FEEDBACK_SOURCE] preflight passed; arming"
ARMED=false
for attempt in $(seq 1 20); do
  timeout 5 ros2 service call /brov/arm_control std_srvs/srv/Trigger "{}" \
    >"$RUN_DIR/arm_attempt_${attempt}.txt" 2>&1 || true
  if grep -Eq "success=(True|true)|success: true" \
      "$RUN_DIR/arm_attempt_${attempt}.txt"; then
    ARMED=true
    cp "$RUN_DIR/arm_attempt_${attempt}.txt" "$RUN_DIR/arm_service.txt"
    break
  fi
  sleep 0.10
done
if [[ "$ARMED" != true ]]; then
  echo "arm transaction failed" >&2
  exit 1
fi

/usr/bin/python3 "$START_BARRIER" --target-world-z -6.2 --timeout-s 15 \
  >"$RUN_DIR/gt_start_barrier.json"

timeout 5 ros2 service call /brov/start_control std_srvs/srv/Trigger "{}" \
  >"$RUN_DIR/start_service.txt" 2>&1 || true
STARTED=false
if grep -Eq "success=(True|true)|success: true" "$RUN_DIR/start_service.txt"; then
  STARTED=true
else
  for _ in $(seq 1 10); do
    timeout 1 ros2 topic echo --once /brov/control_active \
      >"$RUN_DIR/control_active_after_start.txt" 2>&1 || true
    if grep -Eq "data: true" "$RUN_DIR/control_active_after_start.txt"; then
      STARTED=true
      break
    fi
    sleep 0.10
  done
fi
if [[ "$STARTED" != true ]]; then
  echo "start transaction failed and control_active never became true" >&2
  exit 1
fi

# model_based_controller_node only accepts /brov/model_based/start once base
# control is active and it has seen at least one fresh post-active
# observation; retry briefly rather than racing the first callback.
MODEL_STARTED=false
for attempt in $(seq 1 20); do
  timeout 5 ros2 service call /brov/model_based/start std_srvs/srv/Trigger "{}" \
    >"$RUN_DIR/model_start_attempt_${attempt}.txt" 2>&1 || true
  if grep -Eq "success=(True|true)|success: true" \
      "$RUN_DIR/model_start_attempt_${attempt}.txt"; then
    MODEL_STARTED=true
    cp "$RUN_DIR/model_start_attempt_${attempt}.txt" "$RUN_DIR/model_start_service.txt"
    break
  fi
  sleep 0.10
done
if [[ "$MODEL_STARTED" != true ]]; then
  echo "model_based/start transaction failed" >&2
  exit 1
fi

echo "[case-c-model-based-$FEEDBACK_SOURCE] active; waiting for 0->1->2->3->0"
wait "$SUPERVISOR_PID"
SUPERVISOR_PID=

ros2 service call /brov/model_based/stop std_srvs/srv/Trigger "{}" \
  >"$RUN_DIR/model_stop_service.txt"
ros2 service call /brov/stop_control std_srvs/srv/Trigger "{}" \
  >"$RUN_DIR/stop_service.txt"
ros2 service call /brov/disarm_control std_srvs/srv/Trigger "{}" \
  >"$RUN_DIR/disarm_service.txt"
sleep 1
timeout 2 ros2 topic echo --once /brov/debug/servo_output_us \
  >"$RUN_DIR/servo_after_stop.txt" || true

echo "[case-c-model-based-$FEEDBACK_SOURCE] one full cycle completed"

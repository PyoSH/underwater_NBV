#!/usr/bin/env bash
# Frozen-policy 0.5 m/s regression on the no-GPS DVL/INS/AHRS EKF chain.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /tmp/stage2_dvl_policy_0p5_<run_id>" >&2
  exit 2
fi
RUN_DIR=$1
case "$RUN_DIR" in
  /tmp/stage2_dvl_policy_0p5_*) ;;
  *) echo "RUN_DIR must be an explicit /tmp/stage2_dvl_policy_0p5_* path" >&2; exit 2 ;;
esac

WORLD=/tmp/stage2_bluerov2_heavy_underwater_8p5m.sdf
PARAMS=/tmp/stage2_dvl_only.parm
BROV_INSTALL=/tmp/brov_stage1_install
POLICY=/home/bluerov2_sitl/brov_ros2/artifacts/policies/demo_policy/policy.pt
ROS_DOMAIN_ID=42
export ROS_DOMAIN_ID

GZ_PID=
ARDUSUB_PID=
MAVPROXY_PID=
DVL_PID=
LAUNCH_PID=
BAG_PID=

stop_pid() {
  local pid=${1:-}
  local signal=${2:-INT}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -"$signal" "$pid" 2>/dev/null || true
  fi
}

stop_group() {
  local pid=${1:-}
  local signal=${2:-INT}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -"$signal" -- -"$pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo "[stage2-policy] stopping control and all task-owned processes"
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    timeout 4 ros2 service call /brov/stop_control std_srvs/srv/Trigger "{}" \
      >"$RUN_DIR/cleanup_stop.txt" 2>&1 || true
    timeout 4 ros2 service call /brov/disarm_control std_srvs/srv/Trigger "{}" \
      >"$RUN_DIR/cleanup_disarm.txt" 2>&1 || true
  fi
  stop_group "$BAG_PID" INT
  stop_group "$LAUNCH_PID" INT
  stop_pid "$DVL_PID" INT
  stop_pid "$MAVPROXY_PID" INT
  stop_pid "$ARDUSUB_PID" INT
  stop_pid "$GZ_PID" INT
  sleep 3
  stop_group "$BAG_PID" TERM
  stop_group "$LAUNCH_PID" TERM
  stop_pid "$DVL_PID" TERM
  stop_pid "$MAVPROXY_PID" TERM
  stop_pid "$ARDUSUB_PID" TERM
  stop_pid "$GZ_PID" TERM
  sleep 1
}

finish() {
  local status=$?
  trap - EXIT INT TERM
  cleanup
  exit "$status"
}
trap finish EXIT INT TERM

mkdir -p "$RUN_DIR"
test -f "$WORLD"
test -f "$PARAMS"
test -f "$POLICY"
echo "[stage2-policy] no-GPS realistic DVL profile; policy=$(sha256sum "$POLICY" | cut -d' ' -f1)"

export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH:-}
export GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH:-}
export GZ_GUI_PLUGIN_PATH=${GZ_GUI_PLUGIN_PATH:-}
set +u
source /home/bluerov2_sitl/gz_ws/gazebo_exports.sh
set -u
gz sim -s -r -v 2 "$WORLD" >"$RUN_DIR/gazebo.log" 2>&1 &
GZ_PID=$!
echo "[stage2-policy] Gazebo PID=$GZ_PID"
sleep 1

(
  cd "$RUN_DIR"
  exec /home/bluerov2_sitl/ardupilot/build/sitl/bin/ardusub \
    -S -w --model JSON --speedup 1 --slave 0 \
    --defaults "/home/bluerov2_sitl/ardupilot/Tools/autotest/default_params/sub-6dof.parm,$PARAMS" \
    --sim-address=127.0.0.1 -I0 \
    --home 55.99541530863445,-3.301022500491058,0.0,0.0
) >"$RUN_DIR/ardusub.stdout.log" 2>&1 &
ARDUSUB_PID=$!
echo "[stage2-policy] ArduSub PID=$ARDUSUB_PID"
sleep 1

(
  cd "$RUN_DIR"
  exec /home/bluerov2_sitl/.local/bin/mavproxy.py --daemon \
    --master=tcp:127.0.0.1:5760 --sitl=127.0.0.1:5501 \
    --streamrate=25 \
    --out=udp:127.0.0.1:14552 \
    --out=udp:127.0.0.1:14554 \
    --out=udp:127.0.0.1:14555
) >"$RUN_DIR/mavproxy.log" 2>&1 &
MAVPROXY_PID=$!
echo "[stage2-policy] MAVProxy PID=$MAVPROXY_PID"

for _ in $(seq 1 30); do
  if grep -q "online system 1" "$RUN_DIR/mavproxy.log" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
grep -q "online system 1" "$RUN_DIR/mavproxy.log"

set +u
source /opt/ros/humble/setup.bash
source /home/bluerov2_sitl/colcon_ws/install/setup.bash
source "$BROV_INSTALL/setup.bash"
set -u

# Start the recorder before bringup so transient-local provenance and every
# control-state edge are present in the same artifact.
setsid ros2 bag record -o "$RUN_DIR/bag" \
  /brov/sim/gazebo_odometry_raw \
  /brov/stage2/dvl_sample /brov/stage2/dvl_schema \
  /brov/stage2/dvl_status /brov/stage2/dvl_valid \
  /brov/observation /brov/action \
  /brov/policy/action_raw /brov/policy/wrench_requested \
  /brov/policy/wrench_after_thruster_limit \
  /brov/policy/thruster_force_requested \
  /brov/policy/thruster_force_limited \
  /brov/policy/thruster_pwm_requested \
  /brov/policy/thruster_pwm_preview /brov/thruster_pwm \
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
echo "[stage2-policy] rosbag PID=$BAG_PID"

setsid ros2 launch brov_bringup sim2sim_0p5_ab.launch.py \
  feedback_source:=mavlink_ekf \
  connection:=udpin:0.0.0.0:14552 \
  policy_path:="$POLICY" send_pwm:=true arm:=true \
  >"$RUN_DIR/brov_launch.log" 2>&1 &
LAUNCH_PID=$!
echo "[stage2-policy] BROV launch PID=$LAUNCH_PID"

MAVLINK20=1 /usr/bin/python3 /tmp/stage2_sitl_dvl_injector.py \
  --connection udpin:0.0.0.0:14555 \
  --topic /brov/sim/gazebo_odometry_raw \
  --rate-hz 15 --delay-s 0.10 --velocity-noise-std 0.003 --fom-mps 0.003 \
  --confirm-sitl >"$RUN_DIR/dvl_injector.log" 2>&1 &
DVL_PID=$!
echo "[stage2-policy] DVL injector PID=$DVL_PID"

MAVLINK20=1 /usr/bin/python3 /tmp/stage2_set_ekf_origin.py \
  --connection udpin:0.0.0.0:14554 --timeout-s 20 \
  >"$RUN_DIR/origin.log" 2>&1

for _ in $(seq 1 45); do
  if timeout 1 ros2 topic echo --once /brov/stage2/dvl_valid 2>/dev/null \
      | grep -q "data: true"; then
    break
  fi
done
timeout 2 ros2 topic echo --once /brov/stage2/dvl_valid | grep -q "data: true"

for _ in $(seq 1 45); do
  if timeout 1 ros2 topic echo --once /brov/debug/feedback_vel_ned \
      >/dev/null 2>&1; then
    break
  fi
done
timeout 2 ros2 topic echo --once /brov/debug/feedback_vel_ned >/dev/null

ros2 param get /brov_obs_node feedback_source >"$RUN_DIR/feedback_source.txt"
ros2 param get /brov_obs_node cruise_speed >"$RUN_DIR/cruise_speed.txt"
ros2 param get /brov_obs_node send_pwm >"$RUN_DIR/send_pwm.txt"
ros2 param get /brov_obs_node arm >"$RUN_DIR/arm_permission.txt"
grep -q "mavlink_ekf" "$RUN_DIR/feedback_source.txt"
grep -Eq "0\.5(0*)?$" "$RUN_DIR/cruise_speed.txt"
grep -qi "true" "$RUN_DIR/send_pwm.txt"
grep -qi "true" "$RUN_DIR/arm_permission.txt"
NODE_COUNT=$(ros2 node list 2>/dev/null | grep -c '^/brov_obs_node$' || true)
if [[ "$NODE_COUNT" -ne 1 ]]; then
  echo "[stage2-policy] expected exactly one /brov_obs_node; found $NODE_COUNT" >&2
  exit 1
fi
ros2 topic info -v /brov/thruster_pwm >"$RUN_DIR/thruster_pwm_authority.txt"
grep -q "Publisher count: 1" "$RUN_DIR/thruster_pwm_authority.txt"
grep -q "Subscription count: 2" "$RUN_DIR/thruster_pwm_authority.txt"
ros2 topic info -v /brov/sim/gazebo_odometry_raw >"$RUN_DIR/gt_authority.txt"
grep -q "Publisher count: 1" "$RUN_DIR/gt_authority.txt"

echo "[stage2-policy] preflight passed; arming and starting 0.5 m/s straight mission"
ros2 service call /brov/arm_control std_srvs/srv/Trigger "{}" \
  >"$RUN_DIR/arm_service.txt"
grep -Eq "success=(True|true)|success: true" "$RUN_DIR/arm_service.txt"
ros2 service call /brov/start_control std_srvs/srv/Trigger "{}" \
  >"$RUN_DIR/start_service.txt"
grep -Eq "success=(True|true)|success: true" "$RUN_DIR/start_service.txt"

sleep 0.5
MISSION_COMPLETE=false
for _ in $(seq 1 900); do
  if timeout 1 ros2 topic echo --once /brov/mission_complete 2>/dev/null \
      | grep -q "data: true"; then
    MISSION_COMPLETE=true
    break
  fi
  if timeout 1 ros2 topic echo --once /brov/control_active 2>/dev/null \
      | grep -q "data: false"; then
    echo "[stage2-policy] control became inactive before mission completion" >&2
    exit 1
  fi
  sleep 0.04
done
if [[ "$MISSION_COMPLETE" != true ]]; then
  echo "[stage2-policy] mission did not complete before timeout" >&2
  exit 1
fi

echo "[stage2-policy] mission complete; holding 2 s for terminal-state logging"
sleep 2
ros2 service call /brov/stop_control std_srvs/srv/Trigger "{}" \
  >"$RUN_DIR/stop_service.txt"
ros2 service call /brov/disarm_control std_srvs/srv/Trigger "{}" \
  >"$RUN_DIR/disarm_service.txt"
echo "[stage2-policy] frozen-policy regression completed successfully"

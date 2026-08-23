#!/usr/bin/env bash
# Reproducible Stage-2 no-GPS DVL/INS/AHRS pulse experiment in the Edo SITL.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /tmp/stage2_dvl_only_<run_id> [realistic|oracle]" >&2
  exit 2
fi
RUN_DIR=$1
PROFILE=${2:-realistic}
case "$RUN_DIR" in
  /tmp/stage2_dvl_only_*) ;;
  *) echo "RUN_DIR must be an explicit /tmp/stage2_dvl_only_* path" >&2; exit 2 ;;
esac

WORLD=/tmp/stage2_bluerov2_heavy_underwater_8p5m.sdf
case "$PROFILE" in
  realistic)
    PARAMS=/tmp/stage2_dvl_only.parm
    DVL_DELAY_S=0.10
    DVL_NOISE_STD=0.003
    DVL_FOM_MPS=0.003
    ;;
  oracle)
    PARAMS=/tmp/stage2_dvl_oracle.parm
    DVL_DELAY_S=0.0
    DVL_NOISE_STD=0.0
    DVL_FOM_MPS=0.0
    ;;
  *)
    echo "profile must be realistic or oracle" >&2
    exit 2
    ;;
esac
BROV_INSTALL=/tmp/brov_stage1_install
ROS_DOMAIN_ID=42
export ROS_DOMAIN_ID

GZ_PID=
ARDUSUB_PID=
MAVPROXY_PID=
DVL_PID=
OBSERVER_PID=
BAG_PID=

stop_pid() {
  local pid=${1:-}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" 2>/dev/null || true
  fi
}

stop_group() {
  local pid=${1:-}
  local signal=${2:-INT}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -"$signal" -- -"$pid" 2>/dev/null || true
  fi
}

term_pid() {
  local pid=${1:-}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo "[stage2-run] stopping recorder and all task-owned processes"
  stop_pid "$BAG_PID"
  stop_group "$OBSERVER_PID" INT
  stop_pid "$DVL_PID"
  stop_pid "$MAVPROXY_PID"
  stop_pid "$ARDUSUB_PID"
  stop_pid "$GZ_PID"
  sleep 2
  # Some ROS launch parents do not exit on SIGINT when detached.  Escalate only
  # the exact PIDs created by this run; never match or kill unrelated processes.
  term_pid "$BAG_PID"
  stop_group "$OBSERVER_PID" TERM
  term_pid "$DVL_PID"
  term_pid "$MAVPROXY_PID"
  term_pid "$ARDUSUB_PID"
  term_pid "$GZ_PID"
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
echo "[stage2-run] profile=$PROFILE params=$PARAMS delay=$DVL_DELAY_S noise=$DVL_NOISE_STD fom=$DVL_FOM_MPS"

export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH:-}
export GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH:-}
export GZ_GUI_PLUGIN_PATH=${GZ_GUI_PLUGIN_PATH:-}
set +u
source /home/bluerov2_sitl/gz_ws/gazebo_exports.sh
set -u
gz sim -s -r -v 2 "$WORLD" >"$RUN_DIR/gazebo.log" 2>&1 &
GZ_PID=$!
echo "[stage2-run] Gazebo PID=$GZ_PID"
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
echo "[stage2-run] ArduSub PID=$ARDUSUB_PID"
sleep 1

(
  cd "$RUN_DIR"
  exec /home/bluerov2_sitl/.local/bin/mavproxy.py --daemon \
    --master=tcp:127.0.0.1:5760 --sitl=127.0.0.1:5501 \
    --streamrate=25 \
    --out=udp:127.0.0.1:14552 \
    --out=udp:127.0.0.1:14553 \
    --out=udp:127.0.0.1:14554 \
    --out=udp:127.0.0.1:14555
) >"$RUN_DIR/mavproxy.log" 2>&1 &
MAVPROXY_PID=$!
echo "[stage2-run] MAVProxy PID=$MAVPROXY_PID"

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
set -u
MAVLINK20=1 /usr/bin/python3 /tmp/stage2_sitl_dvl_injector.py \
  --connection udpin:0.0.0.0:14555 \
  --topic /brov/sim/gazebo_odometry_raw \
  --rate-hz 15 --delay-s "$DVL_DELAY_S" \
  --velocity-noise-std "$DVL_NOISE_STD" --fom-mps "$DVL_FOM_MPS" \
  --confirm-sitl >"$RUN_DIR/dvl_injector.log" 2>&1 &
DVL_PID=$!
echo "[stage2-run] DVL injector PID=$DVL_PID"

ros2 bag record -o "$RUN_DIR/bag" \
  /brov/sim/gazebo_odometry_raw \
  /brov/stage2/dvl_sample /brov/stage2/dvl_schema \
  /brov/stage2/dvl_status /brov/stage2/dvl_valid \
  /brov/stage2/phase /brov/stage2/pulse_pwm \
  /brov/stage2/mavlink_snapshot \
  /brov/debug/gazebo_truth_pos_ned /brov/debug/gazebo_truth_vel_ned \
  /brov/debug/gazebo_truth_att_quat_ned \
  /brov/debug/feedback_pos_ned /brov/debug/feedback_vel_ned \
  /brov/debug/feedback_att_quat_ned /brov/debug/feedback_body_rates_frd \
  /brov/debug/feedback_timing /brov/debug/feedback_timing_schema \
  /brov/debug/pos_ned /brov/debug/vel_ned /brov/debug/att_quat_ned \
  /brov/debug/servo_output_us /brov/odometry/local \
  /brov/odometry/local_with_session /brov/odometry/session_id \
  /brov/control_active /brov/thruster_pwm /rosout /parameter_events \
  >"$RUN_DIR/rosbag.log" 2>&1 &
BAG_PID=$!
echo "[stage2-run] rosbag PID=$BAG_PID"

MAVLINK20=1 /usr/bin/python3 /tmp/stage2_set_ekf_origin.py \
  --connection udpin:0.0.0.0:14554 --timeout-s 20 \
  >"$RUN_DIR/origin.log" 2>&1

set +u
source "$BROV_INSTALL/setup.bash"
set -u
setsid ros2 launch brov_bringup base.launch.py \
  vehicle_config:="$BROV_INSTALL/brov_base/share/brov_base/config/vehicle_sitl.yaml" \
  connection:=udpin:0.0.0.0:14552 \
  feedback_source:=mavlink_ekf \
  gazebo_truth_logging_enabled:=true \
  gazebo_truth_topic:=/brov/sim/gazebo_odometry_raw \
  send_pwm:=false arm:=false \
  require_pool_localization:=false require_resolved_mission:=false \
  >"$RUN_DIR/brov_observer.log" 2>&1 &
OBSERVER_PID=$!
echo "[stage2-run] BROV observer PID=$OBSERVER_PID"

for _ in $(seq 1 30); do
  if timeout 1 ros2 topic echo --once /brov/stage2/dvl_valid 2>/dev/null \
      | grep -q "data: true"; then
    break
  fi
done
timeout 2 ros2 topic echo --once /brov/stage2/dvl_valid \
  | grep -q "data: true"

for _ in $(seq 1 35); do
  if timeout 1 ros2 topic echo --once /brov/debug/feedback_vel_ned \
      >/dev/null 2>&1; then
    break
  fi
done
timeout 2 ros2 topic echo --once /brov/debug/feedback_vel_ned >/dev/null

echo "[stage2-run] DVL and EKF feedback healthy; beginning policy-free pulse"
/usr/bin/python3 /tmp/stage2_sitl_axis_pulse.py \
  --connection udpin:0.0.0.0:14553 \
  --amplitude 0.106 --pulse-s 3.0 --settle-s 6.0 \
  --initial-neutral-s 3.0 --final-neutral-s 3.0 --rate-hz 25 \
  --vertical-trim 0.01 --max-speed-mps 0.8 --max-body-rate-rps 0.8 \
  --output-csv "$RUN_DIR/pulse.csv" --confirm-sitl \
  | tee "$RUN_DIR/pulse.stdout.log"

echo "[stage2-run] pulse completed successfully"

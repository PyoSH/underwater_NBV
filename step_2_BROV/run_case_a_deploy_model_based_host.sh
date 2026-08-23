#!/usr/bin/env bash
# Build/sync the MK2 ROS overlay, then run one fresh Gazebo Case-A cycle
# with the classical model-based controller (causal-isolation baseline).

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 gazebo_truth|mavlink_ekf [run_tag]" >&2
  exit 2
fi

feedback_source=$1
case "$feedback_source" in
  gazebo_truth|mavlink_ekf) ;;
  *) echo "feedback source must be gazebo_truth or mavlink_ekf" >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
container_name=${BROV_MK2_CONTAINER:-brov-sim2sim-sitl}
target_repo=/home/bluerov2_sitl/brov_ros2
run_tag=${2:-$(date +%Y%m%d_%H%M%S)}
run_dir="$target_repo/runtime/experiments/case_a_model_based_${run_tag}_${feedback_source}"

declare -a support_files=(
  stage2_bluerov2_heavy_underwater_8p5m.sdf
  stage2_waterlinked_default.parm
  stage2_sitl_dvl_injector.py
  stage2_set_ekf_origin.py
  stage2_wait_case_a_cycle.py
  stage2_wait_gt_start.py
  run_case_a_deploy_model_based.sh
)

for file_name in "${support_files[@]}"; do
  test -f "$script_dir/$file_name"
  docker cp "$script_dir/$file_name" "$container_name:/tmp/$file_name"
done

docker exec -w "$target_repo" "$container_name" bash -lc '
  set -e
  source /opt/ros/humble/setup.bash
  source /home/bluerov2_sitl/colcon_ws/install/setup.bash
  colcon --log-base log_mk2 build \
    --build-base build_mk2 \
    --install-base install_mk2 \
    --symlink-install \
    --packages-select \
      brov_interfaces brov_base brov_control brov_localization \
      brov_mission brov_perception brov_viz brov_bringup
  source install_mk2/setup.bash
  ros2 pkg executables brov_control | grep -q "model_based_controller_node"
  ros2 launch brov_bringup sim2sim_case_a_model_based.launch.py --show-args >/dev/null
'

echo "[case-a-model-based-host] fresh run: $run_dir"
docker exec -i "$container_name" bash "/tmp/run_case_a_deploy_model_based.sh" \
  "$run_dir" "$feedback_source"
echo "[case-a-model-based-host] completed: $run_dir"

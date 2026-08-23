#!/usr/bin/env bash
# Build/sync cmg_deploy inside the SITL container, then run one fresh
# Gazebo hover cycle. Mirrors run_mk2_case_a_deploy_host.sh.
#
# Archived (2026-08-23): superseded/paused, kept for reference only. This
# script lives in legacy/cmg_deploy/ while the shared stage2_* support files
# it depends on remain at the step_2_BROV/ root (still used by the current
# MK2 pipeline) -- hence the split shared_dir/script_dir sourcing below.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shared_dir=$(cd -- "$script_dir/../.." && pwd)
container_name=${BROV_MK2_CONTAINER:-brov-sim2sim-sitl}
target_repo=/home/bluerov2_sitl/brov_ros2
run_tag=${1:-$(date +%Y%m%d_%H%M%S)}
run_dir="$target_repo/runtime/experiments/sim2sim_cmg_hover_${run_tag}"

declare -a shared_support_files=(
  stage2_bluerov2_heavy_underwater_8p5m.sdf
  stage2_waterlinked_default.parm
  stage2_sitl_dvl_injector.py
  stage2_set_ekf_origin.py
  stage2_wait_gt_start.py
)
declare -a local_support_files=(
  stage2_wait_cmg_hover.py
  run_cmg_sim2sim_deploy.sh
)

for file_name in "${shared_support_files[@]}"; do
  test -f "$shared_dir/$file_name"
  docker cp "$shared_dir/$file_name" "$container_name:/tmp/$file_name"
done
for file_name in "${local_support_files[@]}"; do
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
      brov_mission brov_perception brov_viz brov_bringup cmg_deploy
  source install_mk2/setup.bash
  ros2 pkg executables cmg_deploy | grep -q "cmg_policy_node"
  ros2 launch cmg_deploy cmg_sim2sim.launch.py --show-args >/dev/null
'

echo "[cmg-host] fresh run: $run_dir (state_source=${CMG_STATE_SOURCE:-mavlink_ekf})"
docker exec -i -e CMG_STATE_SOURCE="${CMG_STATE_SOURCE:-mavlink_ekf}" "$container_name" \
  bash "/tmp/run_cmg_sim2sim_deploy.sh" "$run_dir"
echo "[cmg-host] completed: $run_dir"

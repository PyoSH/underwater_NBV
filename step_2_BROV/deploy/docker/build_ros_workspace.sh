#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source /workspace/deploy/docker/ros_env.sh

cd "${BROV_ROS_WS}"
colcon build --symlink-install --event-handlers console_direct+

# 이번 build에서 생성된 overlay로 package discovery까지 확인한다.
# shellcheck disable=SC1091
set +u
source "${BROV_ROS_WS}/install/setup.bash"
set -u
for package in brov_base brov_control brov_perception brov_bringup; do
    ros2 pkg prefix "${package}" >/dev/null
done

echo "BROV ROS 2 workspace ready: ${BROV_ROS_WS}/install"

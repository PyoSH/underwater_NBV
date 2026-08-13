#!/usr/bin/env bash
set -e

if [[ -f /usr/local/bin/brov-ros-env ]]; then
    # Docker image에 포함된 환경 로더. bind-mounted colcon overlay가 있으면 함께 로드한다.
    # shellcheck disable=SC1091
    source /usr/local/bin/brov-ros-env
else
    # 개발 중 이미지 재빌드 전에도 base ROS 환경은 유지한다.
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

exec "$@"

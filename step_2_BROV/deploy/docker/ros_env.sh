#!/usr/bin/env bash
# ROS 2 Humble과 BROV colcon overlay를 현재 shell에 로드한다.
# 이 파일은 실행하는 대신 source해도 shell option을 변경하지 않는다.

_brov_ros_setup="/opt/ros/humble/setup.bash"
_brov_overlay_setup="/workspace/deploy/ros2_ws/install/setup.bash"
_brov_restore_nounset=false

# ROS/colcon setup은 일부 선택 변수를 `${name}`으로 직접 참조한다. 호출 shell이
# `set -u`인 경우 setup 동안만 nounset을 해제하고 마지막에 원상 복구한다.
case "$-" in
    *u*)
        _brov_restore_nounset=true
        set +u
        ;;
esac

if [[ -f "${_brov_ros_setup}" ]]; then
    # ROS setup 스크립트가 설정하는 환경변수를 현재 shell에 유지해야 한다.
    # shellcheck disable=SC1090
    source "${_brov_ros_setup}"
fi

if [[ -f "${_brov_overlay_setup}" ]]; then
    # colcon build 전에는 overlay가 없을 수 있으므로 선택적으로 로드한다.
    # shellcheck disable=SC1090
    source "${_brov_overlay_setup}"
fi

export BROV_ROS_WS="/workspace/deploy/ros2_ws"
case ":${PYTHONPATH:-}:" in
    *:/workspace:*) ;;
    *) export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

if [[ "${_brov_restore_nounset}" == "true" ]]; then
    set -u
fi

unset _brov_ros_setup _brov_overlay_setup _brov_restore_nounset

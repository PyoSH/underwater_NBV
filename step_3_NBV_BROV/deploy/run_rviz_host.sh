#!/usr/bin/env bash
# NBV 미션 RViz 시각화를 **호스트(또는 실험 laptop)** 에서 띄운다 (2026-09-11).
#   nbv_viz_node (brov_viz, 표준 메시지만 사용 → 워크스페이스 빌드 불필요) + rviz2
# 컨테이너 SITL 은 host 네트워크라 ROS_DOMAIN_ID 만 맞으면 DDS 로 그대로 보인다. 실기도 동일.
#   ./run_rviz_host.sh            # 실행
#   ./run_rviz_host.sh stop
set -euo pipefail
BROV_ROS2=${BROV_ROS2:-/home/pyo/Programing/brov_ros2-main}
DEP=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DEP/../.." && pwd)
MESH=${OBJECT_MESH_URI:-file://$REPO/robots/data/real_object/plaster_mold.obj}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
PIDFILE=/tmp/nbv_rviz_host.pids

if [[ "${1:-}" == "stop" ]]; then
  [[ -f "$PIDFILE" ]] && { xargs -r kill -INT < "$PIDFILE" 2>/dev/null || true; sleep 1; xargs -r kill -KILL < "$PIDFILE" 2>/dev/null || true; rm -f "$PIDFILE"; }
  echo "stopped"; exit 0
fi
test -f "$BROV_ROS2/brov_viz/brov_viz/nbv_viz_node.py"
: > "$PIDFILE"
# 셸에 남은 LD_LIBRARY_PATH(cuda, gazebo-11)·conda 경로가 rviz2 의 GL 컨텍스트 생성을 깨뜨린다
# ("Failed to create an OpenGL context. GLXBadDrawable", 2026-09-11 실측) → 깨끗한 환경에서
# ROS 만 source 해서 띄운다. RViz/tf2_ros 는 시스템 python(3.10) 이어야 한다.
CLEAN=(env -i HOME="$HOME" USER="${USER:-$(id -un)}" DISPLAY="${DISPLAY:-:1}" XAUTHORITY="${XAUTHORITY:-}"
       PATH=/usr/local/bin:/usr/bin:/bin TERM="${TERM:-xterm}" ROS_DOMAIN_ID="$ROS_DOMAIN_ID"
       ROS_LOCALHOST_ONLY=0
       FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$DEP/fastdds_udp_only.xml}")
# 컨테이너(host 네트워크)와는 UDP 로만: SHM 전송은 /dev/shm 미공유로 데이터가 안 온다 (fastdds_udp_only.xml)
setsid "${CLEAN[@]}" bash -c "source /opt/ros/humble/setup.bash && exec /usr/bin/python3 '$BROV_ROS2/brov_viz/brov_viz/nbv_viz_node.py' --ros-args \
  --params-file '$BROV_ROS2/brov_viz/config/nbv_viz.yaml' -p 'object_mesh_uri:=$MESH'" \
  > /tmp/nbv_viz_node.log 2>&1 &
echo $! >> "$PIDFILE"
setsid "${CLEAN[@]}" bash -c "source /opt/ros/humble/setup.bash && exec rviz2 -d '$BROV_ROS2/brov_viz/rviz/nbv_mission.rviz'" \
  > /tmp/nbv_rviz.log 2>&1 &
echo $! >> "$PIDFILE"
echo "nbv_viz_node + rviz2 started (ROS_DOMAIN_ID=$ROS_DOMAIN_ID, mesh=$MESH). 로그: /tmp/nbv_viz_node.log /tmp/nbv_rviz.log"

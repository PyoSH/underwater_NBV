#!/usr/bin/env bash
# Gazebo GUI 를 **호스트에서** GPU 가속 sidecar 컨테이너로 띄운다 (2026-09-11).
#
# 왜 sidecar 인가: SITL 컨테이너(bluerov2_sitl)에는 NVIDIA 라이브러리가 없어 NVIDIA X 서버
# 위에서 Qt Quick 의 GL 컨텍스트가 안 잡히고(창은 뜨지만 전부 검정, DRI3 off/llvmpipe 도 실패),
# 호스트의 gz 는 Fortress 라 Garden 서버와 호환되지 않는다. 같은 이미지를 nvidia 런타임으로
# 한 번 더 띄우면 libGLX_nvidia 가 주입돼 GUI 가 정상 렌더된다. host 네트워크 + 같은
# GZ_PARTITION 이면 gz-transport 가 실행 중인 서버(run_nbv_sitl.sh)를 자동으로 찾는다.
#
# 용례:  ./run_gz_gui_host.sh            # 서버가 떠 있으면 장면이 보인다
#        ./run_gz_gui_host.sh stop
set -euo pipefail
EDO=${EDO:-/home/pyo/Programing/Edo_Project/gazebosim_bluerov2_ardupilot_sitl}
DEP=$(cd "$(dirname "$0")" && pwd)
# deploy/models/plaster_mold 의 mesh 는 저장소의 robots/data/real_object 로 가는 상대 심링크라
# 저장소 루트째 마운트해야 GUI 가 mesh 를 찾는다 (deploy 만 마운트하면 "Unable to find file").
REPO=$(cd "$DEP/../.." && pwd)
NAME=nbv_gz_gui
PARTITION=${GZ_PARTITION:-nbv_sitl}     # run_nbv_sitl.sh 의 서버와 같아야 한다

if [[ "${1:-}" == "stop" ]]; then docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped"; exit 0; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" --runtime nvidia --gpus all --net host \
  -e DISPLAY="${DISPLAY:-:1}" -e QT_X11_NO_MITSHM=1 -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e GZ_PARTITION="$PARTITION" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$EDO/gz_ws:/home/bluerov2_sitl/gz_ws" \
  -v "$EDO/SITL_Models:/home/bluerov2_sitl/SITL_Models" \
  -v "$REPO:/tmp/nbv_repo:ro" \
  bluerov2_sitl:latest bash -c '
    set +u; source /home/bluerov2_sitl/gz_ws/gazebo_exports.sh
    export GZ_SIM_RESOURCE_PATH=/tmp/nbv_repo/step_3_NBV_BROV/deploy/models:$GZ_SIM_RESOURCE_PATH
    exec gz sim -g --gui-config /tmp/nbv_repo/step_3_NBV_BROV/deploy/nbv_gui.config' >/dev/null
echo "GUI sidecar '$NAME' started (DISPLAY=${DISPLAY:-:1}, GZ_PARTITION=$PARTITION). 로그: docker logs $NAME"
# 서버가 떠 있으면 ROV 를 카메라가 따라가게 한다 (best effort: GUI 초기화 뒤 몇 번 시도)
if [[ "${1:-}" != "nofollow" ]]; then
  for _ in $(seq 1 15); do
    sleep 2
    if docker exec "$NAME" bash -c "set +u; source /home/bluerov2_sitl/gz_ws/gazebo_exports.sh; \
        gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 1000 \
        --req 'data: \"bluerov2_heavy\"'" 2>/dev/null | grep -q "data: true"; then
      docker exec "$NAME" bash -c "set +u; source /home/bluerov2_sitl/gz_ws/gazebo_exports.sh; \
        gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean --timeout 1000 \
        --req 'x: -2.5, y: -1.5, z: 1.0'" >/dev/null 2>&1 || true
      echo "GUI 카메라가 bluerov2_heavy 를 follow 한다 (offset -2.5,-1.5,+1.0)"; break
    fi
  done
fi

"""bluerov2_heavy 모델에 RGBD 카메라를 얹은 파생 모델을 **런타임에 생성**한다. 2026-09-10.

왜 상류를 고치지 않는가: `SITL_Models`는 다른 프로젝트와 공유하는 bind mount이고,
복제본을 커밋하면 상류와 드리프트한다(step_2 `sync_vendor.sh` 전례). 그래서 매 실행마다
상류 `model.sdf`에서 파생시킨다.

왜 새 link가 아니라 base_link 안의 sensor 인가: gz에서 `<inertial>` 없는 link는 기본
질량 1 kg를 얻어 14.635 kg 차체의 동역학을 바꾼다. 센서는 link 안에 `<pose>`로 놓으면
질량이 전혀 붙지 않는다.

모델 이름을 `bluerov2_heavy` 그대로 두는 이유: ArduPilotPlugin, 부력 플러그인의
`<enable>bluerov2_heavy</enable>`, `/model/bluerov2_heavy/odometry` 토픽이 전부 이름에
묶여 있다. GZ_SIM_RESOURCE_PATH 앞쪽에 두어 상류를 **가린다**.

카메라 규약 (DEPLOY_3WEEK_PLAN.md §12.0):
  위치   base_link 기준 (0.15751, 0.00529, 0.06784)  = brov_ros2 `base_to_camera_xyz`
         = step_3 `_CAMERA_FRAME_POS` (소수점까지 동일)
  자세   identity. gz 센서 프레임은 +X 전방 / +Y 좌 / +Z 상이고, 렌더 이미지의
         +u = -Y_sensor, +v = -Z_sensor 이므로 광학축 규약이
         brov_ros2 `base_to_camera_rpy = (-90, 0, -90) deg` 와 정확히 일치한다.
  화각   fx 465.518 (실기 calibration) 을 640 폭에서 재현하는 HFOV
         = 2 atan(320/465.518) = 68.997 deg = 1.204225 rad
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from pathlib import Path

CAM_XYZ = (0.15751251578330994, 0.0052856863476336, 0.06784216314554214)
CAM_FX = 465.5181034880913          # runtime/calibration/camera_intrinsics.yaml
IMG_W, IMG_H = 640, 480


def camera_block(rate_hz: float, near: float, far: float) -> str:
    hfov = 2.0 * math.atan(0.5 * IMG_W / CAM_FX)
    return f"""
      <!-- NBV 배포용 RGBD 카메라 (deploy/make_rov_with_camera.py 가 주입).
           HFOV {math.degrees(hfov):.3f} deg 는 실기 fx={CAM_FX:.3f} 를 폭 {IMG_W} 에서
           재현하는 값이다. gz 의 camera_info 도 같은 fx 를 낸다(cx,cy 는 정중앙이라
           실기의 (324.67, 243.11) 과 4 px 차이가 있고 왜곡은 없다 — sim2real 갭으로 기록). -->
      <sensor name="nbv_camera" type="rgbd_camera">
        <pose>{CAM_XYZ[0]:.8f} {CAM_XYZ[1]:.8f} {CAM_XYZ[2]:.8f} 0 0 0</pose>
        <camera>
          <horizontal_fov>{hfov:.6f}</horizontal_fov>
          <image><width>{IMG_W}</width><height>{IMG_H}</height><format>R8G8B8</format></image>
          <clip><near>{near}</near><far>{far}</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>{rate_hz}</update_rate>
        <topic>brov/camera</topic>
      </sensor>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path,
                    default=Path(os.path.expanduser(
                        "~/SITL_Models/Gazebo/models/bluerov2_heavy")))
    ap.add_argument("--out", type=Path, required=True,
                    help="파생 모델 디렉터리 (이름은 bluerov2_heavy 로 고정)")
    ap.add_argument("--rate-hz", type=float, default=15.0)
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=20.0)
    args = ap.parse_args()

    src = args.upstream / "model.sdf"
    text = src.read_text()
    if "nbv_camera" in text:
        raise SystemExit("상류에 이미 nbv_camera 가 있다 — 상류가 오염됐는지 확인할 것")

    # base_link 의 imu_sensor 블록 바로 뒤에 넣는다. imu 는 ArduPilotPlugin 이 이름으로
    # 참조하므로 그 위치가 base_link 안이라는 것이 보증된다.
    anchor = re.search(r"[ \t]*</sensor>\s*\n", text[text.index('<sensor name="imu_sensor"'):])
    if anchor is None:
        raise SystemExit("imu_sensor 블록의 끝을 찾지 못했다 — 상류 구조가 바뀌었다")
    cut = text.index('<sensor name="imu_sensor"') + anchor.end()
    patched = text[:cut] + camera_block(args.rate_hz, args.near, args.far) + text[cut:]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.sdf").write_text(patched)
    (out / "model.config").write_text(
        "<?xml version=\"1.0\"?>\n<model>\n"
        "  <name>bluerov2_heavy</name>\n  <version>1.0</version>\n"
        "  <sdf version=\"1.9\">model.sdf</sdf>\n"
        "  <description>\n"
        "    BlueROV2 Heavy with an RGBD camera injected for the step_3 NBV deployment.\n"
        f"    Derived at runtime from {src} by deploy/make_rov_with_camera.py.\n"
        "    Do not commit: regenerate so it never drifts from upstream.\n"
        "  </description>\n</model>\n")

    meshes = out / "meshes"
    if meshes.is_symlink() or meshes.exists():
        if meshes.is_symlink():
            meshes.unlink()
        else:
            shutil.rmtree(meshes)
    meshes.symlink_to(args.upstream / "meshes")

    hfov = 2.0 * math.atan(0.5 * IMG_W / CAM_FX)
    print(f"[rov-cam] 상류 {src}")
    print(f"[rov-cam] 파생 {out/'model.sdf'}  (+{len(patched)-len(text)} bytes)")
    print(f"[rov-cam] HFOV {math.degrees(hfov):.3f} deg -> fx {0.5*IMG_W/math.tan(hfov/2):.3f} "
          f"(실기 {CAM_FX:.3f})")
    print(f"[rov-cam] topic brov/camera{{,/depth_image,/camera_info}}  @{args.rate_hz} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

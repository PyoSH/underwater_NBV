"""
setup_scene.py
--------------
OceanSim 수중 환경 최소 구성 (bottom-up 첫 단계).

구성:
  - 해저 평면 (Cube)
  - Rock (oceansim_asset/collected_rock/rock.usd)
  - UsdLux.SphereLight (인공 수중 조명)
  - UW_Camera (OceanSim, GUI viewport 포함)

좌표계: Z-up (Isaac Sim 기본)
  - 수면:   z =  0.0 m
  - 카메라: z = -0.5 m  (수면 아래 0.5 m, 정수직 하방(-Z) 주시)
  - 조명:   z = -0.45 m (카메라 약간 위, x 방향 offset → baseline 0.15 m)
  - Rock:   z = -2.5 m  (해저 근처)
  - 해저:   z = -3.25 m (평면)

경로 (Docker 컨테이너 기준):
  OceanSim:  /isaac-sim/extsUser/OceanSim
  작업 코드: /workspace/OceanNBV_test

실행 (컨테이너 내부):
  /isaac-sim/python.sh /workspace/OceanNBV_test/scene/setup_scene.py
"""

import json
import os
import sys
import numpy as np

# ── 경로 (Docker 컨테이너 내부 기준) ─────────────────────────────────────────
OCEANSIM_DIR = "/isaac-sim/extsUser/OceanSim"
ASSET_DIR    = os.path.join(OCEANSIM_DIR, "oceansim_asset")
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# asset_path.json 작성 — isaacsim.oceansim import 전에 필요
_asset_json = os.path.join(OCEANSIM_DIR, "isaacsim", "oceansim", "utils", "asset_path.json")
with open(_asset_json, "w") as _f:
    json.dump({"asset_path": ASSET_DIR}, _f)

# ── Isaac Sim 시작 (SimulationApp은 가장 먼저 생성) ───────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,              # GUI 모드
    "renderer": "RayTracedLighting",
    "width":  1280,
    "height": 720,
})

# ── SimulationApp 이후 import ─────────────────────────────────────────────────
import omni.usd
import omni.timeline
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf

from isaacsim.core.utils.stage     import create_new_stage, add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view


# ── OceanSim namespace 등록 ───────────────────────────────────────────────────
# isaacsim은 Isaac Sim이 관리하는 namespace package.
# sys.path만으로는 부족하고, 로드된 isaacsim.__path__를 직접 확장해야 함.
import isaacsim as _isaacsim_pkg
_oceansim_isaacsim = os.path.join(OCEANSIM_DIR, "isaacsim")
if _oceansim_isaacsim not in _isaacsim_pkg.__path__:
    _isaacsim_pkg.__path__.append(_oceansim_isaacsim)

from isaacsim.oceansim.sensors.UW_Camera_light import UW_Camera
from omni.isaac.sensor import Camera

# ─────────────────────────────────────────────────────────────────────────────
# 씬 구성
# ─────────────────────────────────────────────────────────────────────────────

def build_seafloor(stage) -> None:
    path = "/World/Seafloor"
    prim = stage.DefinePrim(path, "Cube")
    UsdGeom.Cube(prim).GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -3.25))
    xf.AddScaleOp().Set(Gf.Vec3d(10.0, 10.0, 0.25))
    UsdPhysics.CollisionAPI.Apply(prim)
    print("[Scene] Seafloor @ z=-3.25m")


def build_rock(stage) -> None:
    # rock.usd 내부에 이미 xformOp이 정의되어 있으므로
    # 부모 Xform 프림에 transform을 적용하고 그 아래에 USD 참조를 로드
    rock_xform_path = "/World/Rock"
    rock_mesh_path  = "/World/Rock/mesh"
    rock_usd        = os.path.join(ASSET_DIR, "collected_rock/rock.usd")

    xform_prim = UsdGeom.Xform.Define(stage, rock_xform_path)
    xf = UsdGeom.Xformable(xform_prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -3.0))
    q = euler_angles_to_quat(np.array([0.0, 0.0, 45.0]), degrees=True)
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    add_reference_to_stage(usd_path=rock_usd, prim_path=rock_mesh_path)
    print(f"[Scene] Rock @ (0, 0, -2.5)m")


# def build_light(stage) -> None:
#     # 주 조명: 수중 인공 조명 (카메라 옆에 위치)
#     light_path = "/World/UnderwaterLight"
#     light = UsdLux.SphereLight.Define(stage, light_path)
#     light.GetIntensityAttr().Set(10000000.0)
#     light.GetRadiusAttr().Set(0.05)
#     xf = UsdGeom.Xformable(light)
#     xf.AddTranslateOp().Set(Gf.Vec3d(0.00, 0.0, -0.15))
#     print("[Scene] UnderwaterLight @ (0.00, 0, -0.15)m  baseline=0.15m")

#     # 환경광: 씬 전체 확인용 약한 ambient
#     # dome_path = "/World/AmbientDome"
#     # dome = UsdLux.DomeLight.Define(stage, dome_path)
#     # dome.GetIntensityAttr().Set(100.0)
#     # print("[Scene] AmbientDome added (intensity=200)")
def build_light(stage) -> None:
    # 1. 조명 경로 및 기본 SphereLight 생성
    light_path = "/World/UnderwaterLight"
    light = UsdLux.SphereLight.Define(stage, light_path)
    
    # 기본 물리적 속성 설정
    light.GetIntensityAttr().Set(15000000.0) # 콘 조명은 에너지가 집중되므로 필요시 강도 조절
    light.GetRadiusAttr().Set(0.05)           # 광원 자체의 크기
    light.GetColorAttr().Set(Gf.Vec3f(0.8, 0.9, 1.0)) # 약간의 수중 푸른빛 추가 (선택사항)

    # 2. ShapingAPI 적용 (중요: 여기서 Cone 형태를 만듭니다)
    # ShapingAPI를 적용하면 일반 점조명이 스포트라이트(Cone)로 변합니다.
    shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
    
    # 원뿔의 각도 (Degree). 값이 커질수록 빛이 퍼지는 범위가 넓어집니다.
    shaping.GetShapingConeAngleAttr().Set(45.0) 
    
    # 원뿔 경계면의 부드러움 (0.0 ~ 1.0). 0에 가까울수록 경계가 날카롭습니다.
    shaping.GetShapingConeSoftnessAttr().Set(0.1) 
    
    # 3. 위치 및 방향 설정
    xf = UsdGeom.Xformable(light)
    
    # 카메라 위치 근처로 이동 (카메라가 z=-0.5에서 하방 -Z를 주시하므로 조명도 위치 조정)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.15))
    
    # 기본적으로 SphereLight는 모든 방향으로 쏘지만, 
    # ShapingAPI가 적용되면 기본 축(보통 -Z 또는 Isaac Sim 기준 전방)을 기준으로 제한됩니다.
    # 만약 빛의 방향을 특정 객체(Rock) 쪽으로 돌려야 한다면 아래와 같이 회전을 추가하세요.
    # 예: 하방(-Z)을 향하도록 설정 (이미 기본값이 -Z일 수 있으나 명시적 제어 필요 시)
    # q = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True)
    # xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    print(f"[Scene] Underwater Cone Light (Spotlight) @ z=-0.15m, Angle=45deg")


def build_camera_uw() -> UW_Camera:
    cam = UW_Camera(
        prim_path="/World/UW_Camera",
        name="UW_Camera",
        resolution=(640, 480),
        position=np.array([-1.0, -0.16, -2.85]),
        light_prim_path="/World/UnderwaterLight"
    )
    cam.set_focal_length(2.1)
    cam.set_clipping_range(0.05, 50.0)
    print("[Scene] UW_Camera @ (0, 0, -0.5)m  주시:-Z(하방)")
    return cam

# def build_camera() -> Camera:
#     cam = Camera(
#         prim_path="/World/StandardCamera",
#         name="StandardCamera",
#         position=np.array([-1.0, -0.16, -2.85]),
#         resolution=(320, 240),
#     )
#     # 필요한 경우 focal_length 등을 설정
#     cam.set_focal_length(2.1)
#     cam.set_clipping_range(0.05, 50.0)
#     print("[Scene] Standard Camera @ (0, 0, -0.5)m  주시:-Z(하방)")
#     return cam

# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OceanSim 수중 씬 초기화 (GUI 모드)")
    print("=" * 60)

    create_new_stage()
    stage = omni.usd.get_context().get_stage()

    # Physics scene (수중 = 무중력 근사)
    phys_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    phys_scene.GetGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    phys_scene.GetGravityMagnitudeAttr().Set(0.0)

    build_seafloor(stage)
    build_rock(stage)
    build_light(stage)
    # cam = build_camera()
    cam_uw = build_camera_uw()

    # Isaac Sim 기본 뷰포트 시점: 씬 전체가 보이는 위치로 설정
    set_camera_view(
        eye=np.array([3.0, -3.0, 1.5]),
        target=np.array([0.0, 0.0, -2.0]),
    )

    # warm-up
    for _ in range(3):
        simulation_app.update()

    # UW_Camera 초기화
    # UW_param [0:3]=backscatter_value, [3:6]=backscatter_coeff, [6:9]=atten_coeff
    # (구현 기준 — docstring과 [3:6]/[6:9] 순서 반전 주의)
    UW_param = np.array([
        0.0,  0.31, 0.24,   # backscatter_value (청록색 배경)
        0.05, 0.05, 0.2,    # [3:6] → backscatter_coeff
        0.05, 0.05, 0.05,   # [6:9] → atten_coeff
    ])
    cam_uw.initialize(
        UW_param=UW_param,
        viewport=True,          # GUI: UW Camera 전용 뷰포트 창 생성
        # writing_dir=OUTPUT_DIR, # 렌더 결과 저장
    )
    # cam.initialize()

    # ── 실행 루프 ─────────────────────────────────────────────────────────────
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    while simulation_app.is_running():
        simulation_app.update()
        try:
            cam_uw.render()
            # rgba = cam.get_rgba()
        except RuntimeError as e:
            # timeline pause 등으로 depth annotator CUDA 버퍼가 무효화될 수 있음
            print(f"[render skip] {e}")

    cam_uw.close()
    # cam.close()

    print("[완료]")


if __name__ == "__main__":
    main()
    simulation_app.close()

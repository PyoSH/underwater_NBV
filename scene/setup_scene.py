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
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Usd  # Usd 추가

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

    marker_path = f"{light_path}/VisualMarker"
    marker = UsdGeom.Sphere.Define(stage,marker_path)
    marker.GetRadiusAttr().Set(0.1)
    marker.GetDisplayColorAttr().Set([Gf.Vec3f(1.0,1.0,0.0)])
    marker.GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    # 기본적으로 SphereLight는 모든 방향으로 쏘지만, 
    # ShapingAPI가 적용되면 기본 축(보통 -Z 또는 Isaac Sim 기준 전방)을 기준으로 제한됩니다.
    # 만약 빛의 방향을 특정 객체(Rock) 쪽으로 돌려야 한다면 아래와 같이 회전을 추가하세요.
    # 예: 하방(-Z)을 향하도록 설정 (이미 기본값이 -Z일 수 있으나 명시적 제어 필요 시)
    # q = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True)
    # xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    print(f"[Scene] Underwater Cone Light (Spotlight) @ z=-0.15m, Angle=45deg")


def build_camera_uw(parent_path) -> UW_Camera:
    cam_path = f"{parent_path}/UW_Camera"

    cam = UW_Camera(
        prim_path=cam_path,
        name="UW_Camera",
        resolution=(640, 480),
        position=np.array([-1.0, -0.16, -2.85]),
        light_prim_path="/World/UnderwaterLight"
    )
    cam.set_focal_length(2.1)
    cam.set_clipping_range(0.05, 50.0)

    stage = omni.usd.get_context().get_stage()
    marker_path = f"{cam_path}/VisualMarker"
    marker = UsdGeom.Cube.Define(stage, marker_path)
    
    # 크기 조절 (0.05m 크기의 큐브)
    marker_xf = UsdGeom.Xformable(marker)
    marker_xf.AddScaleOp().Set(Gf.Vec3d(0.1, 0.2, 0.1))
    marker.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.5, 1.0)]) # 파란색
    
    return cam

def build_camera_with_body(stage, initial_pos) -> tuple:
    """
    직육면체 바디를 생성하고, 카메라를 그 자식으로 등록합니다.
    """
    # 1. 부모가 될 Xform (빈 상자) - 움직임의 주체
    rig_path = "/World/MovingCameraRig"
    rig_prim = UsdGeom.Xform.Define(stage, rig_path)
    xf_rig = UsdGeom.Xformable(rig_prim)
    xf_rig.AddTranslateOp().Set(Gf.Vec3d(*initial_pos.tolist()))

    # 2. 시각적 형태 (직육면체 모델) - rig 하위에 생성
    body_path = f"{rig_path}/VisualBody"
    cube = UsdGeom.Cube.Define(stage, body_path)
    cube.GetSizeAttr().Set(1.0)
    
    # 큐브를 스케일링하여 직육면체(x방향으로 긴)로 만듦
    xf_cube = UsdGeom.Xformable(cube)
    # 가로 0.2m, 세로 0.1m, 높이 0.1m의 직육면체
    xf_cube.AddScaleOp().Set(Gf.Vec3d(0.2, 0.1, 0.1)) 
    
    # 색상 설정 (시안색, 뷰포트에서 보임)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 1.0)])
    # *중요*: Purpose guide를 설정하지 않아 렌더링/뷰포트에서 보이게 함

    # 3. 카메라 프림 생성 (rig 하위에 'Camera'라는 이름으로 생성)
    cam_child_path = f"{rig_path}/Camera"
    
    # 4. OceanSim UW_Camera 인스턴스화
    # 주의: 부모(rig) 하위 경로를 전달하고, position은 Gf.Vec3d(0)을 주어
    # 부모 좌표계 기준 로컬 오프셋이 없도록 설정 (완벽 부착)
    cam = UW_Camera(
        prim_path=cam_child_path,
        name="UW_Camera",
        resolution=(640, 480),
        # 부모 rig 좌표계를 그대로 사용 (로컬 0,0,0)
        position=np.array([initial_pos[0]+0.1, initial_pos[1], initial_pos[2]]), 
        light_prim_path="/World/UnderwaterLight" # 고정된 조명 참조
    )
    cam.set_focal_length(2.1)
    cam.set_clipping_range(0.05, 50.0)

    print(f"[Scene] Camera coupled with Rectangular parallelepiped at {rig_path}")
    return rig_path, cam


def main():
    print("=" * 60)
    print("OceanSim 수중 씬 초기화 및 병진 운동 시작")
    print("=" * 60)

    create_new_stage()
    stage = omni.usd.get_context().get_stage()

    # Physics scene 설정
    phys_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    phys_scene.GetGravityMagnitudeAttr().Set(0.0)

    build_seafloor(stage)
    build_rock(stage)
    build_light(stage)
    # cam_uw = build_camera_uw("/World")
    initial_cam_pos = np.array([-3.0, -0.0, -2.85])
    cam_rig_path, cam_uw = build_camera_with_body(stage, initial_cam_pos)

    set_camera_view(
        eye=np.array([3.0, -3.0, 1.5]),
        target=np.array([0.0, 0.0, -2.0]),
    )

    # warm-up
    for _ in range(3):
        simulation_app.update()

    # UW_Camera 초기화
    UW_param = np.array([
        0.06,  0.61, 0.34,   
        0.05, 0.05, 0.2,    
        0.05, 0.05, 0.05,   
    ])
    cam_uw.initialize(UW_param=UW_param, viewport=True)
    timeline = omni.timeline.get_timeline_interface()

    # ── 병진 운동을 위한 설정 ──────────────────────────────────────────
    rig_path = cam_rig_path
    rig_prim = stage.GetPrimAtPath(rig_path)
    xf_rig = UsdGeom.Xformable(rig_prim)

    # 2. 조명 프림 (조명은 Rig 외부에 있으므로 별도 제어 혹은 Rig 자식으로 이동 필요)
    light_prim = stage.GetPrimAtPath("/World/UnderwaterLight")
    xf_light = UsdGeom.Xformable(light_prim)

    # 속도 및 초기 위치
    velocity = np.array([0.05, 0.0, 0.0]) 
    current_rig_pos = initial_cam_pos.copy() # Rig의 초기 위치
    current_light_pos = np.array([0.0, 0.0, -0.15])
    dt = 1.0 / 60.0

    timeline.play()

    while simulation_app.is_running():
        # 루프 내부에서 매번 프림 유효성 검사 (안전장치)
        if not rig_prim.IsValid():
            break

        # 1. 위치 계산
        current_rig_pos += velocity * dt
        current_light_pos += 0.5*velocity * dt

        # 2. USD 속성 업데이트
        # Rig만 움직이면 그 안의 카메라와 직육면체는 알아서 따라감
        rig_ops = xf_rig.GetOrderedXformOps()
        if rig_ops:
            rig_ops[0].Set(Gf.Vec3d(*current_rig_pos.tolist()))
        
        light_ops = xf_light.GetOrderedXformOps()
        if light_ops:
            light_ops[0].Set(Gf.Vec3d(*current_light_pos.tolist()))

        simulation_app.update()
        
        try:
            cam_uw.render()
        except RuntimeError as e:
            print(f"[render skip] {e}")

    cam_uw.close()
    print("[완료]")


if __name__ == "__main__":
    main()
    simulation_app.close()

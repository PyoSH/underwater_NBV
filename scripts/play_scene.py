import argparse
import os
import json
import numpy as np
from isaaclab.app import AppLauncher

# 1. 앱 실행기 설정
parser = argparse.ArgumentParser(description="OceanSim Scene Viewer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- OceanSim 환경 강제 주입 (Isaac Lab 시작 전 필수) ---
OCEANSIM_DIR = "/isaac-sim/extsUser/OceanSim"
ASSET_DIR = os.path.join(OCEANSIM_DIR, "oceansim_asset")
_asset_json = os.path.join(OCEANSIM_DIR, "isaacsim", "oceansim", "utils", "asset_path.json")
with open(_asset_json, "w") as _f:
    json.dump({"asset_path": ASSET_DIR}, _f)

import isaacsim as _isaacsim_pkg
_oceansim_isaacsim = os.path.join(OCEANSIM_DIR, "isaacsim")
if _oceansim_isaacsim not in _isaacsim_pkg.__path__:
    _isaacsim_pkg.__path__.append(_oceansim_isaacsim)

from isaacsim.oceansim.sensors.UW_Camera_light import UW_Camera
from pxr import UsdLux
import omni.usd

# 2. 필수 라이브러리 임포트
from isaaclab.scene import InteractiveScene
from OceanRL_test.scripts.ocean_scene_cfg import OceanSimSceneCfg

def apply_light_shaping(stage, light_path):
    """조명에 Spotlight (ShapingAPI) 적용"""
    light_prim = stage.GetPrimAtPath(light_path)
    if light_prim.IsValid():
        shaping = UsdLux.ShapingAPI.Apply(light_prim)
        shaping.GetShapingConeAngleAttr().Set(45.0)
        shaping.GetShapingConeSoftnessAttr().Set(0.1)

def main():
    # 시뮬레이션 물리 환경 설정 (무중력)
    import isaaclab.sim as sim_utils
    sim_cfg = sim_utils.SimulationCfg(gravity=(0.0, 0.0, 0.0))
    
    # 씬 구성
    scene_cfg = OceanSimSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    
    stage = omni.usd.get_context().get_stage()

    # 조명 원뿔 형태(Shaping) 사후 적용
    apply_light_shaping(stage, "/World/UnderwaterLight")

    # 3. UW_Camera 초기화 (Isaac Lab Scene 생성 직후 배치)
    cam_uw = UW_Camera(
        prim_path="/World/UW_Camera",
        name="UW_Camera",
        resolution=(640, 480),
        position=np.array([-1.0, -0.16, -2.85]),
        light_prim_path="/World/UnderwaterLight"
    )
    cam_uw.set_focal_length(2.1)
    cam_uw.set_clipping_range(0.05, 50.0)

    # 파라미터 셋업 [0:3]=backscatter_val, [3:6]=backscatter_coeff, [6:9]=atten_coeff
    UW_param = np.array([0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05])
    
    for _ in range(3):
        simulation_app.update()

    cam_uw.initialize(UW_param=UW_param, viewport=True)

    print("[INFO]: Scene loaded. Press Ctrl+C to stop.")
    
    # 시뮬레이션 루프
    while simulation_app.is_running():
        # 물리 단계 업데이트
        scene.update(dt=0.01)
        
        # 수중 광학 카메라 렌더링 업데이트
        try:
            cam_uw.render()
        except RuntimeError as e:
            print(f"[render skip] {e}")

        # 메인 시뮬레이션 업데이트
        simulation_app.update()

    cam_uw.close()

if __name__ == "__main__":
    main()
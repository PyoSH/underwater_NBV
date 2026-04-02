"""
sceneCfg.py
-----------
Isaac Lab InteractiveSceneCfg.
{ENV_REGEX_NS} 패턴으로 num_envs 개의 환경이 자동 복제됨.

프림 트리 (env 당):
    {ENV_REGEX_NS}/Seafloor         - 정적 해저면 (CollisionAPI)
    {ENV_REGEX_NS}/Rock             - 대상 물체 (USD 참조)
    {ENV_REGEX_NS}/CameraRig        - 카메라 강체 (RigidBodyAPI + CollisionAPI)
      └── Camera                   - Pinhole 카메라 (CameraCfg 자동 생성)
    {ENV_REGEX_NS}/LightRig         - 조명 강체  (RigidBodyAPI + CollisionAPI)

_setup_scene() 에서 추가되는 자식 프림:
    LightRig/SphereLight            - 스포트라이트
"""

import math
import os

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

OCEANSIM_DIR = "/isaac-sim/extsUser/OceanSim"
ASSET_DIR    = os.path.join(OCEANSIM_DIR, "oceansim_asset")
ROCK_USD     = os.path.join(ASSET_DIR, "collected_rock/rock.usd")

_oceansim_isaacsim = os.path.join(OCEANSIM_DIR, "isaacsim")
if _oceansim_isaacsim not in _isaacsim_pkg.__path__:
    _isaacsim_pkg.__path__.append(_oceansim_isaacsim)

# from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
from isaacsim.oceansim.sensors.UW_Camera_parallel import UWCamera

# rock 45° Z 회전 쿼터니언 [w, x, y, z]
_ROT_45Z = (math.cos(math.radians(22.5)), 0.0, 0.0, math.sin(math.radians(22.5)))

floorDepth = -3.25
wallHeight = 5.0
wallWidth = 0.01
wallLength = 10.0

@configclass
class OceanSceneCfg(InteractiveSceneCfg):
    """수중 탐색 씬."""

    # ── 해저면 (정적 충돌체) ─────────────────────────────────────────────────
    seafloor: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Seafloor",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, floorDepth)),
    )

    # 북쪽 벽 (Y+)
    wall_north: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall1",
        spawn=sim_utils.CuboidCfg(
            size=(wallLength, wallWidth, wallHeight),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)), # 어두운 색이 빛 반사 억제에 유리
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, wallLength/2, floorDepth + wallHeight/2))
    )
    # 남쪽 벽 (Y-)
    wall_south: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall2",
        spawn=sim_utils.CuboidCfg(
            size=(wallLength, wallWidth, wallHeight),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -wallLength/2, floorDepth + wallHeight/2))
    )
    # 동쪽 벽 (X+)
    wall_east: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall3",
        spawn=sim_utils.CuboidCfg(
            size=(wallWidth, wallLength, wallHeight),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(wallLength/2, 0.0, floorDepth + wallHeight/2))
    )
    # 서쪽 벽 (X-)
    wall_west: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall4",
        spawn=sim_utils.CuboidCfg(
            size=(wallWidth, wallLength, wallHeight),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-wallLength/2, 0.0, floorDepth + wallHeight/2))
    )

    # ── 대상 물체 (시각 전용) ───────────────────────────────────────────────
    rock: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rock",
        spawn=sim_utils.UsdFileCfg(usd_path=ROCK_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0), rot=_ROT_45Z),
    )

    # ── 카메라 리그 (동적 강체, 하늘색) ────────────────────────────────────
    camera_rig: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CameraRig",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    )

    # sonar_rig: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/SonarRig",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.10, 0.05, 0.05),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=True,
    #             linear_damping=0.5,
    #             angular_damping=0.5,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    # )

    # ── 조명 리그 (동적 강체, 노란색) ──────────────────────────────────────
    light_rig: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LightRig",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, -0.45)),
    )

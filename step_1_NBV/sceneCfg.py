import math
import os, sys

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

OCEANSIM_DIR = "/isaac-sim/extsUser/OceanSim"
ASSET_DIR    = os.path.join(OCEANSIM_DIR, "oceansim_asset")
ROCK_USD     = os.path.join(ASSET_DIR, "collected_rock/rock.usd")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sensors.UWCamera.UW_Camera_cfg import UWCameraCfg
from sensors.ImagingSonar.ImagingSonarCfg import ImagingSonarCfg

# rock 45° Z 회전 쿼터니언 [w, x, y, z]
_ROT_45Z = (math.cos(math.radians(22.5)), 0.0, 0.0, math.sin(math.radians(22.5)))

floorDepth = -3.25
wallHeight = 5.0
wallWidth = 0.01
wallLength = 10.0

wall_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.01, 0.01, 0.01),  # 난반사(Diffuse)를 0으로 설정 (완전 검정)
    metallic=0.0,                 # 금속성 제거 (금속성 반사 차단)
    roughness=1.0,                # 거칠기를 최대화하여 정반사(Specular) 억제
)

# @configclass
# class OceanSceneCfg(InteractiveSceneCfg):
#     """수중 탐색 씬."""

#     # ── 해저면 (정적 충돌체) ─────────────────────────────────────────────────
#     seafloor: AssetBaseCfg = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/Seafloor",
#         spawn=sim_utils.CuboidCfg(
#             size=(10.0, 10.0, 0.25),
#             collision_props=sim_utils.CollisionPropertiesCfg(),
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
#         ),
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, floorDepth)),
#     )

#     # 북쪽 벽 (Y+)
#     wall_north: AssetBaseCfg = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/Wall1",
#         spawn=sim_utils.CuboidCfg(
#             size=(wallLength, wallWidth, wallHeight),
#             visual_material=wall_material,
#         ),
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, wallLength/2, floorDepth + wallHeight/2))
#     )


wall_mat_north = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), roughness=1.0, metallic=0.0)
wall_mat_south = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), roughness=1.0, metallic=0.0)
wall_mat_east  = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), roughness=1.0, metallic=0.0)
wall_mat_west  = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0, metallic=0.0)

@configclass
class OceanSceneCfg(InteractiveSceneCfg):
    """수중 탐색 씬."""
 
    # ── 글로벌 조명 (태양광 역할) ─────────────────────────────────────────────
    # prim_path 에 ENV_REGEX_NS 를 쓰지 않아 모든 env 에 공유되는 단일 광원.
    # DistantLight: 무한 거리의 평행광 → 씬 전체를 균일하게 비춰
    # 벽 색상이 카메라 이미지에 명확히 드러나게 함.
    # distant_light: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="/World/DistantLight",
    #     spawn=sim_utils.DistantLightCfg(
    #         intensity=3000.0,
    #         color=(1.0, 1.0, 1.0),
    #         angle=0.53,             # 태양 반각 [도]
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 10.0),
    #         rot=(0.9239, 0.0, 0.3827, 0.0),  # Y축 기준 45° 기울여 아래를 향함
    #     ),
    # )
 
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
 
    # ── 벽 (방향별 색상) ─────────────────────────────────────────────────────
    # 북쪽 벽 (Y+): 빨강
    wall_north: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall1",
        spawn=sim_utils.CuboidCfg(
            size=(wallLength, wallWidth, wallHeight),
            visual_material=wall_mat_north,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, wallLength/2, floorDepth + wallHeight/2))
    )
    # 남쪽 벽 (Y-): 초록
    wall_south: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall2",
        spawn=sim_utils.CuboidCfg(
            size=(wallLength, wallWidth, wallHeight),
            visual_material=wall_mat_south,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -wallLength/2, floorDepth + wallHeight/2))
    )
    # 동쪽 벽 (X+): 파랑
    wall_east: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall3",
        spawn=sim_utils.CuboidCfg(
            size=(wallWidth, wallLength, wallHeight),
            visual_material=wall_mat_east,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(wallLength/2, 0.0, floorDepth + wallHeight/2))
    )
    # 서쪽 벽 (X-): 노랑
    wall_west: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall4",
        spawn=sim_utils.CuboidCfg(
            size=(wallWidth, wallLength, wallHeight),
            visual_material=wall_mat_west,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-wallLength/2, 0.0, floorDepth + wallHeight/2))
    )
 
    # ── 대상 물체 (시각 전용) ───────────────────────────────────────────────
    rock: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=ROCK_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0), rot=_ROT_45Z),
    )
 
    # ── 센서 리그 (동적 강체, 하늘색) ────────────────────────────────────────
    sensor_rig: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorRig",
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
 
    camera: UWCameraCfg = UWCameraCfg(
        prim_path="{ENV_REGEX_NS}/SensorRig/Camera",
        update_period=0,
        height=480,
        width=640,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            clipping_range=(0.1, 20.0)
        ),
        offset=UWCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            # rot=(0.5, -0.5, -0.5, 0.5),   # [w, x, y, z]
            rot=(1.0, 0.0, 0.0, 0.0),   # [w, x, y, z]
            convention="world",
        ),
        backscatter_value  =(0.05, 0.31, 0.24),
        atten_coeff        =(0.05, 0.05, 0.20),
        backscatter_coeff  =(0.05, 0.05, 0.05),
    )
    # camera: CameraCfg = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/SensorRig/Camera",
    #     update_period=0,
    #     height=480,
    #     width=640,
    #     data_types=["rgba", "distance_to_camera"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         clipping_range=(0.1, 20.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.0, 0.0),
    #         rot=(1.0, 0.0, 0.0, 0.0),   # [w, x, y, z]
    #         convention="world",
    #     ),
    # )
 
    '''
    SensorRig에 부착된 위치 수정할 것.
    '''
    sphere_light_0: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SensorRig/SphereLight_0",
        spawn=sim_utils.SphereLightCfg(
            intensity=10000000.0,
            radius=0.05,
            color=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.06, 0.5, 0.0),
            rot=(0.7071, 0.0, -0.7071, 0.0),   # [w, x, y, z]
        ),
    )
 
    sphere_light_1: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SensorRig/SphereLight_1",
        spawn=sim_utils.SphereLightCfg(
            intensity=10000000.0,
            radius=0.05,
            color=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.06, -0.5, 0.0),
            rot=(0.7071, 0.0, -0.7071, 0.0),   # [w, x, y, z]
        ),
    )
 


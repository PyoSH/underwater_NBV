import math
import os, sys

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

OCEANSIM_DIR = "/isaac-sim/extsUser/OceanSim"
ASSET_DIR    = os.path.join(OCEANSIM_DIR, "oceansim_asset")
ROCK_USD     = os.path.join(ASSET_DIR, "collected_rock/rock.usd")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sensors.UWCamera.UW_Camera_cfg import UWCameraCfg
from sensors.ImagingSonar.ImagingSonarCfg import ImagingSonarCfg

# rock 45° Z 회전 쿼터니언 [w, x, y, z]
_ROT_45Z = (math.cos(math.radians(22.5)), 0.0, 0.0, math.sin(math.radians(22.5)))

floorDepth = -3.25
wallHeight = 5.5
wallWidth = 0.01
wallLength = 10.0

wall_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.01, 0.01, 0.01),  # 난반사(Diffuse)를 0으로 설정 (완전 검정)
    metallic=0.0,                 # 금속성 제거 (금속성 반사 차단)
    roughness=1.0,                # 거칠기를 최대화하여 정반사(Specular) 억제
)

# wall_mat_north = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), roughness=1.0, metallic=0.0)
# wall_mat_south = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), roughness=1.0, metallic=0.0)
# wall_mat_east  = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), roughness=1.0, metallic=0.0)
# wall_mat_west  = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0, metallic=0.0)

@configclass
class OceanSceneCfg(InteractiveSceneCfg):
    """수중 탐색 씬.""" 
    # ── 해저면 (정적 충돌체) ─────────────────────────────────────────────────
    seafloor: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Seafloor",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.50, 0.50)),
            # visual_material=wall_material,    # 조명 영향 큼!!
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, floorDepth)),
    )
 
    # ── 벽 (방향별 색상) ─────────────────────────────────────────────────────
    # 북쪽 벽 (Y+): 빨강
    wall_north: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall1",
        spawn=sim_utils.CuboidCfg(
            size=(wallLength, wallWidth, wallHeight),
            visual_material=wall_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, wallLength/2, floorDepth + wallHeight/2))
    )
    # 남쪽 벽 (Y-): 초록
    wall_south: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall2",
        spawn=sim_utils.CuboidCfg(
            size=(wallLength, wallWidth, wallHeight),
            visual_material=wall_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -wallLength/2, floorDepth + wallHeight/2))
    )
    # 동쪽 벽 (X+): 파랑
    wall_east: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall3",
        spawn=sim_utils.CuboidCfg(
            size=(wallWidth, wallLength, wallHeight),
            visual_material=wall_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(wallLength/2, 0.0, floorDepth + wallHeight/2))
    )
    # 서쪽 벽 (X-): 노랑
    wall_west: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall4",
        spawn=sim_utils.CuboidCfg(
            size=(wallWidth, wallLength, wallHeight),
            visual_material=wall_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-wallLength/2, 0.0, floorDepth + wallHeight/2))
    )
 
    # ── 대상 물체 (시각 전용) ───────────────────────────────────────────────
    rock: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=ROCK_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0), rot=_ROT_45Z), # 2.5
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
        height=240,
        width=320,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            clipping_range=(0.1, 20.0)
        ),
        offset=UWCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            # rot=(0.0, -0.5, 0.5, -0.5),   # [w, x, y, z]
            rot=(1.0, 0.0, 0.0, 0.0),   # [w, x, y, z]
            convention="world",
        ),
        backscatter_value  =(0.05, 0.31, 0.24),
        atten_coeff        =(0.05, 0.05, 0.20),
        backscatter_coeff  =(0.05, 0.05, 0.05),
    )

    # ── 이미징 소나 (Oculus M750d 기준, 1.2MHz 모드) ────────────────────────────
    # hori_fov=130°, vert_fov=20°, angular_res=0.6°, range_res=2.5mm, max_range=40m(씬:5m)
    # hori_res=4000, vert_fov=20 → height=int(4000*(20/130))=615, width=4000
    # range: 0.1~5.0m, range_res=0.0025 → R=1960bins, angular_res=0.6 → A=216bins
    # polar canvas: H=1960, W=int(2*1960*sin(65°))=3552
    sonar: ImagingSonarCfg = ImagingSonarCfg(
        prim_path="{ENV_REGEX_NS}/SensorRig/Sonar",
        update_period=0,
        height=615,
        width=4000,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=103.0,   # 2*24*tan(65°) ≈ 103mm → hori_fov≈130°
            clipping_range=(0.05, 6.0),
        ),
        offset=ImagingSonarCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.2),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        hori_res=4000.0,
        hori_fov=130.0,
        vert_fov=20.0,
        min_range=0.1,
        max_range=5.0,
        range_res=0.0025,
        angular_res=0.6,
        attenuation=0.1,
        binning_method="sum",
        normalizing_method="range",
        enable_viewport=False,
    )
 
    '''
    SensorRig에 부착된 위치 수정할 것.
    '''
    sphere_light_0: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SensorRig/SphereLight_0",
        spawn=sim_utils.SphereLightCfg(
            intensity=0.0,
            radius=0.05,
            color=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.055, 0.185, -0.125),
            rot=(0.7071, 0.0, -0.7071, 0.0),   # [w, x, y, z]
        ),
    )
 
    sphere_light_1: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SensorRig/SphereLight_1",
        spawn=sim_utils.SphereLightCfg(
            intensity=0.0,
            radius=0.05,
            color=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.055, -0.185, -0.125),
            rot=(0.7071, 0.0, -0.7071, 0.0),   # [w, x, y, z]
        ),
    )
 


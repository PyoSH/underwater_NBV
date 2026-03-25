from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

@configclass
class OceanSimSceneCfg(InteractiveSceneCfg):
    """OceanSim 요소를 포함한 수중 환경 설정"""

    # 1. 해저 지형 (충돌체 추가)
    seafloor = AssetBaseCfg(
        prim_path="/World/Seafloor",
        spawn=sim_utils.MeshPlaneCfg(
            size=(50, 50),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(), # 물리 충돌 속성 부여
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.25))
    )

    # 2. 바위 에셋 (Z축 45도 회전 쿼터니언 적용)
    rock = AssetBaseCfg(
        prim_path="/World/Objects/Rock",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/oceansim/oceansim_asset/collected_rock/rock.usd",
            scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0), rot=(1.0, 0.0, 0.0, 0.0))
    )

    # 3. 수중 조명 (스펙 복구 및 속성 매핑)
    # 참고: Isaac Lab의 SphereLightCfg에 ShapingAPI가 기본 노출되지 않는 경우,
    underwater_light = AssetBaseCfg(
        prim_path="/World/UnderwaterLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=15000000.0, 
            radius=0.05, 
            color=(0.8, 0.9, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.25))
    )
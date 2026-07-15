"""
BROV2 궤적 추종 씬 구성
========================
InteractiveSceneCfg 서브클래스.
IsaacLab DirectRLEnv 가 super().__init__() 시점에 자동으로 스폰한다.

포함 에셋
---------
robot     : BROV2 Heavy Articulation (brov_rigid.py, BROV_0706.usd)
seafloor  : 정적 해저면 (충돌체)
dome_light: 전역 반구 조명
"""

import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robots.assets.brov_rigid import BROV_RIGID_CFG

_SEAFLOOR_Z = -15.0


@configclass
class BROVSceneCfg(InteractiveSceneCfg):
    """BROV2 궤적 추종 수중 씬."""

    robot: ArticulationCfg = BROV_RIGID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    seafloor: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Seafloor",
        spawn=sim_utils.CuboidCfg(
            size=(50.0, 50.0, 0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.12, 0.18),
                roughness=0.95,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _SEAFLOOR_Z)),
    )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1200.0,
            color=(0.18, 0.35, 0.55),
        ),
    )

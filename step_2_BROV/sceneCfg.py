"""
BROV2 궤적 추종 씬 구성
========================
InteractiveSceneCfg 서브클래스.
IsaacLab DirectRLEnv 가 super().__init__() 시점에 자동으로 스폰한다.

포함 에셋
---------
robot     : BROV2 Heavy Articulation (8-thruster, brov_joint.py)
seafloor  : 정적 해저면 (충돌체)
dome_light: 전역 반구 조명
"""

import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# robots 패키지 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robots.assets.brov_joint import BROV_CFG

# ── 씬 상수 ────────────────────────────────────────────────────────────────────
_SEAFLOOR_Z = -15.0   # 해저 깊이 [m] (world Z 기준)


@configclass
class BROVSceneCfg(InteractiveSceneCfg):
    """BROV2 궤적 추종 수중 씬."""

    # ── BROV2 Heavy 로봇 (ArticulationCfg) ─────────────────────────────────────
    # brov_joint.py 의 BROV_CFG 를 그대로 사용.
    # init_state.pos 의 z = 5 m (수심 5 m) 로 스폰 → _reset_idx 에서 재설정.
    robot: ArticulationCfg = BROV_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ── 해저면 (정적 강체, 충돌 전용) ──────────────────────────────────────────
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

    # ── 전역 반구 조명 ─────────────────────────────────────────────────────────
    # prim_path 에 {ENV_REGEX_NS} 없음 → 씬 전체에 하나만 생성
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1200.0,
            color=(0.18, 0.35, 0.55),   # 수중 청색 분위기
        ),
    )

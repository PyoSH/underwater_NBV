import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

USD_PATH = os.path.join(
    os.path.dirname(__file__), "../data/BROV2/brov2_custom_physics.usda"
)

BROV_RIGID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)
"""BROV2 ArticulationCfg — brov2_custom_physics.usda 기반 (brov2_spec.md 정본).

defaultPrim "/BROV2_Heavy" 자체가 hull 전체 mesh 통합 + RigidBody/MassAPI, 관절 없는 단일
body(articulation root는 spawn 시 articulation_props로 이 prim에 런타임 적용됨).
스러스터 8개 + DVL/카메라/조명 프레임은 mesh 없는 참조용 Xform(위치 + userProperties:axis 등)
으로만 존재.
  → 실제 추력/유체역학은 Python에서 계산 후 이 prim에 set_external_force_and_torque로 주입.
linear_damping=0, angular_damping=0: Fossen 항력 모델과 중복 방지.
"""

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

import os
USD_PATH = os.path.join(os.path.dirname(__file__), "../data/BROV2/BlueROV2_buoyancy.usd")

BROV_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),

        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5),
        joint_pos={
            "t1_ccw":0.0,
            "t2_ccw":0.0,
            "t3_cw":0.0,
            "t4_cw":0.0,
            "t5_ccw":0.0,
            "t6_cw":0.0,
            "t7_cw":0.0,
            "t8_ccw":0.0,
        },
    ),
    actuators={
        "t1_ccw": ImplicitActuatorCfg(
            joint_names_expr=["t1_ccw"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
        "t2_ccw": ImplicitActuatorCfg(
            joint_names_expr=["t2_ccw"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
        "t3_cw": ImplicitActuatorCfg(
            joint_names_expr=["t3_cw"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
        "t4_cw": ImplicitActuatorCfg(
            joint_names_expr=["t4_cw"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
        "t5_ccw": ImplicitActuatorCfg(
            joint_names_expr=["t5_ccw"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
        "t6_cw": ImplicitActuatorCfg(
            joint_names_expr=["t6_cw"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
        "t7_cw": ImplicitActuatorCfg(
            joint_names_expr=["t7_cw"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
        "t8_ccw": ImplicitActuatorCfg(
            joint_names_expr=["t8_ccw"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
    },
)
"""Configuration for the BROV2."""

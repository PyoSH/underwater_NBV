import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

import os
USD_PATH = os.path.join(os.path.dirname(__file__), "../data/I90/I90_arm.usd")

I90_CFG = ArticulationCfg(
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
            "b_l1":0.0,
            "l1_l2":0.0,
            "l2_l3":0.0,
            "l3_l4":0.0,
            "l4_ee":0.0,
            "g1":0.0,
            "g2":0.0,
        },
    ),
    actuators={
        "b_l1": ImplicitActuatorCfg(
            joint_names_expr=["b_l1"],
            stiffness=10000.0,
            damping=100.0,
            effort_limit_sim=4000.0,
        ),
        "l1_l2": ImplicitActuatorCfg(
            joint_names_expr=["l1_l2"],
            stiffness=10000.0,
            damping=100.0,
            effort_limit_sim=4000.0,
        ),
        "l2_l3": ImplicitActuatorCfg(
            joint_names_expr=["l2_l3"],
            stiffness=10000.0,
            damping=100.0,
            effort_limit_sim=4000.0,
        ),
        "l3_l4": ImplicitActuatorCfg(
            joint_names_expr=["l3_l4"],
            stiffness=10000.0,
            damping=100.0,
            effort_limit_sim=4000.0,
        ),
        "l4_ee": ImplicitActuatorCfg(
            joint_names_expr=["l4_ee"],
            stiffness=10000.0,
            damping=100.0,
            effort_limit_sim=4000.0,
        ),
        "g1": ImplicitActuatorCfg(
            joint_names_expr=["g1"],
            stiffness=10000.0,
            damping=100.0,
            effort_limit_sim=4000.0,
        ),
        "g2": ImplicitActuatorCfg(
            joint_names_expr=["g2"],
            stiffness=10000.0,
            damping=100.0,
            effort_limit_sim=4000.0,
        )
    },
)

"""Configuration for the I90+ with arm."""
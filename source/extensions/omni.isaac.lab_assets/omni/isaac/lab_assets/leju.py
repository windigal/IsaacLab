import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

Bite_s42_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/root/IsaacLab/models/biped_s42_fine/xml/biped_s42_collision/biped_s42_noworld_mass_singlelayer.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # "zhead_1_joint": 0.0,
            # "zhead_2_joint": 0.0,
            # "leg_l1_joint": 0.0,
            # "leg_l2_joint": 0.0,
            # "leg_l3_joint": -0.349066,  # -20 degrees
            # "leg_l4_joint": 0.785399,  # 45 degrees
            # "leg_l5_joint": -0.523599,  # -30 degrees
            # "leg_l6_joint": 0.0,
            # "leg_r1_joint": 0.0,
            # "leg_r2_joint": 0.0,
            # "leg_r3_joint": -0.349066,  # -20 degrees
            # "leg_r4_joint": 0.785399,  # 45 degrees
            # "leg_r5_joint": -0.523599,  # -30 degrees
            # "leg_r6_joint": 0.0,
            # "zarm_l1_joint": 0.279253,  # 16 degrees
            # "zarm_l2_joint": 0.0,
            # "zarm_l3_joint": 0.0,
            # "zarm_l4_joint": 0.0,
            # "zarm_l5_joint": 1.570797,  # 90 degrees
            # "zarm_l6_joint": 0.0,
            # "zarm_l7_joint": -1.570797,  # -90 degrees
            # "zarm_r1_joint": 0.279253,  # 16 degrees
            # "zarm_r2_joint": 0.0,
            # "zarm_r3_joint": 0.0,
            # "zarm_r4_joint": 0.0,
            # "zarm_r5_joint": -1.570797,  # -90 degrees
            # "zarm_r6_joint": 0.0,
            # "zarm_r7_joint": 1.570797,  # 90 degrees
            "zhead_1_joint": 0.0,
            "zhead_2_joint": 0.0,
            "leg_l1_joint": 0.0,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": 0.0,  # -20 degrees
            "leg_l4_joint": 0.0,  # 45 degrees
            "leg_l5_joint": 0.0,  # -30 degrees
            "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.0,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": 0.0,  # -20 degrees
            "leg_r4_joint": 0.0,  # 45 degrees
            "leg_r5_joint": 0.0,  # -30 degrees
            "leg_r6_joint": 0.0,
            "zarm_l1_joint": 0.0,  # 16 degrees
            "zarm_l2_joint": 0.0,
            "zarm_l3_joint": 0.0,
            "zarm_l4_joint": 0.0,
            "zarm_l5_joint": 0.0,  # 90 degrees
            "zarm_l6_joint": 0.0,
            "zarm_l7_joint": 0.0,  # -90 degrees
            "zarm_r1_joint": 0.0,  # 16 degrees
            "zarm_r2_joint": 0.0,
            "zarm_r3_joint": 0.0,
            "zarm_r4_joint": 0.0,
            "zarm_r5_joint": 0.0,  # -90 degrees
            "zarm_r6_joint": 0.0,
            "zarm_r7_joint": 0.0,  # 90 degrees
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "neck":
        ImplicitActuatorCfg(
            joint_names_expr=["zhead_1_joint", "zhead_2_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "zhead_1_joint": 150.0,
                "zhead_2_joint": 150.0,
            },
            damping={
                "zhead_1_joint": 5.0,
                "zhead_2_joint": 5.0,
            },
        ),
        "legs":
        ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_r1_joint",
                "leg_r2_joint", "leg_r3_joint", "leg_r4_joint"
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "leg_l1_joint": 200.0,
                "leg_l2_joint": 200.0,
                "leg_l3_joint": 200.0,
                "leg_l4_joint": 200.0,
                "leg_r1_joint": 200.0,
                "leg_r2_joint": 200.0,
                "leg_r3_joint": 200.0,
                "leg_r4_joint": 200.0,
            },
            damping={
                "leg_l1_joint": 5.0,
                "leg_l2_joint": 5.0,
                "leg_l3_joint": 5.0,
                "leg_l4_joint": 5.0,
                "leg_r1_joint": 5.0,
                "leg_r2_joint": 5.0,
                "leg_r3_joint": 5.0,
                "leg_r4_joint": 5.0,
            },
        ),
        "feet":
        ImplicitActuatorCfg(
            joint_names_expr=["leg_l5_joint", "leg_l6_joint", "leg_r5_joint", "leg_r6_joint"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={
                "leg_l5_joint": 20.0,
                "leg_l6_joint": 20.0,
                "leg_r5_joint": 20.0,
                "leg_r6_joint": 20.0,
            },
            damping={
                "leg_l5_joint": 4.0,
                "leg_l6_joint": 4.0,
                "leg_r5_joint": 20.0,
                "leg_r6_joint": 20.0,
            },
        ),
        "arms":
        ImplicitActuatorCfg(
            joint_names_expr=["zarm_.*"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "zarm_.*": 40.0,
            },
            damping={
                "zarm_.*": 10.0,
            },
        ),
    },
)

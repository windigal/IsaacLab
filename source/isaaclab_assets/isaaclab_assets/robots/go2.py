# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree_actuators import UnitreeActuatorCfg_Go2HV


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.8


UNITREE_MODEL_DIR = "D:/github/IsaacLab/models"

UNITREE_GO2_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV":
        UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint",
        "FL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint",
        "RL_thigh_joint", "RL_calf_joint"
    ],
    # fmt: on
)

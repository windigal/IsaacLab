# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg_leju_wbc import (
    LocomotionVelocityHighFreqRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import LejuKuavo42_V2_CFG  # isort: skip

@configclass
class LejuWBCRewards():
    Foot_Swing_Tracking = RewTerm(
        func=mdp.Foot_Swing_Tracking,
        weight=3.2,
        params={
            "command_name":"base_velocity",
        },
    )
    Feet_Symmetry = RewTerm(
        func=mdp.Feet_Symmetry,
        weight=-5.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"])
        },
    )
    Termination = RewTerm(
        func=mdp.termination,
        weight=-200.0
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "target_feet_height":0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        }
    )
    feet_contact_number = RewTerm(
        func=mdp.feet_contact_number,
        weight=1.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
        }
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
        }
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        }
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=0.2,
        params={
            "min":0.25,
            "max":0.6,
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        }
    )
    knee_distance = RewTerm(
        func=mdp.knee_distance,
        weight=0.2,
        params={
            "min":0.25,
            "max":0.6,
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l4_link", "leg_r4_link"]),
        }
    )
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "max_contact_force": 550,
        }
    )
    tracking_lin_vel = RewTerm(
        func=mdp.tracking_lin_vel,
        weight=1.6,
        params={
            "tracking_sigma":5,
            "command_name":"base_velocity",
        }
    )
    tracking_ang_vel = RewTerm(
        func=mdp.tracking_ang_vel,
        weight=1.1,
        params={
            "tracking_sigma":5,
            "command_name":"base_velocity",
        }
    )
    vel_mismatch_exp = RewTerm(
        func=mdp.vel_mismatch_exp,
        weight=0.5
    )
    low_speed = RewTerm(
        func=mdp.low_speed,
        weight=0.2,
        params={
            "command_name":"base_velocity",
        }
    )
    track_vel_hard = RewTerm(
        func=mdp.track_vel_hard,
        weight=0.5,
        params={
            "command_name":"base_velocity",
        }        
    )
    default_joint_pos = RewTerm(
        func=mdp.default_joint_pos,
        weight=0.5,
    )
    orientation = RewTerm(
        func=mdp.orientation,
        weight=1.0,
    )
    base_height = RewTerm(
        func=mdp.base_height,
        weight=0.2,
        params={
            "base_height_target":0.85,
        }
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness,
        weight=-2e-3,
    )
    torque = RewTerm(
        func=mdp.torque,
        weight=-1e-5,
    )
    dof_vel = RewTerm(
        func=mdp.dof_vel,
        weight=-5e-4,
    )
    dof_acc = RewTerm(
        func=mdp.dof_acc,
        weight=-1e-7,
    )
        
  
@configclass
class LejuWBCRoughEnvCfg(LocomotionVelocityHighFreqRoughEnvCfg):
    """ Leju v2: Remove arm joints."""
    rewards = LejuWBCRewards()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = LejuKuavo42_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

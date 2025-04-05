# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg_leju import (
    LocomotionVelocityHighFreqRoughEnvCfg,
    RewardsCfg,
)
# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg_leju_test import (
#     LocomotionVelocityRoughEnvCfg,
#     LocomotionVelocityHighFreqRoughEnvCfg,
#     RewardsCfg,
# )

##
# Pre-defined configs
##
from isaaclab_assets import LejuKuavo42_CFG, LejuKuavo42_V1_CFG, LejuKuavo42_V2_CFG  # isort: skip


@configclass
class LejuRewards(RewardsCfg):
    """Reward terms for the MDP."""
    feet_alternate = None

@configclass
class LejuV1Rewards(LejuRewards):
    feet_alternate = None


@configclass
class LejuRoughEnvCfg(LocomotionVelocityHighFreqRoughEnvCfg):
    """ Leju v0 """
    rewards: LejuRewards = LejuRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = LejuKuavo42_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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


@configclass
class LejuV1RoughEnvCfg(LocomotionVelocityHighFreqRoughEnvCfg):
    """ Leju v1: Remove two head joints and low pd control for the legs and arms. """
    rewards: LejuRewards = LejuV1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = LejuKuavo42_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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


@configclass
class LejuRoughEnvCfg_PLAY(LejuRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class LejuV2Rewards():
    joint_pos = RewTerm(
        func=mdp.joint_pos,
        weight=3.2,
        params={
            "cycle_steps":64,
            "command_name":"base_velocity",
            "constant":True,
        }
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "cycle_steps":64,
            "target_feet_height":0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        }
    )
    feet_contact_number = RewTerm(
        func=mdp.feet_contact_number,
        weight=1.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "cycle_steps":64,
        }
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "cycle_steps":64,
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
            "cycle_steps":64,
            "base_height_target":0.85,
        }
    )
    # base_acc = RewTerm(
    #     func=mdp.base_acc,
    #     weight=0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])
    #     }
    # )
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
    termination = RewTerm(
        func=mdp.termination,
        weight=-200.0
    )

  
@configclass
class LejuV2RoughEnvCfg(LejuV1RoughEnvCfg):
    """ Leju v2: Remove arm joints."""
    rewards: LejuRewards = LejuV2Rewards()
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
        # Commands
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


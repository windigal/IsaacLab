# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg_leju import (
    LocomotionVelocityRoughEnvCfg,
    LocomotionVelocityHighFreqRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets import LejuKuavo42_CFG, LejuKuavo42_V1_CFG, LejuKuavo42_V2_CFG  # isort: skip


@configclass
class LejuRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.125,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "threshold": 0.35,
        },
    )
    feet_alternate = RewTerm(
        func=mdp.feet_alternate,
        weight=-1e2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "threshold": 0.15,
            "has_head": True,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_pos_limits_legs = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_.*"])},
    )
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-1.2,
        params={"target_height": 0.85})

@configclass
class LejuV1Rewards(LejuRewards):
    feet_alternate = RewTerm(
        func=mdp.feet_alternate,
        weight=-1e2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "threshold": 0.15,
            "version": "v1",
        },
    )


@configclass
class LejuRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
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

        # Terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

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

        # Terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

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
class LejuV2Rewards(LejuRewards):
    feet_alternate = None
    feet_air_time = None
    dof_pos_limits = None
    dof_torques_l2 = None
    joint_deviation_arms = None
    joint_pos_limits_legs = None
    joint_pos_ref_dif = RewTerm(
        func = mdp.joint_pos_ref_dif,
        weight = 2.0,
        params={"cycle_steps":120,
                "command_name": "base_velocity",
        }
    )
    feet_air_time_leju = RewTerm(
        func = mdp.feet_air_time_leju,
        weight = 1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_l6_link", "leg_r6_link"]),
            "threshold": 0.6,
            "cycle_steps":120,
        },
    )
    default_joint_pos = RewTerm(
        func = mdp.default_joint_pos,
        weight = 0.5, 
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

        # Terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


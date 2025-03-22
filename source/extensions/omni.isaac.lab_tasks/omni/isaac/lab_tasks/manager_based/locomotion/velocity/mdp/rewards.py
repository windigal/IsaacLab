# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from .utils import interpolation, trun_sin
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # print(f"{air_time=}")
    # print(f"{contact_time=}")
    long_air_time_indices = torch.nonzero(air_time > torch.full_like(air_time, 1.5))
    long_contact_time_indices = torch.nonzero(contact_time > torch.full_like(contact_time, 1.5))
    reward[long_air_time_indices[:, 0]] = -threshold / 5
    reward[long_contact_time_indices[:, 0]] = -threshold / 5
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_alternate(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, version: str = "v0") -> torch.Tensor:
    """ Calculate the reward for alternating feet.
    This rewards include contact ratio and actoins of the left and right legs.

    Contact ratio is the ratio of the contact force between the left and right legs, we expect ratio to be 0.5,
    so the reward is the absolute difference between the ratio and 0.5.

    Also, we expect the left and right legs to alternate their actions, same as before, we expect this ratio to be 0.5.
    """
    assert version in ["v0", "v1", "v2"], "version should be in v0, v1 or v2"
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)
    is_contact = contact_history > contact_sensor.cfg.force_threshold
    contact_ratio = torch.sum(is_contact.float(), dim=1) / is_contact.shape[1]
    devition = torch.abs(contact_ratio - 0.5)
    reward = torch.where(devition < threshold, torch.zeros_like(devition), devition - threshold)
    reward = torch.sum(reward, dim=-1)
    leg_index = {"v0": [10, 14, 11, 15], "v1": [8, 12, 9, 13], "v2": [4, 6, 5, 7]}
    left_leg_action = env.action_manager.action_history[:, :, [leg_index[version][0], leg_index[version][1]]]
    right_leg_action = env.action_manager.action_history[:, :, [leg_index[version][2], leg_index[version][3]]]
    is_right_leg_higher = torch.where(left_leg_action <= right_leg_action, torch.zeros_like(left_leg_action), torch.ones_like(left_leg_action))
    right_leg_higher_ratio = torch.sum(is_right_leg_higher, dim=1) / is_right_leg_higher.shape[1]
    # print(f"{devition_2=}")
    devition_2 = torch.abs(right_leg_higher_ratio - 0.5)
    reward_2 = torch.where(devition_2 < threshold, torch.zeros_like(devition_2), devition_2 - threshold)
    reward += torch.sum(reward_2, dim=-1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward = torch.where(env.reset_terminated | env.reset_time_outs, reward, torch.zeros_like(reward))
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


# rewards from humanoidgym
def joint_pos_ref_dif(env: ManagerBasedRLEnv, cycle_steps: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    def _compute_ref_state(env, cycle_steps: float, command_name: str) -> torch.Tensor:
        command_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1).cpu()
        phase = env.episode_length_buf / cycle_steps
        # left foot stance phase set to default joint pos, l3, l5, l6
        ref_dof_pos = torch.zeros(env.scene.num_envs, sum(env.action_manager.action_term_dim), device = env.device)
        ref_dof_pos[:, 4] = trun_sin(2 * torch.pi * phase, interpolation['min'][0](command_vel), interpolation['max'][0](command_vel))
        ref_dof_pos[:, 6] = trun_sin(2 * torch.pi * phase, interpolation['min'][1](command_vel), interpolation['max'][1](command_vel))
        ref_dof_pos[:, 8] = trun_sin(2 * torch.pi * phase, interpolation['min'][2](command_vel), interpolation['max'][2](command_vel))
        # right foot stance phase set to default joint pos, r3, r5, r6
        ref_dof_pos[:, 5] = trun_sin(2 * torch.pi * (phase - 1/2), interpolation['min'][0](command_vel), interpolation['max'][0](command_vel))
        ref_dof_pos[:, 7] = trun_sin(2 * torch.pi * (phase - 1/2), interpolation['min'][1](command_vel), interpolation['max'][1](command_vel))
        ref_dof_pos[:, 9] = trun_sin(2 * torch.pi * (phase - 1/2), interpolation['min'][2](command_vel), interpolation['max'][2](command_vel))
        return ref_dof_pos
    
    ref_dof_pos = _compute_ref_state(env, cycle_steps, command_name)
    asset = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - ref_dof_pos
    rew = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    return rew

def feet_air_time_leju(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, cycle_steps: float):
    def _get_gait_phase(env, cycle_steps: float) -> torch.Tensor:
        phase = env.episode_length_buf / cycle_steps
        sin_pos = torch.sin(2 * torch.pi * phase)
        stance_mask = torch.zeros(env.scene.num_envs, 2, device=env.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    stance_mask = _get_gait_phase(env, cycle_steps)
    contact_flit = torch.all(torch.eq(in_contact, stance_mask), dim=1).int()
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    fit_state = torch.logical_and(single_stance, contact_flit)
    reward = torch.min(torch.where(fit_state.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    long_air_time_indices = torch.nonzero(air_time > torch.full_like(air_time, 1.0))
    long_contact_time_indices = torch.nonzero(contact_time > torch.full_like(contact_time, 1.0))
    reward[long_air_time_indices[:, 0]] = 0
    reward[long_contact_time_indices[:, 0]] = 0
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def default_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_diff =  joint_pos - torch.zeros_like(joint_pos)
    left_yaw_roll = joint_diff[:, [0,2]]
    right_yaw_roll = joint_diff[:, [1,3]]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
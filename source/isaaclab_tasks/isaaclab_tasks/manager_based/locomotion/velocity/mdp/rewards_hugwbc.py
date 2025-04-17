# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
从HUGWBC移植的所有奖励, 用于leju_v2训练
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from .utils import interpolation, trun_sin
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import phi_smooth, euler_xyz_from_quat
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_ref_state_wbc(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    phi = mdp.phi(env, command_name=command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_x = env.command_manager.get_command(command_name)[:, 0]
    delta = torch.max(vel_x - torch.ones_like(vel_x), torch.zeros_like(vel_x))
    # left foot stance phase set to default joint pos, l3, l4, l5
    ref_dof_pos = torch.zeros(env.scene.num_envs, sum(env.action_manager.action_term_dim), device = env.device)
    ref_dof_pos[:, 4] = trun_sin(2 * torch.pi * phi[:, 0], -0.52 - delta * 0.18, -0.02)
    ref_dof_pos[:, 6] = trun_sin(2 * torch.pi * (phi[:, 0] - 1/2), 0.02, 1.02 + delta * 0.2)
    ref_dof_pos[:, 6] = torch.max(ref_dof_pos[:, 6], asset.data.default_joint_pos[:, 6])
    ref_dof_pos[:, 8] = trun_sin(2 * torch.pi * phi[:, 0], -0.55, -0.05)
    ref_dof_pos[:, 8] = torch.where(ref_dof_pos[:, 8] < asset.data.default_joint_pos[:, 8], 
                                    ref_dof_pos[:, 8], - ref_dof_pos[:, 4] - ref_dof_pos[:, 6])
    # right foot stance phase set to default joint pos, r3, r4, r5
    ref_dof_pos[:, 5] = trun_sin(2 * torch.pi * phi[:, 1], -0.52 - delta * 0.18, -0.02)
    ref_dof_pos[:, 7] = trun_sin(2 * torch.pi * (phi[:, 1] - 1/2), 0.02, 1.02 + delta * 0.2)
    ref_dof_pos[:, 7] = torch.max(ref_dof_pos[:, 7], asset.data.default_joint_pos[:, 7])
    ref_dof_pos[:, 9] = trun_sin(2 * torch.pi * phi[:, 1], -0.55, -0.05)
    ref_dof_pos[:, 9] = torch.where(ref_dof_pos[:, 9] < asset.data.default_joint_pos[:, 9], 
                                    ref_dof_pos[:, 9], - ref_dof_pos[:, 5] - ref_dof_pos[:, 7])
    # run test
    # ref_dof_pos[:, 4] = trun_sin(2 * torch.pi * phi[:, 0], -0.7854, -0)
    # ref_dof_pos[:, 6] = trun_sin(2 * torch.pi * (phi[:, 0] + 3/16), 0.5236, 1.7453)
    # ref_dof_pos[:, 8] = trun_sin(2 * torch.pi * (phi[:, 0] - 3/8), -0.4363, 0.0873)
    # # right foot stance phase set to default joint pos, r3, r4, r5
    # ref_dof_pos[:, 5] = trun_sin(2 * torch.pi * phi[:, 1], -0.7854, -0)
    # ref_dof_pos[:, 7] = trun_sin(2 * torch.pi * (phi[:, 1] + 3/16), 0.5236, 1.7453)
    # ref_dof_pos[:, 9] = trun_sin(2 * torch.pi * (phi[:, 1] - 3/8), -0.4363, 0.0873)
    return ref_dof_pos.to(device=env.device)

# ================================================ Rewards ================================================== #

# Task Rewards
def Linear_Velocity_Tracking(env: ManagerBasedRLEnv, 
                             tracking_sigma: float, 
                             command_name: str, 
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # 计算线速度误差
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_b[:, :2]),
        dim=1,
    )
    # 计算奖励
    rew = torch.exp(-lin_vel_error * tracking_sigma)
    return rew

def Angular_Velocity_Tracking(env: ManagerBasedRLEnv, 
                              tracking_sigma: float, 
                              command_name: str, 
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # 计算角速度误差
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_b[:, 2]
    )
    # 计算奖励
    rew = torch.exp(-ang_vel_error * tracking_sigma)
    return rew

# Behavior Rewards
def Body_Height_Tracking(env: ManagerBasedRLEnv, 
                         base_height_target: float,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    根据机器人的基础高度计算奖励, 根据平均抬脚高度计算
    '''
    # 计算双脚平均高度
    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = asset.data.joint_pos[:, [10, 11]]
    stance_mask = mdp.get_gait_phase(env)
    measured_heights = torch.sum(feet_height * stance_mask, dim=1) / torch.sum(stance_mask)
    # 计算机身高度
    base_height = asset.data.root_link_pos_w[:, 2] - (measured_heights - 0.05)
    # 计算奖励
    rew = torch.exp(-torch.abs(base_height - base_height_target) * 100)
    return rew

def Foot_Swing_Tracking(env: ManagerBasedRLEnv, 
                        command_name: str, 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    计算当前关节位置和参考关节位置的差, 参考joint_pos
    '''
    # 计算当前参考位置
    ref_dof_pos = compute_ref_state_wbc(env, command_name, asset_cfg)
    # 计算差值
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - ref_dof_pos
    # 计算奖励
    rew = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    # env.dof_pos_buf[env.episode_length_buf] = asset.data.joint_pos[0, [4,6,8,5,7,9]]
    # env.ref_dof_pos_buf[env.episode_length_buf] = ref_dof_pos[0, [4,6,8,5,7,9]]
    return rew


def Contact_Swing_Tracking(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    '''
    根据与步态阶段对齐的脚接触次数计算奖励
    '''
    # 确定双脚接触状态
    phi = mdp.phi(env, command_name="base_velocity")
    C = phi_smooth(phi)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    body_vel_norm = asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    rew = - (1 - C) * (1 - torch.exp(-contact_force / 50 )) - C * (1 - torch.exp(-body_vel_norm / 5))
    return torch.sum(rew, dim=1)


# Behavior Rewards
def Vertical_Body_Movement(env: ManagerBasedRLEnv, 
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    减小lin_z_vel
    '''
    # 计算lin_z_vel
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_mismatch = torch.abs(asset.data.root_com_lin_vel_b[:, 2])
    # 计算奖励
    rew = lin_mismatch
    return rew

def RP_Angular_Velocity_Tracking(env: ManagerBasedRLEnv, 
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    减小ang_xy_vel
    '''
    # 计算ang_xy_vel
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_mismatch = torch.norm(asset.data.root_com_ang_vel_b[:, :2], dim=1)
    # 计算奖励
    rew = ang_mismatch
    return rew

def Action_Rate(env: ManagerBasedRLEnv):
    '''
    保持机器人动作平滑, 这里与IsaacLab保持一致, 采用一步插值
    '''
    # 计算当前动作与上一帧动作的差
    action_diff = torch.norm(env.action_manager.action - env.action_manager.prev_action, dim=1)
    # 计算奖励
    rew = action_diff
    return rew

def Joint_Torque(env: ManagerBasedRLEnv, 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    限制较大的关节力矩
    '''
    # 获取当前应用的力矩
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    # 计算奖励
    rew = torch.norm(torques, dim=1)
    return rew

def Joint_Acceleration(env: ManagerBasedRLEnv, 
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    限制较大的关节加速度
    '''
    # 获取当前的关节加速度
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    # 计算奖励
    rew = torch.norm(joint_acc, dim=1)
    return rew

def Hip_Joint_Deviation(env: ManagerBasedRLEnv, 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    计算保持关节位置接近默认位置的奖励，重点惩罚偏航和横滚方向的偏差
    '''
    # 计算关节误差
    asset: Articulation = env.scene[asset_cfg.name]
    joint_diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # 提取大腿偏航与横滚关节数据
    hip_yaw_roll = joint_diff[:, [0, 1, 2, 3]]
    # 计算奖励
    rew = torch.norm(hip_yaw_roll, dim=1)
    return rew

def Feet_Symmetry(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    在双脚相位一致时鼓励对称性
    '''
    phi = mdp.phi(env, command_name=command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_com_pos_w[:, asset_cfg.body_ids, :2]   
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1) 
    rew = foot_dist * (phi[:, 0] == phi[:, 1]).float()
    return rew


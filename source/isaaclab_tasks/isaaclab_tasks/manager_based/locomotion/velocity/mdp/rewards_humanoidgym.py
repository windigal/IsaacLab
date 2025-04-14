# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
从Humanoidgym移植的所有奖励, 用于leju_v2训练
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from .utils import interpolation, trun_sin
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, euler_xyz_from_quat
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    if env.spec is not None and "wbc" in env.spec.id:
        phase = mdp.phi(env, command_name="base_velocity")
        sin_pos = torch.sin(2 * torch.pi * phase)
        stance_mask = torch.zeros(env.scene.num_envs, 2, device=env.device)
        stance_mask[:, 0] = sin_pos[:, 0] >= 0
        stance_mask[:, 1] = sin_pos[:, 0] >= 0
    else:
        episode_length_buf = env.episode_length_buf if hasattr(env, "episode_length_buf") else torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        cycle_steps = env.cycle_steps if hasattr(env, "cycle_steps") else 64
        phase = episode_length_buf / cycle_steps
        sin_pos = torch.sin(2 * torch.pi * phase)
        stance_mask = torch.zeros(env.scene.num_envs, 2, device=env.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0

    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    return stance_mask.to(device=env.device)
    
def compute_ref_state(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1).cpu()
    cycle_steps = env.cycle_steps if hasattr(env, "cycle_steps") else 64
    phase = env.episode_length_buf / cycle_steps
    # left foot stance phase set to default joint pos, l3, l4, l5
    ref_dof_pos = torch.zeros(env.scene.num_envs, sum(env.action_manager.action_term_dim), device = env.device)
    ref_dof_pos[:, 4] = trun_sin(2 * torch.pi * phase, interpolation['min'][0](command_vel), interpolation['max'][0](command_vel))
    ref_dof_pos[:, 6] = trun_sin(2 * torch.pi * (phase - 1/2), interpolation['min'][1](command_vel), interpolation['max'][1](command_vel))
    ref_dof_pos[:, 8] = trun_sin(2 * torch.pi * phase, interpolation['min'][2](command_vel), interpolation['max'][2](command_vel))
    # right foot stance phase set to default joint pos, r3, r4, r5
    ref_dof_pos[:, 5] = trun_sin(2 * torch.pi * (phase - 1/2), interpolation['min'][0](command_vel), interpolation['max'][0](command_vel))
    ref_dof_pos[:, 7] = trun_sin(2 * torch.pi * phase, interpolation['min'][1](command_vel), interpolation['max'][1](command_vel))
    ref_dof_pos[:, 9] = trun_sin(2 * torch.pi * (phase - 1/2), interpolation['min'][2](command_vel), interpolation['max'][2](command_vel))
    return ref_dof_pos.to(device=env.device)

def compute_ref_state_constant(env: ManagerBasedRLEnv) -> torch.Tensor:
    # command_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1).cpu()
    # command_vel.fill_(0.4)
    episode_length_buf = env.episode_length_buf if hasattr(env, "episode_length_buf") else torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    cycle_steps = env.cycle_steps if hasattr(env, "cycle_steps") else 64
    phase = episode_length_buf / cycle_steps
    # left foot stance phase set to default joint pos, l3, l4, l5
    ref_dof_pos = torch.zeros(env.scene.num_envs, sum(env.action_manager.action_term_dim), device = env.device)
    ref_dof_pos[:, 4] = trun_sin(2 * torch.pi * phase, -0.52, -0.02)
    ref_dof_pos[:, 6] = trun_sin(2 * torch.pi * (phase - 1/2), 0.02, 1.02)
    ref_dof_pos[:, 8] = trun_sin(2 * torch.pi * phase, -0.55, -0.05)
    # right foot stance phase set to default joint pos, r3, r4, r5
    ref_dof_pos[:, 5] = trun_sin(2 * torch.pi * (phase - 1/2), -0.52, -0.02)
    ref_dof_pos[:, 7] = trun_sin(2 * torch.pi * phase, 0.02, 1.02)
    ref_dof_pos[:, 9] = trun_sin(2 * torch.pi * (phase - 1/2), -0.55, -0.05)
    return ref_dof_pos.to(device=env.device)

# ================================================ Rewards ================================================== #
def joint_pos(env: ManagerBasedRLEnv, 
              command_name: str, 
              constant: bool = False,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    计算当前关节位置和参考关节位置的差
    '''
    # 计算当前参考位置
    if not constant:
        ref_dof_pos = compute_ref_state(env, command_name)
    else:
        ref_dof_pos = compute_ref_state_constant(env)
    # 计算差值
    asset: Articulation = env.scene[asset_cfg.name]
    ref_dof_pos[:, 4] = torch.min(ref_dof_pos[:, 4], asset.data.default_joint_pos[:, 4])
    ref_dof_pos[:, 5] = torch.min(ref_dof_pos[:, 5], asset.data.default_joint_pos[:, 5])
    ref_dof_pos[:, 6] = torch.max(ref_dof_pos[:, 6], asset.data.default_joint_pos[:, 6])
    ref_dof_pos[:, 7] = torch.max(ref_dof_pos[:, 7], asset.data.default_joint_pos[:, 7])
    ref_dof_pos[:, 8] = torch.min(ref_dof_pos[:, 8], asset.data.default_joint_pos[:, 8])
    ref_dof_pos[:, 9] = torch.min(ref_dof_pos[:, 9], asset.data.default_joint_pos[:, 9])
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - ref_dof_pos
    # 计算奖励
    rew = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    # env.dof_pos_buf[env.episode_length_buf] = asset.data.joint_pos[0, [4,6,8,5,7,9]]
    # env.ref_dof_pos_buf[env.episode_length_buf] = ref_dof_pos[0, [4,6,8,5,7,9]]
    return rew


def feet_clearance(env: ManagerBasedRLEnv, 
                   sensor_cfg: SceneEntityCfg, 
                   target_feet_height: float,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    鼓励在步态阶段抬脚
    '''
    # 确定双脚接触状态
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5
    # 计算脚高度的变化
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_com_pos_w[:, asset_cfg.body_ids, 2]
    delta_z = feet_z - env.last_feet_z
    env.feet_height += delta_z
    env.last_feet_z = feet_z
    # 计算当前步态下处于摆动阶段的脚
    swing_mask = 1 - get_gait_phase(env)
    # 计算奖励, 在跑步阶段提高抬脚目标
    # lin_x = env.command_manager.get_command("base_velocity")[:, 0]
    # target_feet_height = target_feet_height + 0.06 * torch.max(lin_x - 1, 0)[0]
    rew = torch.abs(env.feet_height - target_feet_height) < 0.01
    rew = torch.sum(rew * swing_mask, dim=1)
    env.feet_height *= ~contact
    return rew

def feet_contact_number(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    '''
    根据与步态阶段对齐的脚接触次数计算奖励
    '''
    # 确定双脚接触状态
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5
    # 计算当前步态下处于站立阶段的脚
    stance_mask = get_gait_phase(env)
    # 计算奖励
    rew = torch.where(in_contact == stance_mask, 1.0, -0.3)
    return torch.mean(rew, dim=1)


def feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    '''
    根据当前是否是刚落地, 给出上次悬空的时间奖励
    '''
    # 判断当前是否刚落地
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5
    last_contacts = contact_sensor.data.net_forces_w_history[:, -2, sensor_cfg.body_ids, 2] <= 5
    stance_mask = get_gait_phase(env)
    contact_flit = torch.logical_and(torch.logical_or(contact, stance_mask), last_contacts)
    # 计算奖励
    rew = torch.sum(torch.clamp(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], 0, 0.5) * contact_flit, dim=1)
    return rew

def foot_slip(env: ManagerBasedRLEnv, 
              sensor_cfg: SceneEntityCfg, 
              asset_cfg: SceneEntityCfg):
    '''
    计算减少脚滑的奖励, 参考feet_slide
    '''
    # 找到历史里最大的接触力判断是否接触
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算奖励，这里参考humanoid_gym对速度进行开方
    body_vel_norm = asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    rew = torch.sqrt(body_vel_norm)
    return torch.sum(rew * contacts, dim=1)

def feet_distance(env: ManagerBasedRLEnv, 
                  min: float,
                  max: float,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    根据双脚之间的距离计算奖励, 惩罚过小或者过大
    '''
    # 计算双脚距离
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_com_pos_w[:, asset_cfg.body_ids, :2]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    d_min = torch.clamp(foot_dist - min, -0.5, 0.)
    d_max = torch.clamp(foot_dist - max, 0., 0.5)
    # 计算奖励
    rew = (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
    return rew

def knee_distance(env: ManagerBasedRLEnv, 
                  min: float,
                  max: float, 
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    根据双脚之间的距离计算奖励, 惩罚过小或者过大
    '''
    # 计算双脚距离
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos = asset.data.body_com_pos_w[:, asset_cfg.body_ids, :2]
    knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
    d_min = torch.clamp(knee_dist - min, -0.5, 0.)
    d_max = torch.clamp(knee_dist - max, 0., 0.5)
    # 计算奖励
    rew = (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
    return rew
    
def feet_contact_forces(env: ManagerBasedRLEnv, 
                        sensor_cfg: SceneEntityCfg, 
                        max_contact_force: float):
    '''
    惩罚过大的触底力
    '''
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    rew = torch.sum((contact_force - max_contact_force).clip(0, 400), dim=1)
    return rew

def tracking_lin_vel(env: ManagerBasedRLEnv, 
                     tracking_sigma: float, 
                     command_name: str, 
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    线速度追踪奖励, 参考track_lin_vel_xy_exp
    '''
    # 计算线速度误差
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_b[:, :2]),
        dim=1,
    )
    # 计算奖励
    rew = torch.exp(-lin_vel_error * tracking_sigma)
    return rew

def tracking_ang_vel(env: ManagerBasedRLEnv, 
                     tracking_sigma: float, 
                     command_name: str, 
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    角速度追踪奖励, 参考track_ang_vel_z_exp
    '''
    # 计算角速度误差
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_b[:, 2]
    )
    # 计算奖励
    rew = torch.exp(-ang_vel_error * tracking_sigma)
    return rew

def vel_mismatch_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    减小lin_z_vel, ang_xy_vel
    '''
    # 计算lin_z_vel, ang_xy_vel
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_mismatch = torch.exp(-torch.square(asset.data.root_com_lin_vel_b[:, 2]) * 10)
    ang_mismatch = torch.exp(-torch.norm(asset.data.root_com_ang_vel_b[:, :2], dim=1) * 5.)
    # 计算奖励
    rew = (lin_mismatch + ang_mismatch) / 2
    return rew

def low_speed(env: ManagerBasedRLEnv, 
              command_name: str, 
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    检查x线速度大小是否和指令一致, 防止过大或者过小。
    '''
    # 计算当前x线速度与指令
    asset: RigidObject = env.scene[asset_cfg.name]
    absolute_speed = torch.abs(asset.data.root_com_lin_vel_b[:, 0])
    absolute_command = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    # 过大过小或者适配判断
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)
    # 检查速度方向是否一致
    sign_mismatch = torch.sign(asset.data.root_com_lin_vel_b[:, 0]) != \
                    torch.sign(env.command_manager.get_command(command_name)[:, 0])
    # 计算奖励
    rew = torch.zeros_like(absolute_speed)
    rew[speed_too_low] = -1.0
    rew[speed_too_high] = 0.
    rew[speed_desired] = 1.2
    rew[sign_mismatch] = -2.0
    rew *= absolute_command > 0.1
    return rew

def track_vel_hard(env: ManagerBasedRLEnv,  
                   command_name: str, 
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    追踪角速度和线速度
    '''
    # 计算线速度和角速度的误差
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_b[:, :2]),
        dim=1,
    )
    lin_vel_error_exp = torch.exp(-lin_vel_error * 10)
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_b[:, 2]
    )
    ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
    # 计算奖励
    linear_error = 0.2 * (lin_vel_error + ang_vel_error)
    rew = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
    return rew

def default_joint_pos(env: ManagerBasedRLEnv, 
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    计算保持关节位置接近默认位置的奖励，重点惩罚偏航和横滚方向的偏差
    '''
    # 计算关节误差
    asset: Articulation = env.scene[asset_cfg.name]
    joint_diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # 提取大腿偏航与横滚关节数据
    left_yaw_roll = joint_diff[:, [0, 2]]
    right_yaw_roll = joint_diff[:, [1, 3]]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    # 计算奖励
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    rew = torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    return rew

def base_height(env: ManagerBasedRLEnv, 
                base_height_target: float,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    根据机器人的基础高度计算奖励, 根据平均抬脚高度计算
    '''
    # 计算双脚平均高度
    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = asset.data.joint_pos[:, [10, 11]]
    stance_mask = get_gait_phase(env)
    measured_heights = torch.sum(feet_height * stance_mask, dim=1) / torch.sum(stance_mask)
    # 计算机身高度
    base_height = asset.data.root_link_pos_w[:, 2] - (measured_heights - 0.05)
    # 计算奖励
    rew = torch.exp(-torch.abs(base_height - base_height_target) * 100)
    return rew

def orientation(env: ManagerBasedRLEnv, 
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    保持机身没有自身旋转与前后左右倾斜
    '''
    # 计算重力投影
    asset: RigidObject = env.scene[asset_cfg.name]
    orientation = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    # 计算欧拉角
    quat = asset.data.root_com_quat_w
    r, p, _ = euler_xyz_from_quat(quat)
    euler_xy = torch.sum(torch.abs(torch.stack((r, p), dim=1)), dim=1)
    # 计算奖励
    rew = (torch.exp(-euler_xy * 10) + torch.exp(-orientation * 20)) / 2
    return rew

def base_acc(env: ManagerBasedRLEnv, 
             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    保持机器人平稳加速
    '''
    # 获取机器人加速度和角加速度
    asset: Articulation = env.scene[asset_cfg.name]
    root_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, :6]
    # 计算奖励
    rew = torch.exp(-torch.norm(root_acc, dim=-1) * 3)
    return torch.sum(rew, dim=1)

def action_smoothness(env: ManagerBasedRLEnv):
    '''
    保持机器人动作平滑, 这里与IsaacLab保持一致, 采用一步插值
    '''
    # 计算当前动作与上一帧动作的差
    action_diff = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    # 计算当前动作, 这里与humanoidgym保持一致, 使用l1范数
    action = torch.sum(torch.square(env.action_manager.action), dim=1)
    # 计算奖励
    rew = action_diff + 0.05 * action
    return rew

def torque(env: ManagerBasedRLEnv, 
           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    限制较大的关节力矩
    '''
    # 获取当前应用的力矩
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    # 计算奖励
    rew = torch.sum(torch.square(torques), dim=1)
    return rew

def dof_vel(env: ManagerBasedRLEnv, 
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    限制较大的关节速度
    '''
    # 获取当前的关节速度
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # 计算奖励
    rew = torch.sum(torch.square(joint_vel), dim=1)
    return rew

def dof_acc(env: ManagerBasedRLEnv, 
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    '''
    限制较大的关节加速度
    '''
    # 获取当前的关节加速度
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    # 计算奖励
    rew = torch.sum(torch.square(joint_acc), dim=1)
    return rew

def termination(env: ManagerBasedRLEnv):
    '''
    是否非自然终止
    '''
    return env.termination_manager.terminated.float()
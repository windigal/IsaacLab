from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.warp import raycast_mesh
from isaaclab.sensors import RayCaster
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
"""
Joint penalties.
"""


def energy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv,
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity,
                         dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                           stand_still_scale: float, velocity_threshold: float) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward,
                       stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2],
                                  dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.
                                                   body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(
                                                       1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.
                                                       body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(
                                                           1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:,
                              i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w,
                                                                    cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:,
                              i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w,
                                                                    cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(
        env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float,
                   std: float, tanh_mult: float) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] -
                                       target_height)
    # Get the feet clearance
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    height_scanner: RayCaster = env.scene.sensors["height_scanner"]
    terrain_mesh_path = height_scanner.cfg.mesh_prim_paths[0]
    terrain_warp_mesh = height_scanner.meshes[terrain_mesh_path]
    ray_starts = foot_positions.clone()
    ray_starts[..., 2] += 1.0
    ray_directions = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(ray_starts)
    ray_hits, _, _, _ = raycast_mesh(ray_starts=ray_starts.view(-1, 3), 
                                     ray_directions=ray_directions.view(-1, 3),
                                     mesh=terrain_warp_mesh)
    terrain_heights_under_feet = ray_hits[:, 2].view(foot_positions.shape[0], foot_positions.shape[1])
    foot_clearance = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - terrain_heights_under_feet

    foot_z_target_error = torch.square(foot_clearance - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh

    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_curriculum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_target_height: float,
    max_target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """
    Reward swinging feet for clearing a height that scales with the terrain difficulty.
    """
    terrain_levels = env.scene.terrain.terrain_levels
    max_terrain_level = env.scene.terrain.max_terrain_level - 1

    progress = torch.clamp(terrain_levels / max_terrain_level, 0.0, 1.0)
    current_target_height = min_target_height + progress * (max_target_height -
                                                            min_target_height) + 0.02

    asset: RigidObject = env.scene[asset_cfg.name]
    current_target_height = current_target_height.unsqueeze(1)

    # Get the feet clearance
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    height_scanner: RayCaster = env.scene.sensors["height_scanner"]
    terrain_mesh_path = height_scanner.cfg.mesh_prim_paths[0]
    terrain_warp_mesh = height_scanner.meshes[terrain_mesh_path]
    ray_starts = foot_positions.clone()
    ray_starts[..., 2] += 1.0
    ray_directions = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(ray_starts)
    ray_hits, _, _, _ = raycast_mesh(ray_starts=ray_starts.view(-1, 3), 
                                     ray_directions=ray_directions.view(-1, 3),
                                     mesh=terrain_warp_mesh)
    terrain_heights_under_feet = ray_hits[:, 2].view(foot_positions.shape[0], foot_positions.shape[1])
    foot_clearance = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - terrain_heights_under_feet

    foot_z_target_error = torch.square(foot_clearance - current_target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(last_air_time, dim=1) + torch.var(last_contact_time, dim=1)

def feet_regulation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_base_height: float):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get the feet clearance
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    height_scanner: RayCaster = env.scene.sensors["height_scanner"]
    terrain_mesh_path = height_scanner.cfg.mesh_prim_paths[0]
    terrain_warp_mesh = height_scanner.meshes[terrain_mesh_path]
    ray_starts = foot_positions.clone()
    ray_starts[..., 2] += 1.0
    ray_directions = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(ray_starts)
    ray_hits, _, _, _ = raycast_mesh(ray_starts=ray_starts.view(-1, 3), 
                                     ray_directions=ray_directions.view(-1, 3),
                                     mesh=terrain_warp_mesh)
    terrain_heights_under_feet = ray_hits[:, 2].view(foot_positions.shape[0], foot_positions.shape[1])
    diff = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - terrain_heights_under_feet
    foot_clearance = torch.max(diff, torch.zeros_like(diff))
    # replace inf/-inf with 0
    foot_clearance = torch.nan_to_num(foot_clearance, posinf=0.0, neginf=0.0)
    
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    rew = torch.sum(torch.norm(feet_vel, dim=-1) * torch.exp(- foot_clearance / 0.03 / target_base_height), dim=1)
    return rew

"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0  # type: ignore

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.pp_action), dim=1)

def track_ang_vel_z_exp_limit(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel. 
       If a joint breaks through the soft limit, the reward will be canceled """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    hip_deviation = joint_asset.data.joint_pos[:, asset_cfg.joint_ids] - joint_asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    in_limit = torch.all(torch.abs(hip_deviation) < 0.2, dim=1)
    return torch.exp(-ang_vel_error / std**2) * in_limit

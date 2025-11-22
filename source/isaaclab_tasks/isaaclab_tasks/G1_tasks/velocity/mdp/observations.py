from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, RayCaster
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import RigidObject
    from isaaclab.utils.warp import raycast_mesh


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def thigh_and_calf_contacts(env: ManagerBasedRLEnv,
                            thigh_sensor_name: str,
                            calf_sensor_name: str,
                            threshold: float = 1.0) -> torch.Tensor:
    """
    检查Go2机器人的大腿和小腿连杆是否与环境发生接触。
    """
    # -- CORRECTED SENSOR ACCESS --
    # 通过 env.scene.sensors 字典来安全地访问传感器
    thigh_sensor: ContactSensor = env.scene.sensors[thigh_sensor_name]
    calf_sensor: ContactSensor = env.scene.sensors[calf_sensor_name]

    # 从传感器对象中获取数据
    thigh_forces = thigh_sensor.data.net_forces_w
    calf_forces = calf_sensor.data.net_forces_w

    # 检查接触力的范数是否超过一个阈值来判断是否发生接触
    thigh_contact_bool = torch.norm(thigh_forces, dim=-1) > threshold
    calf_contact_bool = torch.norm(calf_forces, dim=-1) > threshold

    # 将布尔值转换为浮点数（0.0 或 1.0）
    thigh_contacts = thigh_contact_bool.float()
    calf_contacts = calf_contact_bool.float()

    # 拼接成一个 (num_envs, 8) 维的观察向量
    all_contacts = torch.cat([thigh_contacts, calf_contacts], dim=1)

    return all_contacts


def foot_contact_forces(env: ManagerBasedRLEnv, scale: float, sensor_name: str) -> torch.Tensor:
    """
    获取机器人脚部的3D接触力向量（在世界坐标系下）。
    """
    foot_sensor: ContactSensor = env.scene.sensors[sensor_name]
    foot_forces = foot_sensor.data.net_forces_w
    flattened_forces = torch.flatten(foot_forces, start_dim=1)

    return flattened_forces * scale

def feet_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
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
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - terrain_heights_under_feet
    foot_height = torch.nan_to_num(foot_height, posinf=0.0, neginf=0.0)
    return foot_height
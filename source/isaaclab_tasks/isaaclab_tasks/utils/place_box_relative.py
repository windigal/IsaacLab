import math
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate


@torch.jit.script
def yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    # 返回 yaw（弧度），输入 (N,4) w,x,y,z
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


@torch.no_grad()
@torch.no_grad()  
def reset_object_pose_relative_to_asset(  
    env,  
    env_ids,  
    asset_cfg_box: SceneEntityCfg,  
    asset_cfg_robot: SceneEntityCfg,  
    offset_xyz,  
    follow_robot_yaw: bool = True,  
    extra_yaw_deg: float = 0.0,  
):  
    scene = env.scene  
  
    asset_cfg_box.resolve(scene)  
    asset_cfg_robot.resolve(scene)  
    box_ent = scene[asset_cfg_box.name]  
    robot_ent = scene[asset_cfg_robot.name]  
  
    # 偏移与角度  
    offset = torch.as_tensor(offset_xyz, device=env.device, dtype=torch.float32)  
    extra_yaw = math.radians(float(extra_yaw_deg))  
  
    # 机器人根状态  
    root_state = robot_ent.data.root_state_w[env_ids]  
    p_robot = root_state[:, 0:3]  
    q_robot = root_state[:, 3:7]  
  
    # 旋转偏移到世界系  
    off_world = quat_rotate(q_robot, offset.unsqueeze(0).expand_as(p_robot))  
    p_box = p_robot + off_world  
  
    # 构造箱子的朝向  
    if follow_robot_yaw:  
        yaw = yaw_from_quat_wxyz(q_robot) + extra_yaw  
        cy = torch.cos(0.5 * yaw)  
        sy = torch.sin(0.5 * yaw)  
        q_box = torch.zeros_like(q_robot)  
        q_box[:, 0] = cy  
        q_box[:, 3] = sy  
    else:  
        q_box = torch.zeros_like(q_robot)  
        q_box[:, 0] = 1.0  
  
    # 使用 Isaac Lab 的正确 API  
    pose = torch.cat([p_box, q_box], dim=-1)  # (N, 7)  
    box_ent.write_root_pose_to_sim(pose, env_ids=env_ids)  
      
    # 清零速度  
    zeros_vel = torch.zeros(len(env_ids), 6, device=env.device)  
    box_ent.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)

# Add by FloatRIslet at 2025-07-01

import torch
import numpy as np
from gymnasium import spaces
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.leju_amp.motion.motion_loader import MotionLoader
from isaaclab.utils.math import quat_apply

class ManagerBasedRLAmpEnv(ManagerBasedRLEnv):
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motion_loader = MotionLoader(motion_file=self.cfg.amp.motion_file, device=self.device)

        amp_state_sample = self._compute_amp_state()                # (N, d)
        self._amp_dim = amp_state_sample.shape[-1]
        self._amp_hist = self.cfg.observations.policy.history_length
        self._incompatible_joint_names = []
        key_body_names = ["zarm_r6_link", "zarm_l6_link", "leg_r6_link", "leg_l6_link"]
        joint_names = [name for name in self.robot.data.joint_names if name not in self._incompatible_joint_names]
        self.lab_dof_index = self.robot.find_joints(joint_names)[0]
        
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        
        self.motion_dof_indexes = self._motion_loader.get_dof_index(joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        self.amp_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._amp_hist * self._amp_dim,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self._amp_hist, self.cfg.amp.amp_observation_space), device=self.device)
        self.extras["amp_obs"] = torch.zeros((self.num_envs, self._amp_hist * self._amp_dim),
            device=self.device, dtype=torch.float32
        )

        
    def _compute_amp_state(self, obs=None) -> torch.Tensor:
        """拼接 root 速度 + 关节位姿/速度，返回 (num_envs, d)。"""
        # base_line_vel = obs[:, :3]
        # base_ang_vel = obs[:, 3:6]
        if obs is None:
            return torch.zeros(
                (self.num_envs, 52),
                device=self.device, dtype=torch.float32
            )
        num_joints = 26
        joint_pos = obs[:, :num_joints]
        joint_vel = obs[:, num_joints:2 * num_joints]
        return torch.cat([joint_pos, joint_vel], dim=-1)      # (N,d)

    # ----------------------------------------------------------- #
    # 1.5  重载 step()：跑父类逻辑 ➜ 更新 AMP-buffer
    # ----------------------------------------------------------- #
    def step(self, action):
        obs, rew, term, trunc, extras = super().step(action)
        self._amp_buf = torch.roll(self._amp_buf, shifts=1, dims=1)
        self._amp_buf[:, 0, :] = self._compute_amp_state(obs['policy'])
        # 扁平化存进 extras，供 collect_amp_observation 使用
        self.extras["amp_obs"] = self._amp_buf.view(self.num_envs, -1)
        return obs, rew, term, trunc, extras


    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        (dof_positions,
         dof_velocities,
         body_positions,
         body_rotations,
         body_linear_velocities,
         body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples)
        # 用随机 MoCap 片段填满新 episode 的历史窗口
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.cfg.amp.amp_observation_space)

    
    def _reset_idx(self, env_ids):
        num_samples = len(env_ids) * self._amp_hist
        super()._reset_idx(env_ids)

        amp_observations = self.collect_reference_motions(num_samples)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)
        self.extras["amp_obs"][env_ids] = amp_observations.view(len(env_ids), -1)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
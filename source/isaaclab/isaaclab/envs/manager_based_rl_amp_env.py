# Add by FloatRIslet at 2025-07-01

import torch
import numpy as np
from gymnasium import spaces
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.leju_amp.motions.motion_loader import MotionLoader
from isaaclab.utils.math import quat_apply

class ManagerBasedRLAmpEnv(ManagerBasedRLEnv):
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motion_loader = MotionLoader(motion_file=self.cfg.amp.motion_file, device=self.device, 
                                           index=self.cfg.amp.index)
        self._amp_dim = self.cfg.amp.amp_observation_space
        self._amp_hist = self.cfg.observations.policy.history_length
        self.amp_observation_size = self._amp_hist * self._amp_dim
        self._incompatible_joint_names = []
        key_body_names = ["zarm_r6_link", "zarm_l6_link", "leg_r6_link", "leg_l6_link"]
        joint_names = [name for name in self.scene["robot"].data.joint_names if name not in self._incompatible_joint_names]
        self.lab_dof_index = self.scene["robot"].find_joints(joint_names)[0]
        
        self.ref_body_index = self.scene["robot"].data.body_names.index(self.cfg.amp.reference_body)
        self.key_body_indexes = [self.scene["robot"].data.body_names.index(name) for name in key_body_names]

        self.motion_dof_indexes = self._motion_loader.get_dof_index(joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.amp.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        self.amp_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self._amp_hist, self.cfg.amp.amp_observation_space), device=self.device)

        
    def _compute_amp_state(self, obs=None) -> torch.Tensor:
        obs = compute_obs(
            self.scene["robot"].data.joint_pos[:, self.lab_dof_index],
            self.scene["robot"].data.joint_vel[:, self.lab_dof_index],
            self.scene["robot"].data.body_pos_w[:, self.ref_body_index],
            self.scene["robot"].data.body_quat_w[:, self.ref_body_index],
            self.scene["robot"].data.body_lin_vel_w[:, self.ref_body_index],
            self.scene["robot"].data.body_ang_vel_w[:, self.ref_body_index],
            self.scene["robot"].data.body_pos_w[:, self.key_body_indexes],
        )
        return obs


    def step(self, action):
        obs, rew, term, trunc, extras = super().step(action)
        # update AMP observation history
        for i in reversed(range(self._amp_hist - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = self._compute_amp_state(obs['policy']).clone()
        # 扁平化存进 extras，供 collect_amp_observation 使用
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)
        return obs, rew, term, trunc, extras


    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        times = self._motion_loader.sample_times(num_samples)
        times = (np.expand_dims(times, axis=-1) 
                 - self._motion_loader.dt * np.arange(0, self._amp_hist)).flatten()
        (dof_positions,
         dof_velocities,
         body_positions,
         body_rotations,
         body_linear_velocities,
         body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
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
        return amp_observation.view(-1, self.amp_observation_size)

    
    def _reset_idx(self, env_ids):
        num_samples = env_ids.shape[0]
        super()._reset_idx(env_ids)

        amp_observations = self.collect_reference_motions(num_samples)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self._amp_hist, -1)


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
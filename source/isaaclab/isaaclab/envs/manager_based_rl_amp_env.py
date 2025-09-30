# Add by FloatRIslet at 2025-07-01

import torch
import numpy as np
from gymnasium import spaces
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.leju_amp.new_motions.motion_loader import MotionLoader
from isaaclab_tasks.manager_based.locomotion.velocity.config.leju_amp.new_motions.multi_motion_loader import MultiMotionLoader
from isaaclab.utils.math import quat_apply

class ManagerBasedRLAmpEnv(ManagerBasedRLEnv):
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._motion_loader = MotionLoader(motion_file=self.cfg.amp.motion_file, device=self.device, 
        #                                    index=self.cfg.amp.index)
        self._motion_loader = MultiMotionLoader(motion_lists=self.cfg.amp.motion_list, device=self.device)
        self._amp_dim = self.cfg.amp.amp_observation_space
        self._amp_hist = 5
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

        self.line_velocity = []
        
    def _compute_amp_state(self) -> torch.Tensor:
        obs = compute_obs(
            self.scene["robot"].data.joint_pos[:, self.lab_dof_index] - self.scene["robot"].data.default_joint_pos[:, self.lab_dof_index],
            self.scene["robot"].data.joint_vel[:, self.lab_dof_index] - self.scene["robot"].data.default_joint_vel[:, self.lab_dof_index],
            self.scene["robot"].data.body_pos_w[:, self.ref_body_index],
            self.scene["robot"].data.projected_gravity_b,
            self.scene["robot"].data.root_lin_vel_b,
            self.scene["robot"].data.root_ang_vel_b,
            self.scene["robot"].data.body_pos_w[:, self.key_body_indexes],
        )
        return obs


    def step(self, action):
        obs, rew, term, trunc, extras = super().step(action)
        # update AMP observation history
        for i in reversed(range(self._amp_hist - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = self._compute_amp_state().clone()
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)
        # self.line_velocity.append(self.scene["robot"].data.root_lin_vel_w[0].cpu().numpy())
        # print("velocity: ", self.scene["robot"].data.root_lin_vel_w)
        return obs, rew, term, trunc, extras


    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None, 
                                  clip_ids: torch.Tensor | None = None) -> torch.Tensor:
        if current_times is None:
            current_times, clip_ids = self._motion_loader.sample_times(num_samples)
        times = (np.expand_dims(current_times, axis=-1) 
                 - self._motion_loader.dt * np.arange(0, self._amp_hist)).flatten()
        clip_ids = clip_ids.repeat_interleave(self._amp_hist)
        (dof_positions, dof_velocities, body_positions, body_rotations,
         projected_gravity, body_linear_velocities, body_angular_velocities
        ) = self._motion_loader.sample(num_samples=num_samples, times=times, clip_ids=clip_ids)
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes], # 这里没有减去默认值，因为是0
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            projected_gravity,
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)

    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # if len(self.line_velocity) >= 500: draw(self.line_velocity)
        root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids)
        self.scene["robot"].write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.scene["robot"].write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_strategy_random(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times, clip_ids = self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            projected_gravity,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times, clip_ids=clip_ids)

        # get root transforms (the humanoid torso)
        motion_torso_index = self.motion_ref_body_index
        root_state = self.scene["robot"].data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        # root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = quat_apply(body_rotations[:, motion_torso_index], body_linear_velocities[:, motion_torso_index])
        root_state[:, 10:13] = quat_apply(body_rotations[:, motion_torso_index], body_angular_velocities[:, motion_torso_index])
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times, clip_ids)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self._amp_hist, -1)

        return root_state, dof_pos, dof_vel


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    projected_gravity: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            projected_gravity,
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs

def draw(line_velocity):
    import matplotlib.pyplot as plt
    line_velocity = np.array(line_velocity)
    plt.figure()
    plt.plot(line_velocity[:,0], label='x')
    # plt.plot(line_velocity[:,1], label='y')
    # plt.plot(line_velocity[:,2], label='z')
    plt.xlabel('Step')
    plt.ylabel('Velocity')
    plt.title('Root Linear Velocity over Time')
    plt.legend()
    plt.savefig('/home/yy/Coding/ghw/IsaacLab/imgs/flat_linear_velocity.png')
    print("Saved root linear velocity plot")
    quit()
from __future__ import annotations
import torch
from dataclasses import MISSING
from collections.abc import Sequence
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.mdp import UniformVelocityCommandCfg, UniformVelocityCommand
from isaaclab.utils import configclass

class UniformLevelVelocityCommand(UniformVelocityCommand):
    cfg: UniformLevelVelocityCommandCfg
    
    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
    
    def _resample(self, env_ids: Sequence[int]):
        if len(env_ids) != 0:
            # resample the time left before resampling
            flat_proportion = self.cfg.flat_proportion
            env_ids_flat = env_ids[env_ids < self._env.num_envs * flat_proportion]
            env_ids_else = env_ids[env_ids >= self._env.num_envs * flat_proportion]
            self.time_left[env_ids_flat] = self.time_left[env_ids_flat].uniform_(*self.cfg.flat_resampling_time_range)
            self.time_left[env_ids_else] = self.time_left[env_ids_else].uniform_(*self.cfg.resampling_time_range)
            # increment the command counter
            self.command_counter[env_ids] += 1
            # resample the command
            self._resample_command(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        flat_proportion = self.cfg.flat_proportion
        env_ids_flat = env_ids[env_ids < self._env.num_envs * flat_proportion]
        env_ids_else = env_ids[env_ids >= self._env.num_envs * flat_proportion]
        r_flat = torch.empty(len(env_ids_flat), device=self.device)
        r_else = torch.empty(len(env_ids_else), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids_flat, 0] = r_flat.uniform_(*self.cfg.flat_ranges.lin_vel_x)
        self.vel_command_b[env_ids_else, 0] = r_else.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids_flat, 1] = r_flat.uniform_(*self.cfg.flat_ranges.lin_vel_y)
        self.vel_command_b[env_ids_else, 1] = r_else.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids_flat, 2] = r_flat.uniform_(*self.cfg.flat_ranges.ang_vel_z)
        self.vel_command_b[env_ids_else, 2] = r_else.uniform_(*self.cfg.ranges.ang_vel_z)
        # update standing envs
        self.is_standing_env[env_ids_flat] = r_flat.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self.is_standing_env[env_ids_else] = r_else.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type = UniformLevelVelocityCommand

    flat_proportion: int = 0.1
    flat_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    flat_limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    flat_resampling_time_range: tuple[float, float] = MISSING




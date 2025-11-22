from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_joint_limit(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    joint_ids: list[int],
    asset_cfg: SceneEntityCfg,
    lower_limit_params: float,
    upper_limit_params: float,
):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    env_ids = torch.arange(int(env.scene.num_envs * 0.1), device=asset.device)

    # resolve joint indices
    if joint_ids is None:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)
    else:
        joint_ids = torch.tensor(joint_ids)
    # joint position limits
    joint_pos_limits = asset.data.default_joint_pos_limits.clone()
    joint_pos_limits[..., 0] = lower_limit_params
    joint_pos_limits[..., 1] = upper_limit_params

    # extract the position limits for the concerned joints
    joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
    if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
        raise ValueError(
            "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater than"
            " upper joint limits. Please check the distribution parameters for the joint position limits."
        )
    # set the position limits into the physics simulation
    asset.write_joint_position_limit_to_sim(
        joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
    )
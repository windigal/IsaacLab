from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(
        env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(
        env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def box_mass_levels(
        env: ManagerBasedRLEnv,
        env_ids,
        *,
        # 用哪个奖励项来判断“达标”，与你的 velocity 课程保持同风格
        judge_reward_term: str = "track_lin_vel_xy",
        # 达标阈值（相对该 reward 的权重）
        success_ratio: float = 0.6,
        # 区间上下限（绝对极限）
        mass_limits=(1.0, 5.0),
        # 初始可采样区间与推进步长
        init_range=(1.0, 2.0),
        step: float = 0.5,
):
    """
    每步都可被调用，但只在 episode 边界推进一次：
    - 若本局该奖励项的平均值 > weight * success_ratio，则把质量上界 += step（不超过 mass_limits[1]）
    - 可选将下界抬到上界的 90%（避免跨度过大）
    返回当前“质量上界”（张量，便于记录/plot）
    """
    # 状态放到 env.extras，和你其他 curriculum 保持一致
    st = env.extras.get("box_mass_curriculum", None)
    if st is None:
        st = {
            "range": [float(init_range[0]), float(init_range[1])],
            "limits": [float(mass_limits[0]), float(mass_limits[1])],
            "step": float(step),
        }
        env.extras["box_mass_curriculum"] = st

    lo, hi = st["range"]
    lo_lim, hi_lim = st["limits"]
    step = st["step"]

    # 取本局该奖励项的平均（和你 velocity 课程同套路）
    rew_cfg = env.reward_manager.get_term_cfg(judge_reward_term)
    # episode 累积和 / episode 时长（按需要可改为直接累积和阈值）
    ep_mean = torch.mean(
        env.reward_manager._episode_sums[judge_reward_term][env_ids]) / env.max_episode_length_s

    # 仅在 episode 边界推进（同 velocity 课程）
    if env.common_step_counter % env.max_episode_length == 0:
        if ep_mean > rew_cfg.weight * success_ratio:
            hi = min(hi + step, hi_lim)
            # 抬一下下界（可按需关掉：lo = max(lo, 0.9 * hi)）
            lo = min(max(lo, 0.9 * hi), hi)
            st["range"] = [float(lo), float(hi)]

    # 便于日志/可视化
    env.extras["box_mass_range"] = (float(lo), float(hi))
    return torch.tensor(hi, device=env.device)

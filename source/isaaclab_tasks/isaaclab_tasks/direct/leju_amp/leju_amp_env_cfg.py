# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import LejuKuavo42_AMP_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class LejuAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 10


    # spaces
    observation_space = 77
    action_space = 26
    state_space = 0
    num_amp_observations = 5
    amp_observation_space = 77

    early_termination = True
    termination_height = 0.3

    motion_file: str = MISSING
    index: int = MISSING
    reference_body = "base_link"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = LejuKuavo42_AMP_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class LejuAmpWalkEnvCfg(LejuAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "CMU02.pkl")
    index = 8

@configclass
class LejuAmpRunEnvCfg(LejuAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "CMU02.pkl")
    index = 6

@configclass
class LejuAmpJumpEnvCfg(LejuAmpEnvCfg):
    episode_length_s = 2.0
    motion_file = os.path.join(MOTIONS_DIR, "CMU13.pkl")
    index = 39
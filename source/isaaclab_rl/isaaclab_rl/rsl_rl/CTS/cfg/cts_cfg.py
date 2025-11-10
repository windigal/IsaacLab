# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This file defines the configuration classes for the Concurrent Teacher-Student (CTS)
# reinforcement learning algorithm, designed for use with Isaac Lab.

from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass

# 导入RSL-RL的基础配置，以便我们可以继承或引用它们
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg

#########################
# Policy configurations #
#########################

@configclass
class CtsRslRlPpoActorCriticCfg:
    """Configuration for the CTS PPO actor-critic networks."""

    class_name: str = "ActorCriticCTS"

    init_noise_std: float = 1.0
    activation: str = "elu"
    
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    prop_encoder_dims: list[int] = MISSING
    encoder_latent_dim: int = MISSING


############################
# Algorithm configurations #
############################

@configclass
class CtsRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPO_CTS"
    student_lr: float = MISSING
    reconstruction_loss_weight: float = MISSING


#########################
# Runner configurations #
#########################

@configclass
class CtsRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: CtsRslRlPpoActorCriticCfg = MISSING
    algorithm: CtsRslRlPpoAlgorithmCfg = MISSING

    teacher_ratio: float = MISSING
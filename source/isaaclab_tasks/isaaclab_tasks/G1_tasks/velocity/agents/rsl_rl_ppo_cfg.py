# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.CTS.cfg.cts_cfg import CtsRslRlOnPolicyRunnerCfg, CtsRslRlPpoActorCriticCfg, CtsRslRlPpoAlgorithmCfg

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class CTSPPORunnerCfg(CtsRslRlOnPolicyRunnerCfg):
    """
    Configuration for the Unitree Go2 agent using the CTS-PPO algorithm.
    """
    num_steps_per_env=24
    max_iterations=50000
    save_interval=500
    experiment_name=""
    empirical_normalization = False
    teacher_ratio=0.75
    policy = CtsRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        activation="elu",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[512, 256],
        prop_encoder_dims=[512, 256],
        encoder_latent_dim=32,
    )
    algorithm = CtsRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        student_lr=1.0e-3,
        reconstruction_loss_weight=1.0,
    )
    

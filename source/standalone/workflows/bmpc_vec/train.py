# Add by Windigal in 2024.11
# SPDX-License-Identifier: BSD-3-Clause
"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video",
                    action="store_true",
                    default=False,
                    help="Record videos during training.")
parser.add_argument("--video_length",
                    type=int,
                    default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval",
                    type=int,
                    default=2000,
                    help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs",
                    type=int,
                    default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations",
                    type=int,
                    default=None,
                    help="RL Policy training iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import os
os.environ['MUJOCO_GL'] = 'off'  # 'osmesa'/'egl'/'off'
os.environ['LAZY_LEGACY_OP'] = '0'

import torch
import hydra
import numpy as np
import gymnasium as gym
from termcolor import colored

sys.path.append("/root")
from bmpcvec.bmpc import BMPC
from bmpcvec.common.buffer import Buffer
from bmpcvec.common.seed import set_seed
from bmpcvec.common.logger import TBLogger
from bmpcvec.common.parser import parse_cfg, save_cfg
from bmpcvec.envs.wrappers.pixels import PixelWrapper
from bmpcvec.envs.wrappers.tensor import TensorWrapper
from bmpcvec.trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True
from omni.isaac.lab_tasks.utils.wrappers.bmpcvec import BMPCEnvWrapper
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry


@hydra.main(config_name='config', config_path='.')
def train(agent_cfg):
    """
    Script for training BMPC agents.
    """
    assert torch.cuda.is_available()
    assert agent_cfg.steps > 0, 'Must train for at least 1 step.'
    agent_cfg = parse_cfg(agent_cfg)
    set_seed(agent_cfg.seed)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.seed = agent_cfg.seed
    env_cfg.num_envs = args_cli.num_envs
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = BMPCEnvWrapper(env)
    env = TensorWrapper(env)
    if agent_cfg.get('obs', 'state') == 'rgb':
        env = PixelWrapper(agent_cfg, env)
    try:  # Dict
        agent_cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:  # Box
        agent_cfg.obs_shape = {agent_cfg.get('obs', 'state'): env.observation_space.shape}

    agent_cfg.action_dim = env.action_space.shape[0]  # type: ignore
    agent_cfg.episode_length = 1000
    agent_cfg.seed_steps = max(1000, 1 * agent_cfg.episode_length) if agent_cfg.debug else max(
        1000, 5 * agent_cfg.episode_length)

    print('pid:', os.getpid())
    print(colored('Work dir:', 'yellow', attrs=['bold']), agent_cfg.work_dir, flush=True)

    trainer_cls = OnlineTrainer
    logger_cls = TBLogger
    trainer = trainer_cls(
        cfg=agent_cfg,
        env=env,
        agent=BMPC(agent_cfg),
        buffer=Buffer(agent_cfg),
        logger=logger_cls(agent_cfg),
    )
    save_cfg(agent_cfg, agent_cfg.work_dir)  # save parsed config, must after the logger's init
    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
    train()
    simulation_app.close()

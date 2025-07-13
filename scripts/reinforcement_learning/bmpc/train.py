# Add by Windigal in 2024.11
# SPDX-License-Identifier: BSD-3-Clause
"""Script to train RL agent with BMPC."""

"""Launch Isaac Sim Simulator first."""
import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with BMPC.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")

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


import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored
import gymnasium as gym
from common.parser import parse_cfg, save_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from bmpc import BMPC
from envs.wrappers.tensor import TensorWrapper
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger, TBLogger
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.bmpc import BMPCEnvWrapper
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(agent_cfg: dict):
	"""
	Script for training single-task / multi-task agents.
	"""
	assert torch.cuda.is_available()
	assert agent_cfg.steps > 0, 'Must train for at least 1 step.'
	agent_cfg.task = args_cli.task
	agent_cfg = parse_cfg(agent_cfg)
	set_seed(agent_cfg.seed)
	env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
	
	env_cfg.seed = agent_cfg.seed
	env_cfg.scene.num_envs = args_cli.num_envs
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
	env = BMPCEnvWrapper(env, agent_cfg)
	env = TensorWrapper(env)
	try:  # Dict
		agent_cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except:  # Box
	    agent_cfg.obs_shape = {agent_cfg.get('obs', 'state'): env.observation_space.shape}

	agent_cfg.action_dim = env.action_space.shape[0]  # type: ignore
	agent_cfg.episode_length = 2000
	agent_cfg.seed_steps = 2000
	# agent_cfg.seed_steps = max(1000, 5 * agent_cfg.episode_length)
	print('pid:', os.getpid(), flush=True)
	print(colored('Work dir:', 'yellow', attrs=['bold']), agent_cfg.work_dir)

	logger_cls = TBLogger if agent_cfg.use_tensorboard else Logger
	trainer = OnlineTrainer(
		cfg=agent_cfg,
		env=env,
		agent=BMPC(agent_cfg),
		buffer=Buffer(agent_cfg),
		logger=logger_cls(agent_cfg),
	)
	save_cfg(agent_cfg, agent_cfg.work_dir) # save parsed config, must do it after the Logger's init
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()

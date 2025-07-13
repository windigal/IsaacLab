# Add by Windigal in 2025.06
# SPDX-License-Identifier: BSD-3-Clause
"""Script to play a checkpoint if an RL agent from BMPC."""

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint if an RL agent from BMPC.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
import gymnasium
from envs.wrappers.tensor import TensorWrapper
from isaaclab_rl.bmpc_vec import BMPCEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from bmpc import BMPC
torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(agent_cfg: dict):
	"""
	Script for evaluating a single-task checkpoint.
	"""
	assert torch.cuda.is_available()
	agent_cfg.task = args_cli.task
	agent_cfg = parse_cfg(agent_cfg)
	print('pid:', os.getpid(), flush=True)
	set_seed(agent_cfg.seed)
	print(colored(f'Task: {agent_cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {agent_cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {agent_cfg.checkpoint}', 'blue', attrs=['bold']))

	# Make environment
	env_agent_cfg = load_cfg_from_registry(args_cli.task, "env_agent_cfg_entry_point")
	env = gymnasium.make(args_cli.task, agent_cfg=env_agent_cfg, render_mode="rgb_array" if args_cli.video else None)
	env = BMPCEnvWrapper(env, agent_cfg)
	env = TensorWrapper(env)

	# Load agent
	agent = BMPC(agent_cfg)
	assert os.path.exists(agent_cfg.checkpoint), f'Checkpoint {agent_cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(agent_cfg.checkpoint)
	
	# Evaluate
	print(colored(f'Evaluating agent on {agent_cfg.task}:', 'yellow', attrs=['bold']))
	if agent_cfg.save_video:
		video_dir = os.path.join(agent_cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	tasks = agent_cfg.tasks if agent_cfg.multitask else [agent_cfg.task]
	for task_idx, task in enumerate(tasks):
		if not agent_cfg.multitask:
			task_idx = None
		obs, t = env.reset(task_idx=task_idx), 0
		while True:
			with torch.inference_mode():
				action, _ = agent.act(obs, t0=t==0, task=task_idx)
				obs, _, done, _ = env.step(action)
				t += 1
				if done:
					t = 0


if __name__ == '__main__':
	evaluate()
	simulation_app.close()
from copy import deepcopy
import gymnasium
from gymnasium.vector import AsyncVectorEnv
import numpy as np


class Vectorized():
	"""
	Vectorized environment for TD-MPC2 online training.
	"""

	def __init__(self, args_cli, env_cfg):
		super().__init__()
		self.cfg = env_cfg

		def make():
			_cfg = deepcopy(env_cfg)
			_cfg.num_envs = 1
			_cfg.seed = env_cfg.seed + np.random.randint(1000)
			return gymnasium.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

		print(f'Creating {args_cli.num_envs} environments...')
		self.env = AsyncVectorEnv([make for _ in range(args_cli.num_envs)])

	def reset(self):
		obs, _ = self.env.reset()
		return obs

	def step(self, action):
		return self.env.step(action)

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate an agent."""
		ep_rewards = []
		for i in range(self.cfg.eval_episodes // self.cfg.num_envs):
			obs, dones, ep_reward, t = self.env.reset(), torch.full((self.cfg.num_envs, ), False), 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not dones.all():
				torch.compiler.cudagraph_mark_step_begin()
				action, _ = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += (reward * ~dones)
				dones = dones | done
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			assert dones.all(), 'Vectorized environments must reset all environments at once.'
			ep_rewards.append(ep_reward)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None, act_info=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if act_info is None:
			act_info = TensorDict({})
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))   
		expert_value = act_info.get("action_value", torch.tensor(float('nan'))).squeeze().unsqueeze(0)
		expert_action_dist = act_info.get("action_dist", \
	  		torch.full(size=(1, 2*self.cfg.action_dim), fill_value=float('nan')))
		last_reanalyze = act_info.get("last_reanalyze", torch.zeros((1,)))
		td = TensorDict(dict(
			obs=obs, 
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0) if reward.shape == torch.Size([]) else reward,
			terminated=terminated.unsqueeze(0),
			expert_value=expert_value,
			expert_action_dist=expert_action_dist,
			last_reanalyze=last_reanalyze,
		), batch_size=(1,))
		return td


	def train(self):
		"""Train an agent."""
		train_metrics, done, eval_next = {}, torch.tensor(True), False
		obs = self.env.reset()
		self._tds = [[self.to_td(obs[i])] for i in range(self.cfg.num_envs)]
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			# if self._step % self.cfg.eval_freq == 0:
			# 	eval_next = True

			# Collect experience
			if self._step > self.cfg.seed_steps:
				torch.compiler.cudagraph_mark_step_begin()
				action, act_info = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = torch.rand((self.cfg.num_envs, *self.env.action_space.shape)) * 40 - 20
				# print(f"inbmpc: {action=}")
				act_info = None
			obs, reward, done, info = self.env.step(action) # done是true的时候obs是自动reset的观测
			# Reset environment
			if done.any():
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					if self.agent.cfg.eval_pi: # evaluate policy prior
						mpc = self.agent.cfg.mpc
						self.agent.cfg.mpc = False
						eval_metrics = self.eval()
						eval_metrics.update(self.common_metrics())
						self.logger.log(eval_metrics, "eval_pi")
						self.agent.cfg.mpc = mpc			
					eval_next = False

				if self._step > 0:
					for i in np.nonzero(done)[0]:
						tds = torch.cat(self._tds[i])
						train_metrics.update(
							episode_reward=tds['reward'].nansum(0).mean(),
							episode_length=tds['reward'].shape[0],
						)
						train_metrics.update(self.common_metrics())
						self.logger.log(train_metrics, 'train')
						self._ep_idx = self.buffer.add(tds)
						self._tds[i] = [self.to_td(obs[i])]

			for i in range(self.cfg.num_envs):
				if i not in np.nonzero(done)[0]:
					self._tds[i].append(self.to_td(obs[i], action[i], reward[i], 
								   info[i]['terminated'], act_info[i] if act_info is not None else None))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					pretrain = True
					print('Pretraining agent on seed data...')
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
					pretrain = False
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer, pretrain)
				train_metrics.update(_train_metrics)

			self._step += self.cfg.num_envs
			if self._step % self.cfg.save_freq == 0: # test
				self.agent.save(self.cfg.work_dir / "models" / f'{"model" + str(self._step)}.pt')
		self.logger.finish(self.agent)

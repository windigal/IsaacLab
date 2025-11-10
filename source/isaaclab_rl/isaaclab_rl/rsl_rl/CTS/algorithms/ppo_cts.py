# Concurrent Teacher-Student (CTS) framework

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain

from rsl_rl.storage import RolloutStorage
from isaaclab_rl.rsl_rl.CTS.modules import ActorCriticCTS

class PPO_CTS:
    """
    A standalone, fully-featured Proximal Policy Optimization algorithm for 
    Concurrent Teacher-Student (CTS) learning.
    """
    policy: ActorCriticCTS

    def __init__(
        self,
        policy: ActorCriticCTS,
        device='cpu',
        num_envs: int = 4096,
        # PPO Standard Parameters
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        normalize_advantage_per_mini_batch=False,
        # CTS Specific Parameters
        teacher_ratio=0.75,
        student_lr=1.0e-3,
        reconstruction_loss_weight=2.0,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs
    ):
        if kwargs:
            print(f"PPO_CTS.__init__ got unexpected arguments which will be ignored: {list(kwargs.keys())}")
            
        # -- Device and multi-GPU setup --
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank, self.gpu_world_size = 0, 1

        # -- Policy --
        self.policy = policy
        self.policy.to(self.device)
        
        # -- Storage & Env Info --
        self.num_envs = num_envs
        self.teacher_ratio = teacher_ratio
        self.num_teachers = int(self.num_envs * self.teacher_ratio)
        self.storage: RolloutStorage = None 
        self.transition = RolloutStorage.Transition()

        # -- PPO parameters --
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.learning_rate = learning_rate
        self.student_lr = student_lr
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.reconstruction_loss_weight = reconstruction_loss_weight

        # -- Optimizers --
        teacher_params = [
            {'params': self.policy.actor.parameters()},
            {'params': self.policy.critic.parameters()},
            {'params': self.policy.privileged_encoder.parameters()},
            {'params': self.policy.std},
        ]
        self.rl_optimizer = optim.Adam(teacher_params, lr=self.learning_rate)
        
        self.student_enc_optimizer = optim.Adam(self.policy.proprioceptive_encoder.parameters(), lr=self.student_lr)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        """Initializes the rollout storage."""
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, 
            actor_obs_shape, critic_obs_shape, actions_shape, 
            device=self.device,
        )

    def compute_returns(self, last_critic_obs):
        """Compute returns and advantages for the collected rollout."""
        last_values = self.policy.evaluate(last_critic_obs.detach()).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, 
            normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        """Main update function to perform learning on the collected rollouts."""
        mean_value_loss, mean_surrogate_loss, mean_entropy, mean_latent_loss = 0.0, 0.0, 0.0, 0.0

        # --- Loop 1: RL Update ---
        generator = self._rl_mini_batch_generator()
        teacher_samples = self.num_teachers * self.storage.num_transitions_per_env // self.num_mini_batches
        student_samples = (self.num_envs - self.num_teachers) * self.storage.num_transitions_per_env // self.num_mini_batches
        for batch in generator:
            obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, _, _, _ = batch
            aug_obs_batch = obs_batch.detach()

            def get_results(start: int, end: int, is_teacher: bool):
                self.policy.act(aug_obs_batch[start:end], critic_obs_batch[start:end], is_teacher)
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch[start:end])
                value_batch = self.policy.evaluate(
                    aug_obs_batch[start:end], critic_obs_batch[start:end], is_teacher
                )
                mu_batch = self.policy.action_mean
                sigma_batch = self.policy.action_std
                entropy_batch = self.policy.entropy
                return actions_log_prob_batch, value_batch, mu_batch, sigma_batch, entropy_batch
            
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            
            teacher_results = get_results(0, teacher_samples, True)
            student_results = get_results(teacher_samples, teacher_samples + student_samples, False)
            results = []
            for x1, x2 in zip(teacher_results, student_results):
                results.append(torch.cat([x1, x2], dim=0))

            actions_log_prob_batch, value_batch, mu_batch, sigma_batch, entropy_batch = results
            # KL divergence for adaptive learning rate
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5) 
                        + (torch.square(old_sigma_batch) 
                           + torch.square(old_mu_batch - mu_batch)) 
                        / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        kl_tensor = torch.tensor(kl_mean, device=self.device)
                        torch.distributed.all_reduce(kl_tensor, op=torch.distributed.ReduceOp.SUM)
                        kl_mean = kl_tensor.item() / self.gpu_world_size
                    
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.rl_optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

                
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped)
            teacher_surrogate_loss = surrogate_loss[:teacher_samples].mean()
            student_surrogate_loss = surrogate_loss[teacher_samples:].mean()
            surrogate_loss = teacher_surrogate_loss + student_surrogate_loss
            
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            self.rl_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu: 
                self.reduce_parameters(self.rl_optimizer)
            nn.utils.clip_grad_norm_(chain.from_iterable(g['params'] for g in self.rl_optimizer.param_groups), self.max_grad_norm)
            self.rl_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        # --- Loop 2: Student Encoder Update ---
        student_sl_generator = self._sl_mini_batch_generator()
        for batch in student_sl_generator:
            obs_batch, critic_obs_batch = batch
            with torch.no_grad(): 
                target_latents = self.policy.get_teacher_latent(critic_obs_batch)
            predicted_latents = self.policy.proprioceptive_encoder(obs_batch)
            latent_loss = F.mse_loss(predicted_latents, target_latents) * self.reconstruction_loss_weight

            self.student_enc_optimizer.zero_grad()
            latent_loss.backward()
            if self.is_multi_gpu: 
                self.reduce_parameters(self.student_enc_optimizer)
            nn.utils.clip_grad_norm_(self.policy.proprioceptive_encoder.parameters(), self.max_grad_norm)
            self.student_enc_optimizer.step()
            mean_latent_loss += latent_loss.item()
        
        # --- Logging Normalization ---
        num_teacher_batches = self.num_learning_epochs * self.num_mini_batches if self.num_teachers > 0 else 0
        num_student_batches = self.num_learning_epochs * self.num_mini_batches if (self.num_envs - self.num_teachers) > 0 else 0
        num_total_rl_updates = num_teacher_batches + num_student_batches

        if num_total_rl_updates > 0:
            mean_value_loss /= num_total_rl_updates
            mean_surrogate_loss /= num_total_rl_updates
            mean_entropy /= num_total_rl_updates
        if num_student_batches > 0:
            mean_latent_loss /= num_student_batches

            
        self.storage.clear()
        
        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "latent_loss": mean_latent_loss,
        }

    def _rl_mini_batch_generator(self):
        teacher_samples_num = self.num_teachers * self.storage.num_transitions_per_env
        student_samples_num = (self.num_envs - self.num_teachers) * self.storage.num_transitions_per_env
        teacher_mini_batch_size = teacher_samples_num // self.num_mini_batches
        student_mini_batch_size = student_samples_num // self.num_mini_batches
        teacher_indices = torch.randperm(teacher_samples_num, requires_grad=False, device=self.device)
        student_indices = teacher_samples_num + torch.randperm(student_samples_num, requires_grad=False, device=self.device)
        
        obs = self.storage.observations.flatten(0, 1)
        critic_obs = self.storage.privileged_observations.flatten(0, 1) 
        actions = self.storage.actions.flatten(0, 1)
        values = self.storage.values.flatten(0, 1)
        advantages = self.storage.advantages.flatten(0, 1)
        returns = self.storage.returns.flatten(0, 1)
        old_actions_log_prob = self.storage.actions_log_prob.flatten(0, 1)
        old_mu = self.storage.mu.flatten(0, 1)
        old_sigma = self.storage.sigma.flatten(0, 1)

        def get_teacher_student_samples(data, slice):
            (i1, i2), (j1, j2) = slice
            return torch.cat([data[teacher_indices[i1:i2]], data[student_indices[j1:j2]]], 0).detach()
        # import pdb; pdb.set_trace()
        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                slice = (
                    (i * teacher_mini_batch_size, (i+1) * teacher_mini_batch_size),
                    (i * student_mini_batch_size, (i+1) * student_mini_batch_size),
                )

                obs_batch = get_teacher_student_samples(obs, slice)
                critic_observations_batch = get_teacher_student_samples(critic_obs, slice)
                actions_batch = get_teacher_student_samples(actions, slice)
                values_batch = get_teacher_student_samples(values, slice)
                advantages_batch = get_teacher_student_samples(advantages, slice)
                returns_batch = get_teacher_student_samples(returns, slice)
                old_actions_log_prob_batch = get_teacher_student_samples(old_actions_log_prob, slice)
                old_mu_batch = get_teacher_student_samples(old_mu, slice)
                old_sigma_batch = get_teacher_student_samples(old_sigma, slice)

                yield obs_batch, critic_observations_batch, actions_batch, values_batch, \
                      advantages_batch, returns_batch, old_actions_log_prob_batch, \
                      old_mu_batch, old_sigma_batch, None, None, None

    def _sl_mini_batch_generator(self):
        num_student_envs = self.num_envs - self.num_teachers

        batch_size = num_student_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(batch_size, device=self.device)

        obs = self.storage.observations[:, self.num_teachers:].flatten(0, 1)
        critic_obs = self.storage.privileged_observations[:, self.num_teachers:].flatten(0, 1) # type: ignore

        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                start, end = i * mini_batch_size, (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield obs[batch_idx], critic_obs[batch_idx]

    def broadcast_parameters(self):
        if not self.is_multi_gpu: return
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self, optimizer):
        if not self.is_multi_gpu: return
        
        grads = [param.grad.view(-1) for group in optimizer.param_groups for param in group['params'] if param.grad is not None]
        if not grads: return
        
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    numel = param.numel()
                    param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                    offset += numel
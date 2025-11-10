import torch
import time
import os
from collections import deque
import statistics
import rsl_rl

from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state
from rsl_rl.modules import EmpiricalNormalization
from isaaclab_rl.rsl_rl.CTS.modules import ActorCriticCTS
from isaaclab_rl.rsl_rl.CTS.algorithms import PPO_CTS


class OnPolicyRunnerCTS:
    """
    为CTS算法定制的功能完备的OnPolicyRunner。
    
    此版本的功能和接口与 rsl_rl.runners.OnPolicyRunner 完全对齐，
    包括日志、模型保存/加载、多GPU支持等，并加入了CTS专属逻辑。
    """
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str, device='cpu'):
        self.cfg = train_cfg
        self.alg_cfg = self.cfg["algorithm"]
        self.policy_cfg = self.cfg["policy"]
        self.device = device
        self.env = env

        # Multi-GPU configuration
        self._configure_multi_gpu()

        # --- CTS 特定参数解析 ---
        self.teacher_ratio = self.cfg["teacher_ratio"]
        self.num_teachers = int(self.env.num_envs * self.teacher_ratio)
        self.teacher_indices = torch.arange(self.num_teachers, device=self.device, dtype=torch.long)
        self.student_indices = torch.arange(self.num_teachers, self.env.num_envs, device=self.device, dtype=torch.long)
        print(f"--- OnPolicyRunnerCTS: {len(self.teacher_indices)} Teachers, {len(self.student_indices)} Students ---")
        
        # --- 获取维度信息 ---
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        self.privileged_obs_type = "critic"
        num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        
        # --- 实例化策略网络 ---
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCriticCTS = policy_class(
            num_proprio_obs=num_obs,
            num_priv_obs=num_privileged_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        
        # --- 实例化算法 ---
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO_CTS = alg_class(policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, 
                                      num_envs=self.env.num_envs, teacher_ratio=self.teacher_ratio, **self.alg_cfg)

        # --- 经验归一化 ---
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)
        
        self.alg.init_storage(
            training_type="rl",
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            critic_obs_shape=[num_privileged_obs],
            actions_shape=[self.env.num_actions]
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        self.shuffle_indices = torch.tensor([i for i in range(self.env.num_envs) if i % 4 != 0] + 
                                            [i for i in range(self.env.num_envs) if i % 4 == 0])


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        obs = self.obs_normalizer(obs)
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)

        obs = self._shuffle_or_recover(obs, 'shuffle')
        privileged_obs = self._shuffle_or_recover(privileged_obs, 'shuffle')

        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device)
                    actions_log_prob = torch.zeros(self.env.num_envs, device=self.device)
                    action_mean = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device)
                    action_sigma = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device)
                    values = torch.zeros(self.env.num_envs, 1, device=self.device)

                    actions_teacher = self.alg.policy.act(obs[self.teacher_indices], 
                                                          privileged_observations=privileged_obs[self.teacher_indices], 
                                                          is_teacher=True).detach()
                    actions[self.teacher_indices] = actions_teacher
                    actions_log_prob[self.teacher_indices] = self.alg.policy.get_actions_log_prob(actions_teacher).detach()
                    action_mean[self.teacher_indices] = self.alg.policy.action_mean.detach()
                    action_sigma[self.teacher_indices] = self.alg.policy.action_std.detach()
                    values[self.teacher_indices] = self.alg.policy.evaluate(obs[self.teacher_indices], 
                                                                            privileged_obs[self.teacher_indices],
                                                                            is_teacher=True).detach()

                    actions_student = self.alg.policy.act(obs[self.student_indices], is_teacher=False).detach()
                    actions[self.student_indices] = actions_student
                    actions_log_prob[self.student_indices] = self.alg.policy.get_actions_log_prob(actions_student).detach()
                    action_mean[self.student_indices] = self.alg.policy.action_mean.detach()
                    action_sigma[self.student_indices] = self.alg.policy.action_std.detach()
                    values[self.student_indices] = self.alg.policy.evaluate(obs[self.student_indices], 
                                                        privileged_obs[self.student_indices],
                                                        is_teacher=False).detach()
                    
                    env_actions = self._shuffle_or_recover(actions, "recover")
                    next_obs_raw, rewards_raw, dones_raw, infos = self.env.step(env_actions)

                    rewards = self._shuffle_or_recover(rewards_raw.to(self.device), "shuffle")
                    dones = self._shuffle_or_recover(dones_raw.to(self.device), "shuffle")

                    next_obs = self.obs_normalizer(next_obs_raw.to(self.device))
                    next_privileged_obs = self.privileged_obs_normalizer(infos["observations"][self.privileged_obs_type].to(self.device))

                    transition = self.alg.storage.Transition()
                    transition.observations = obs
                    transition.privileged_observations = privileged_obs
                    transition.actions = actions
                    transition.rewards = rewards.clone()
                    transition.dones = dones
                    transition.values = values
                    transition.actions_log_prob = actions_log_prob
                    transition.action_mean = action_mean
                    transition.action_sigma = action_sigma

                    
                    self.alg.storage.add_transitions(transition)
                    obs = self._shuffle_or_recover(next_obs, 'shuffle')
                    privileged_obs = self._shuffle_or_recover(next_privileged_obs, 'shuffle')

                    rewards = self._shuffle_or_recover(rewards.to(self.device), "recover")
                    dones = self._shuffle_or_recover(dones.to(self.device), "recover")

                    if self.log_dir is not None and not self.disable_logs:
                        if "episode" in infos: 
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards_raw
                        cur_episode_length += 1
                        new_ids = (dones_raw > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                last_critic_obs = torch.clone(privileged_obs)
                last_values = torch.concat([
                    self.alg.policy.evaluate(obs[:self.num_teachers], last_critic_obs[:self.num_teachers], True).detach(),
                    self.alg.policy.evaluate(obs[self.num_teachers:], last_critic_obs[self.num_teachers:], False).detach(),
                ], dim=0)
                self.alg.storage.compute_returns(last_values, self.alg.gamma, self.alg.lam, 
                                                 normalize_advantage=not self.alg.normalize_advantage_per_mini_batch)
            loss_dict = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            
            ep_infos.clear()

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    if self.writer is not None: self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    if self.writer is not None: self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        if self.writer is not None:
            # -- Losses
            for key, value in locs["loss_dict"].items():
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
            self.writer.add_scalar("Loss/student_lr", self.alg.student_lr, locs["it"])

            # -- Policy
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

            # -- Performance
            self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
            self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

            # -- Training
            if len(locs["rewbuffer"]) > 0:
                self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)


        str_title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses (终端打印)
            for key, value in locs["loss_dict"].items():
                    log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            
            # -- Rewards
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['tot_iter'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            # "optimizer_state_dict": self.alg.optimizer.state_dict(),
            # "student_optimizer_state_dict": self.alg.student_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # if load_optimizer:
        #     self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        #     self.alg.student_optimizer.load_state_dict(loaded_dict["student_optimizer_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        print(f"Loaded model from {path}, resuming at iteration {self.current_learning_iteration}")
        return loaded_dict["infos"]
    
    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.policy.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        self.alg.policy.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self):
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        torch.cuda.set_device(self.gpu_local_rank)


    def _shuffle_or_recover(self, tensor, mode='shuffle'):
        """
        对tensor进行打乱或复原操作
        
        Args:
            tensor: 输入tensor，第0维为4096
            mode: 'shuffle' 或 'recover'
            seed: 随机种子，仅在shuffle_indices为None时使用
        
        Returns:
            processed_tensor: 处理后的tensor
            shuffle_indices: 使用的打乱索引（便于后续使用）
        """
        assert mode in ['shuffle', 'recover'], "mode必须是'shuffle'或'recover'"
        assert tensor.shape[0] == self.env.num_envs, f"tensor第0维必须和环境数量{self.env.num_envs}相等，当前是{tensor.shape[0]}"
        # return tensor
        # 如果未提供索引且需要打乱，则生成新的索引
        
        # 根据模式执行操作
        if mode == 'shuffle':
            processed_tensor = tensor[self.shuffle_indices]
        else:  # mode == 'recover'
            recovery_indices = torch.argsort(self.shuffle_indices)
            processed_tensor = tensor[recovery_indices]
        
        return processed_tensor
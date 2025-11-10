import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation


def l2normalize(x: torch.Tensor, dim: int):
    l2norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
    x = x / torch.maximum(l2norm, torch.tensor(1e-8))  # prevent zero vector
    return x

class L2Norm(nn.Module):
    
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return l2normalize(x, dim=-1)

class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self):
		super().__init__()
		self.dim = 4

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"
     
    
class ActorCriticCTS(nn.Module):
    def __init__(self,
                 num_proprio_obs,
                 num_priv_obs,
                 num_actions,
                 priv_encoder_dims=[512, 256],
                 prop_encoder_dims=[512, 256],
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 encoder_latent_dim=32,
                 activation='elu',
                 init_noise_std=1.0,
                 noise_std_type: str = "scalar",
                 **kwargs):
        super().__init__()
        
        if kwargs:
            print(f"ActorCriticCTS.__init__ got unexpected arguments: {list(kwargs.keys())}")

        print("--- Initializing ActorCriticCTS ---")

        activation_fn = resolve_nn_activation(activation)
        self.num_actions = num_actions
        self.is_recurrent = False

        # Privileged Encoder (Teacher)
        priv_layers = []
        priv_layers.append(nn.Linear(num_priv_obs, priv_encoder_dims[0]))
        priv_layers.append(activation_fn)
        for i in range(len(priv_encoder_dims) - 1):
            priv_layers.append(nn.Linear(priv_encoder_dims[i], priv_encoder_dims[i+1]))
            priv_layers.append(activation_fn)
        priv_layers.append(nn.Linear(priv_encoder_dims[-1], encoder_latent_dim))
        priv_layers.append(L2Norm())
        self.privileged_encoder = nn.Sequential(*priv_layers)

        # Proprioceptive Encoder (Student)
        prop_layers = []
        prop_layers.append(nn.Linear(num_proprio_obs, prop_encoder_dims[0]))
        prop_layers.append(activation_fn)
        for i in range(len(prop_encoder_dims) - 1):
            prop_layers.append(nn.Linear(prop_encoder_dims[i], prop_encoder_dims[i+1]))
            prop_layers.append(activation_fn)
        prop_layers.append(nn.Linear(prop_encoder_dims[-1], encoder_latent_dim))
        prop_layers.append(L2Norm())
        self.proprioceptive_encoder = nn.Sequential(*prop_layers)

        # Shared Actor Network
        actor_input_dim = 45 + encoder_latent_dim
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Shared Critic Network
        critic_input_dim = num_priv_obs + encoder_latent_dim
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(activation_fn)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Privileged Encoder: {self.privileged_encoder}")
        print(f"Proprioceptive Encoder: {self.proprioceptive_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        # 禁用默认参数验证以提速
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, input):
        # compute mean
        mean = self.actor(input)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)        
    
    def act(self, observations, privileged_observations=None, is_teacher=True):
        if is_teacher:
            if privileged_observations is None:
                raise ValueError("privileged_observations must be provided for the teacher.")
            latent = self.get_teacher_latent(privileged_observations)
        else:
            latent = self.get_student_latent(observations).detach()
        
        # L2-normalize the latent vector
        observation = self._single_frame(observations)
        actor_input = torch.cat((observation, latent), dim=-1)
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def act_inference(self, observations):
        latent = self.get_student_latent(observations)
        observation = self._single_frame(observations)
        actor_input = torch.cat((observation, latent), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def evaluate(self, obs, privileged_observations, is_teacher=False):
        if is_teacher:
            if privileged_observations is None:
                raise ValueError("privileged_observations must be provided for the teacher.")
            latent = self.get_teacher_latent(privileged_observations)
        else:
            latent = self.get_student_latent(obs)
             
        critic_input = torch.cat((privileged_observations, latent), dim=-1)
        value = self.critic(critic_input)
        return value
    
    def get_student_latent(self, proprioceptive_observations):
        return self.proprioceptive_encoder(proprioceptive_observations)

    def get_teacher_latent(self, privileged_observations):
        return self.privileged_encoder(privileged_observations)
    
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
    
    def _single_frame(self, observations):
        assert observations.shape[1] % 45 == 0, \
            "observations must be divided by 45, which indicates multi-frame of num_proprio_obs"
        
        frames = observations.shape[1] // 45
        terms_shape = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        single_frame_term = [observations[:, (sum(terms_shape[:i+1]) * frames - terms_shape[i]):sum(terms_shape[:i+1]) * frames] 
                             for i in range(len(terms_shape))]
        return torch.concat(single_frame_term, dim=1)
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

from util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
epsilon = 1e-6

class SquashedGaussianPolicy(nn.Module):
    """Squashed Gaussian Actor, which maps the given obs to a parameterized Gaussian Distribution,
    followed by a Tanh transformation to squash the action sample to [-1, 1]. """
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, conditioned_logstd=True):
        super().__init__()
        self.conditioned_logstd = conditioned_logstd
        if self.conditioned_logstd is True:
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), 2 * act_dim])
        else:
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
            self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        out = self.net(obs)
        if self.conditioned_logstd is True:
            mean, self.log_std = torch.split(out, out.shape[-1] // 2, dim=-1)
        else:
            mean = out
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    def sample(self, obs, deterministic=False):
        """For training and evaluation."""
        dist = self(obs)
        if deterministic:
            raw_action, log_prob = dist.mean, None
            action = torch.tanh(raw_action)
        else:
            raw_action = dist.rsample()
            log_prob = dist.log_prob(raw_action)
            action = torch.tanh(raw_action)
            # Enforcing Action Bound
            log_prob -= torch.log((1 - action.pow(2)) + epsilon)
            log_prob = log_prob.sum(-1)
        return action, log_prob
    
    def act(self, obs, deterministic=False, enable_grad=False):
        """For training and evaluation."""
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            if deterministic:
                action = torch.tanh(dist.mean)
            else:
                action = torch.tanh(dist.rsample())
            return action

    def evaluate(self, obs, action):
        """For Behavior Cloning."""
        dist = self(obs)
        action = torch.clip(action, -1.0, 1.0)
        raw_action = 0.5 * (action.log1p() - (-action).log1p())
        # Enforcing Action Bound
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log((1 - action.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1)
        return log_prob

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, use_tanh="False"):
        super().__init__()
        self.use_tanh = use_tanh
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            action = dist.mean if deterministic else dist.rsample()
            action = torch.tanh(action) if self.use_tanh else torch.clip(action, min=-1.0, max=1.0)
            return action


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            action = self(obs)
            action = torch.clip(action, min=-1.0, max=1.0)
            return action
"""
models/networks.py
==================
Lightweight actor-critic networks for MAPPO.
* Decentralised Actor: takes local observation → Gaussian distribution over action.
* Centralised Critic: takes concatenated global observation → single value estimate.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal initialisation for RL layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    Decentralised Actor: obs_i -> a_i
    Outputs mean and standard deviation for a continuous action.
    """

    def __init__(self, obs_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mu = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        
        # Trainable log standard deviation parameter, independent of state
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.mu(h)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.sample()
        
        # Bound action to [-1, 1] using tanh
        action_tanh = torch.tanh(action)
        
        # Enforce bounds for log_prob calculation
        log_prob = dist.log_prob(action)
        log_prob -= torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action_tanh, log_prob, dist.entropy().sum(dim=-1, keepdim=True)

    def evaluate_actions(self, x: torch.Tensor, action: torch.Tensor):
        """Used during PPO updates to evaluate previously taken actions."""
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        
        # Inverse tanh to get unbounded action for log_prob
        # Clamping prevents nan in atanh
        action_clipped = torch.clamp(action, -0.9999, 0.9999)
        action_unbounded = torch.atanh(action_clipped)
        
        log_prob = dist.log_prob(action_unbounded)
        log_prob -= torch.log(1.0 - action_clipped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class CentralisedCritic(nn.Module):
    """
    Centralised Critic: [obs_1, obs_2, ..., obs_N] -> V
    Takes the concatenated observations of all agents and predicts a single shared value.
    """

    def __init__(self, global_obs_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs)

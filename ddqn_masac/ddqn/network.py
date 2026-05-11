"""
ddqn/network.py — Q-Network for Double DQN.

Architecture: Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear(n_actions)
Two instances used: online_net and target_net.
"""

import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    """Q-value network for discrete action DQN.

    Maps observations to Q-values for each discrete action.

    Args:
        obs_dim: Observation dimension.
        n_actions: Number of discrete actions.
        hidden_dims: List of hidden layer sizes.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_actions))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).

        Returns:
            Q-values tensor of shape (batch, n_actions).
        """
        return self.network(obs)

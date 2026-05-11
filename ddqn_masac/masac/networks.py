"""
masac/networks.py — Actor and Critic networks for Multi-Agent SAC.

Actor (one per agent):
    GaussianActor: outputs mean and log_std for a squashed Gaussian policy.
    Tanh squashing with log-prob correction for proper entropy computation.

TwinCritic (shared, centralized):
    Two independent Q-networks taking joint observations and joint actions.
    Used for clipped double-Q to reduce overestimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    """Gaussian policy network with tanh squashing.

    Architecture: Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU
                  → two heads: mean_layer, log_std_layer
                  → tanh squashing with log-prob correction

    Args:
        obs_dim: Observation dimension for this agent.
        action_dim: Action dimension for this agent.
        hidden_dims: List of hidden layer sizes.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None):
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

        self.trunk = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, action_dim)
        self.log_std_layer = nn.Linear(in_dim, action_dim)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for output layers
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log_std of the policy.

        Args:
            obs: Observation tensor (batch, obs_dim).

        Returns:
            Tuple of (mean, log_std), each (batch, action_dim).
        """
        h = self.trunk(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from the policy with reparameterization trick.

        Applies tanh squashing and corrects log-probability.

        Args:
            obs: Observation tensor (batch, obs_dim).

        Returns:
            Tuple of (action, log_prob, mean):
                action: Tanh-squashed action in [-1, 1], shape (batch, action_dim).
                log_prob: Log probability of the action, shape (batch, 1).
                mean: Tanh of the mean (deterministic action), shape (batch, action_dim).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # pre-tanh value
        action = torch.tanh(x_t)

        # Log probability with tanh squashing correction
        # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (tanh of mean).

        Args:
            obs: Observation tensor (batch, obs_dim).

        Returns:
            Deterministic action in [-1, 1], shape (batch, action_dim).
        """
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """Single Q-network for centralized critic.

    Architecture: Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear(1)

    Args:
        input_dim: Total input dim (joint_obs_dim + joint_action_dim).
        hidden_dims: List of hidden layer sizes.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute Q-value.

        Args:
            obs: Joint observation tensor (batch, joint_obs_dim).
            actions: Joint action tensor (batch, joint_action_dim).

        Returns:
            Q-value tensor (batch, 1).
        """
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x)


class TwinCritic(nn.Module):
    """Twin Q-network (clipped double-Q) for centralized critic.

    Two independent Q-networks for reducing overestimation bias.

    Args:
        joint_obs_dim: Total observation dimension across all agents.
        joint_action_dim: Total action dimension across all agents.
        hidden_dims: List of hidden layer sizes.
    """

    def __init__(self, joint_obs_dim: int, joint_action_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = joint_obs_dim + joint_action_dim
        self.q1 = QNetwork(input_dim, hidden_dims)
        self.q2 = QNetwork(input_dim, hidden_dims)

    def forward(self, joint_obs: torch.Tensor,
                joint_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values from both critics.

        Args:
            joint_obs: Joint observation tensor (batch, joint_obs_dim).
            joint_actions: Joint action tensor (batch, joint_action_dim).

        Returns:
            Tuple of (q1, q2), each (batch, 1).
        """
        q1 = self.q1(joint_obs, joint_actions)
        q2 = self.q2(joint_obs, joint_actions)
        return q1, q2

"""
Network definitions for MARLISA CityLearn scripts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _layer_init(layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class GaussianActor(nn.Module):
    """Gaussian policy matching MARLISA checkpoints (backbone + mu/log_std heads)."""

    def __init__(self, obs_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mu = _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        mu = self.mu(h)
        log_std = self.log_std(h)
        std = log_std.exp()
        return mu, std


class ValueNetwork(nn.Module):
    """Centralized critic: concatenated global observation -> scalar value."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

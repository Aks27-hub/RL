"""
utils/buffers.py
================
Rollout buffer for Multi-Agent PPO (MAPPO).
Handles storing transitions and computing Generalized Advantage Estimation (GAE).
"""

from typing import Dict, Generator, Tuple

import numpy as np
import torch


class MAPPORolloutBuffer:
    """
    Buffer to store trajectories for all agents and the central critic.
    """

    def __init__(
        self,
        rollout_steps: int,
        n_agents: int,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.rollout_steps = rollout_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.device = device

        self.reset()

    def reset(self):
        self.obs = np.zeros((self.rollout_steps, self.n_agents, self.obs_dim), dtype=np.float32)
        self.global_obs = np.zeros((self.rollout_steps, self.global_obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.rollout_steps, self.n_agents, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_steps, self.n_agents), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps,), dtype=np.float32)  # Shared reward
        self.values = np.zeros((self.rollout_steps,), dtype=np.float32)   # Central value
        self.dones = np.zeros((self.rollout_steps,), dtype=np.float32)

        self.advantages = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps,), dtype=np.float32)

        self.step = 0

    def add(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: float,
        value: float,
        done: bool,
    ):
        assert self.step < self.rollout_steps, "Buffer is full"

        self.obs[self.step] = obs
        self.global_obs[self.step] = global_obs
        self.actions[self.step] = action.reshape(self.n_agents, self.action_dim)
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = float(done)

        self.step += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute advantages and returns using Generalized Advantage Estimation."""
        last_gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_generator(self, batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yields mini-batches of data for PPO updates."""
        indices = np.random.permutation(self.rollout_steps)
        
        # Flatten advantages and returns since they are shared
        adv = torch.tensor(self.advantages, dtype=torch.float32, device=self.device)
        ret = torch.tensor(self.returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages at batch level
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for start_idx in range(0, self.rollout_steps, batch_size):
            end_idx = min(start_idx + batch_size, self.rollout_steps)
            batch_inds = indices[start_idx:end_idx]

            yield {
                "obs": torch.tensor(self.obs[batch_inds], dtype=torch.float32, device=self.device),
                "global_obs": torch.tensor(self.global_obs[batch_inds], dtype=torch.float32, device=self.device),
                "actions": torch.tensor(self.actions[batch_inds], dtype=torch.float32, device=self.device),
                "log_probs": torch.tensor(self.log_probs[batch_inds], dtype=torch.float32, device=self.device),
                "advantages": adv[batch_inds],
                "returns": ret[batch_inds],
            }

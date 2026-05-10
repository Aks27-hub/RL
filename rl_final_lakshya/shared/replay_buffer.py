"""
shared/replay_buffer.py — Pre-allocated numpy circular replay buffer.

Design choices:
- Pre-allocated np.float32 arrays for obs, next_obs, actions, rewards, dones
- Circular buffer with integer head pointer — no Python lists or dicts
- RAM target: ≤ 2 GB for 100k transitions with obs_dim ≤ 200
- Returns torch tensors on the correct device when sampled
- pin_memory=True when CUDA is available for faster host-to-device transfer
"""

import numpy as np
import torch
from typing import Dict, Tuple


class ReplayBuffer:
    """Pre-allocated circular replay buffer storing transitions as numpy arrays.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation dimension (total, possibly joint for multi-agent).
        action_dim: Action dimension (total, possibly joint for multi-agent).
        device: Torch device for returned tensors.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self._pin_memory = device.type == "cuda"

        # Pre-allocate storage as contiguous float32 arrays
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self._head = 0  # Next write position
        self._size = 0  # Current number of stored transitions

    @property
    def size(self) -> int:
        """Return current number of stored transitions."""
        return self._size

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: Observation array of shape (obs_dim,).
            action: Action array of shape (action_dim,).
            reward: Scalar reward.
            next_obs: Next observation array of shape (obs_dim,).
            done: Whether the episode ended.
        """
        idx = self._head
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx, 0] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx, 0] = float(done)

        self._head = (self._head + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones',
            each a torch.Tensor on the configured device.
        """
        indices = np.random.randint(0, self._size, size=batch_size)

        batch = {
            "obs": self._to_tensor(self.obs[indices]),
            "actions": self._to_tensor(self.actions[indices]),
            "rewards": self._to_tensor(self.rewards[indices]),
            "next_obs": self._to_tensor(self.next_obs[indices]),
            "dones": self._to_tensor(self.dones[indices]),
        }
        return batch

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on the correct device.

        Uses pin_memory for faster CUDA transfers when applicable.

        Args:
            array: Numpy array to convert.

        Returns:
            Torch tensor on self.device.
        """
        tensor = torch.from_numpy(array)
        if self._pin_memory:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=True)

    def __len__(self) -> int:
        return self._size

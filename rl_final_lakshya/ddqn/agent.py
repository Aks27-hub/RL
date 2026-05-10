"""
ddqn/agent.py — Double DQN Agent for multi-agent CityLearn.

Discrete action handling:
    CityLearn actions are continuous in [-1, 1]. We discretize each agent's
    action space into n_bins uniform bins per action dimension.
    Example with n_bins=5: bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
    The agent selects a bin index; we map it back to a continuous value.

    For an agent with action_dim=d, total discrete actions = n_bins^d.
    With n_bins=5 and d=1, total = 5 actions per agent.

Architecture:
    One independent QNetwork per agent (independent learners).
    One shared ReplayBuffer for all agents.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from ddqn.network import QNetwork
from shared.replay_buffer import ReplayBuffer


class DDQNAgent:
    """Double DQN agent for multi-agent CityLearn with discretized actions.

    Args:
        obs_dim: Observation dimension per agent.
        action_dims_per_agent: List of continuous action dimensions per agent.
        n_agents: Number of agents.
        config: Configuration dict with all hyperparameters.
        device: Torch device.
    """

    def __init__(self, obs_dim: int, action_dims_per_agent: List[int],
                 n_agents: int, config: Dict[str, Any], device: torch.device):
        self.obs_dim = obs_dim
        self.action_dims_per_agent = action_dims_per_agent
        self.n_agents = n_agents
        self.config = config
        self.device = device

        algo_cfg = config.get("algo_specific", {})
        self.n_bins = algo_cfg.get("n_bins", 5)
        self.epsilon_start = algo_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = algo_cfg.get("epsilon_end", 0.05)
        self.epsilon_decay_steps = algo_cfg.get("epsilon_decay_steps", 50000)
        self.target_update_freq = algo_cfg.get("target_update_freq", 500)
        self.grad_clip = algo_cfg.get("grad_clip", 10.0)

        hidden_dims = config.get("hidden_dims", [256, 256])
        lr = config.get("lr", 3e-4)
        gamma = config.get("gamma", 0.99)
        self.gamma = gamma

        # Precompute discrete action bins for each agent
        # bins maps bin_index -> continuous action value in [-1, 1]
        self._bin_values = np.linspace(-1.0, 1.0, self.n_bins).astype(np.float32)

        # Compute total discrete actions per agent: n_bins^action_dim
        self._n_discrete_per_agent = [self.n_bins ** d for d in action_dims_per_agent]

        # Precompute multi-dim bin index mappings for each agent
        # For agent i with action_dim d:
        #   action_index in [0, n_bins^d) maps to d continuous values
        self._action_maps = []
        for d in action_dims_per_agent:
            n_total = self.n_bins ** d
            mapping = np.zeros((n_total, d), dtype=np.float32)
            for idx in range(n_total):
                remainder = idx
                for dim in range(d - 1, -1, -1):
                    mapping[idx, dim] = self._bin_values[remainder % self.n_bins]
                    remainder //= self.n_bins
            self._action_maps.append(mapping)

        # Create Q-networks: one online + one target per agent
        self.online_nets = []
        self.target_nets = []
        self.optimizers = []
        for i in range(n_agents):
            n_actions = self._n_discrete_per_agent[i]
            online = QNetwork(obs_dim, n_actions, hidden_dims).to(device)
            target = QNetwork(obs_dim, n_actions, hidden_dims).to(device)
            target.load_state_dict(online.state_dict())
            target.eval()

            optimizer = torch.optim.Adam(online.parameters(), lr=lr)
            self.online_nets.append(online)
            self.target_nets.append(target)
            self.optimizers.append(optimizer)

        # Shared replay buffer
        # Joint obs: concatenated observations of all agents
        # Actions stored as discrete indices (one per agent)
        joint_obs_dim = obs_dim * n_agents
        buffer_size = config.get("replay_buffer_size", 100000)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            obs_dim=joint_obs_dim,
            action_dim=n_agents,  # one discrete index per agent
            device=device
        )

        self._step_count = 0
        self._epsilon = self.epsilon_start

    @property
    def epsilon(self) -> float:
        """Current epsilon value for epsilon-greedy exploration."""
        return self._epsilon

    def _update_epsilon(self) -> None:
        """Linear epsilon decay schedule."""
        fraction = min(1.0, self._step_count / max(1, self.epsilon_decay_steps))
        self._epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert single agent observation to tensor on device."""
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _discrete_to_continuous(self, agent_idx: int, action_idx: int) -> np.ndarray:
        """Map discrete action index to continuous action values.

        Args:
            agent_idx: Index of the agent.
            action_idx: Discrete action index.

        Returns:
            Continuous action array in [-1, 1].
        """
        return self._action_maps[agent_idx][action_idx].copy()

    def act(self, obs: list, epsilon: float = 0.0, deterministic: bool = False) -> List[np.ndarray]:
        """Select actions for all agents.

        Args:
            obs: List of per-agent observation arrays.
            epsilon: Exploration rate (ignored if deterministic=True).
            deterministic: If True, always greedy (for eval).

        Returns:
            List of continuous action arrays, one per agent.
        """
        actions = []
        eps = 0.0 if deterministic else epsilon

        for i in range(self.n_agents):
            if np.random.random() < eps:
                # Random action
                action_idx = np.random.randint(0, self._n_discrete_per_agent[i])
            else:
                # Greedy action
                obs_tensor = self._obs_to_tensor(np.asarray(obs[i], dtype=np.float32))
                with torch.no_grad():
                    q_values = self.online_nets[i](obs_tensor)
                action_idx = q_values.argmax(dim=1).item()

            continuous_action = self._discrete_to_continuous(i, action_idx)
            actions.append(continuous_action)

        return actions

    def _obs_list_to_joint(self, obs_list: list) -> np.ndarray:
        """Concatenate per-agent observations into joint observation."""
        return np.concatenate([np.asarray(o, dtype=np.float32) for o in obs_list])

    def _actions_to_indices(self, actions: List[np.ndarray]) -> np.ndarray:
        """Convert continuous actions to discrete indices for storage."""
        indices = np.zeros(self.n_agents, dtype=np.float32)
        for i, act in enumerate(actions):
            act = np.asarray(act, dtype=np.float32)
            # Find closest bin combination
            d = self.action_dims_per_agent[i]
            best_idx = 0
            best_dist = float("inf")
            for idx in range(self._n_discrete_per_agent[i]):
                dist = np.sum((self._action_maps[i][idx] - act) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            indices[i] = float(best_idx)
        return indices

    def store(self, obs: list, actions: List[np.ndarray], reward: float,
              next_obs: list, done: bool) -> None:
        """Store a transition in the replay buffer.

        Args:
            obs: List of per-agent observations.
            actions: List of per-agent continuous actions.
            reward: Scalar reward (sum of per-agent rewards).
            next_obs: List of per-agent next observations.
            done: Whether episode ended.
        """
        joint_obs = self._obs_list_to_joint(obs)
        joint_next_obs = self._obs_list_to_joint(next_obs)
        action_indices = self._actions_to_indices(actions)
        self.replay_buffer.add(joint_obs, action_indices, reward, joint_next_obs, done)
        self._step_count += 1
        self._update_epsilon()

    def update(self) -> Dict[str, float]:
        """Perform one Double DQN update step.

        Double DQN: online net selects action, target net evaluates value.
        Uses Huber loss. Hard target update every target_update_freq steps.

        Returns:
            Dict of losses per agent.
        """
        batch_size = self.config.get("batch_size", 256)
        if self.replay_buffer.size < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        obs = batch["obs"]           # (B, joint_obs_dim)
        actions = batch["actions"]   # (B, n_agents) - discrete indices
        rewards = batch["rewards"]   # (B, 1)
        next_obs = batch["next_obs"] # (B, joint_obs_dim)
        dones = batch["dones"]       # (B, 1)

        losses = {}
        for i in range(self.n_agents):
            # Extract per-agent observations
            start = i * self.obs_dim
            end = start + self.obs_dim
            agent_obs = obs[:, start:end]
            agent_next_obs = next_obs[:, start:end]
            agent_actions = actions[:, i].long()  # (B,)

            # Current Q-values
            q_values = self.online_nets[i](agent_obs)  # (B, n_actions)
            q_current = q_values.gather(1, agent_actions.unsqueeze(1))  # (B, 1)

            # Double DQN target
            with torch.no_grad():
                # Online net selects best action
                next_q_online = self.online_nets[i](agent_next_obs)
                best_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B, 1)
                # Target net evaluates value of that action
                next_q_target = self.target_nets[i](agent_next_obs)
                q_next = next_q_target.gather(1, best_actions)  # (B, 1)
                # TD target
                target = rewards + self.gamma * q_next * (1.0 - dones)

            # Huber loss
            loss = F.smooth_l1_loss(q_current, target)

            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_nets[i].parameters(), self.grad_clip)
            self.optimizers[i].step()

            losses[f"agent_{i}_loss"] = loss.item()

        # Hard target update
        if self._step_count % self.target_update_freq == 0:
            for i in range(self.n_agents):
                self.target_nets[i].load_state_dict(self.online_nets[i].state_dict())

        return losses

    def save(self, path: str) -> None:
        """Save agent state to a checkpoint file.

        Args:
            path: File path for the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "step_count": self._step_count,
            "epsilon": self._epsilon,
            "config": self.config,
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "action_dims_per_agent": self.action_dims_per_agent,
        }
        for i in range(self.n_agents):
            checkpoint[f"online_net_{i}"] = self.online_nets[i].state_dict()
            checkpoint[f"target_net_{i}"] = self.target_nets[i].state_dict()
            checkpoint[f"optimizer_{i}"] = self.optimizers[i].state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load agent state from a checkpoint file.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self._step_count = checkpoint["step_count"]
        self._epsilon = checkpoint["epsilon"]
        for i in range(self.n_agents):
            self.online_nets[i].load_state_dict(checkpoint[f"online_net_{i}"])
            self.target_nets[i].load_state_dict(checkpoint[f"target_net_{i}"])
            self.optimizers[i].load_state_dict(checkpoint[f"optimizer_{i}"])

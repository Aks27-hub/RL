"""
masac/agent.py — Multi-Agent Soft Actor-Critic Agent.

Architecture:
    - One GaussianActor per agent (decentralized execution)
    - One shared TwinCritic + target TwinCritic (centralized training)
    - One log_alpha per agent for automatic entropy tuning
    - One shared ReplayBuffer storing joint obs/actions
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from masac.networks import GaussianActor, TwinCritic
from shared.replay_buffer import ReplayBuffer


class MASACAgent:
    """Multi-Agent Soft Actor-Critic agent.

    Args:
        obs_dims: List of observation dimensions per agent.
        action_dims: List of action dimensions per agent.
        n_agents: Number of agents.
        config: Configuration dict with all hyperparameters.
        device: Torch device.
    """

    def __init__(self, obs_dims: List[int], action_dims: List[int],
                 n_agents: int, config: Dict[str, Any], device: torch.device):
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.n_agents = n_agents
        self.config = config
        self.device = device

        algo_cfg = config.get("algo_specific", {})
        self.tau = algo_cfg.get("tau", 0.005)
        self.auto_alpha = algo_cfg.get("auto_alpha", True)
        self.target_entropy_scale = algo_cfg.get("target_entropy_scale", 1.0)
        self.actor_update_freq = algo_cfg.get("actor_update_freq", 1)
        self.grad_clip = algo_cfg.get("grad_clip", 5.0)
        self.warmup_steps = algo_cfg.get("warmup_steps", 1000)
        alpha_init = algo_cfg.get("alpha_init", 0.2)

        hidden_dims = config.get("hidden_dims", [256, 256])
        lr = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)

        # Joint dimensions
        self.joint_obs_dim = sum(obs_dims)
        self.joint_action_dim = sum(action_dims)

        # Create actors (one per agent)
        self.actors = []
        self.actor_optimizers = []
        for i in range(n_agents):
            actor = GaussianActor(obs_dims[i], action_dims[i], hidden_dims).to(device)
            optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
            self.actors.append(actor)
            self.actor_optimizers.append(optimizer)

        # Shared twin critic (centralized)
        self.critic = TwinCritic(self.joint_obs_dim, self.joint_action_dim, hidden_dims).to(device)
        self.critic_target = TwinCritic(self.joint_obs_dim, self.joint_action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning: one log_alpha per agent
        self.log_alphas = []
        self.alpha_optimizers = []
        self.target_entropies = []
        for i in range(n_agents):
            log_alpha = torch.tensor(
                np.log(alpha_init), dtype=torch.float32, device=device, requires_grad=True
            )
            alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr)
            target_entropy = -float(action_dims[i]) * self.target_entropy_scale
            self.log_alphas.append(log_alpha)
            self.alpha_optimizers.append(alpha_optimizer)
            self.target_entropies.append(target_entropy)

        # Shared replay buffer (stores joint obs/actions)
        buffer_size = config.get("replay_buffer_size", 100000)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            obs_dim=self.joint_obs_dim,
            action_dim=self.joint_action_dim,
            device=device,
        )

        self._step_count = 0
        self._update_count = 0

    @property
    def alphas(self) -> List[float]:
        """Current entropy temperature values."""
        return [la.exp().item() for la in self.log_alphas]

    def act(self, obs_list: list, deterministic: bool = False) -> List[np.ndarray]:
        """Select actions for all agents.

        Args:
            obs_list: List of per-agent observation arrays.
            deterministic: If True, use mean action (no sampling).

        Returns:
            List of action arrays, one per agent.
        """
        actions = []
        for i in range(self.n_agents):
            obs_tensor = torch.as_tensor(
                np.asarray(obs_list[i], dtype=np.float32),
                dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                if deterministic:
                    action = self.actors[i].get_deterministic_action(obs_tensor)
                else:
                    action, _, _ = self.actors[i].sample(obs_tensor)

            actions.append(action.squeeze(0).cpu().numpy())
        return actions

    def _obs_list_to_joint(self, obs_list: list) -> np.ndarray:
        """Concatenate per-agent observations into joint observation."""
        return np.concatenate([np.asarray(o, dtype=np.float32) for o in obs_list])

    def _actions_to_joint(self, actions: List[np.ndarray]) -> np.ndarray:
        """Concatenate per-agent actions into joint action."""
        return np.concatenate([np.asarray(a, dtype=np.float32) for a in actions])

    def store(self, obs_list: list, actions: List[np.ndarray], rewards: list,
              next_obs_list: list, done: bool) -> None:
        """Store a transition in the replay buffer.

        Args:
            obs_list: List of per-agent observations.
            actions: List of per-agent actions.
            rewards: List of per-agent rewards.
            next_obs_list: List of per-agent next observations.
            done: Whether episode ended.
        """
        joint_obs = self._obs_list_to_joint(obs_list)
        joint_action = self._actions_to_joint(actions)
        joint_next_obs = self._obs_list_to_joint(next_obs_list)
        reward = float(np.mean(rewards))  # average reward across agents
        self.replay_buffer.add(joint_obs, joint_action, reward, joint_next_obs, done)
        self._step_count += 1

    def update(self) -> Dict[str, float]:
        """Perform one MASAC update step.

        Updates:
        1. Twin critics with clipped double-Q target
        2. Each actor with reparameterized policy gradient
        3. Each alpha with entropy constraint

        Returns:
            Dict of all losses and alpha values.
        """
        batch_size = self.config.get("batch_size", 256)
        if self.replay_buffer.size < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        obs = batch["obs"]           # (B, joint_obs_dim)
        actions = batch["actions"]   # (B, joint_action_dim)
        rewards = batch["rewards"]   # (B, 1)
        next_obs = batch["next_obs"] # (B, joint_obs_dim)
        dones = batch["dones"]       # (B, 1)

        losses = {}

        # ---- Critic update ----
        with torch.no_grad():
            # Sample next actions from all actors
            next_actions_list = []
            next_log_probs_list = []
            act_offset = 0
            obs_offset = 0
            for i in range(self.n_agents):
                agent_next_obs = next_obs[:, obs_offset:obs_offset + self.obs_dims[i]]
                obs_offset += self.obs_dims[i]
                na, nlp, _ = self.actors[i].sample(agent_next_obs)
                next_actions_list.append(na)
                next_log_probs_list.append(nlp)

            next_joint_actions = torch.cat(next_actions_list, dim=-1)
            # Weighted sum of log probs with per-agent alpha
            total_alpha_log_prob = sum(
                self.log_alphas[i].exp().detach() * next_log_probs_list[i]
                for i in range(self.n_agents)
            )

            # Target Q-values
            q1_next, q2_next = self.critic_target(next_obs, next_joint_actions)
            q_next = torch.min(q1_next, q2_next) - total_alpha_log_prob
            target_q = rewards + self.gamma * (1.0 - dones) * q_next

        # Current Q-values
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        losses["critic_loss"] = critic_loss.item()

        self._update_count += 1

        # ---- Actor and Alpha updates ----
        if self._update_count % self.actor_update_freq == 0:
            # Re-sample actions from current policies
            new_actions_list = []
            new_log_probs_list = []
            obs_offset = 0
            for i in range(self.n_agents):
                agent_obs = obs[:, obs_offset:obs_offset + self.obs_dims[i]]
                obs_offset += self.obs_dims[i]
                na, nlp, _ = self.actors[i].sample(agent_obs)
                new_actions_list.append(na)
                new_log_probs_list.append(nlp)

            new_joint_actions = torch.cat(new_actions_list, dim=-1)

            # Compute Q-values for joint actions once
            q1_new, q2_new = self.critic(obs, new_joint_actions)
            q_new = torch.min(q1_new, q2_new)

            # Compute combined actor loss for all agents in one backward pass
            # This avoids the inplace modification error from sequential backwards
            total_actor_loss = torch.tensor(0.0, device=self.device)
            for i in range(self.n_agents):
                alpha = self.log_alphas[i].exp().detach()
                agent_actor_loss = (alpha * new_log_probs_list[i] - q_new).mean()
                total_actor_loss = total_actor_loss + agent_actor_loss
                losses[f"actor_{i}_loss"] = agent_actor_loss.item()

            # Zero all actor gradients, backward once, then step all
            for i in range(self.n_agents):
                self.actor_optimizers[i].zero_grad()
            total_actor_loss.backward()
            for i in range(self.n_agents):
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.grad_clip)
                self.actor_optimizers[i].step()

            # Alpha updates (detached from actor graph, safe to do per-agent)
            for i in range(self.n_agents):
                if self.auto_alpha:
                    alpha_loss = -(self.log_alphas[i] * (
                        new_log_probs_list[i].detach() + self.target_entropies[i]
                    )).mean()

                    self.alpha_optimizers[i].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizers[i].step()

                    losses[f"alpha_{i}"] = self.log_alphas[i].exp().item()
                    losses[f"alpha_{i}_loss"] = alpha_loss.item()

        # Soft update target critic
        self._soft_update_target()

        return losses

    def _soft_update_target(self) -> None:
        """Soft update target critic: θ_target ← τ*θ + (1-τ)*θ_target."""
        critic_state = self.critic.state_dict()
        target_state = self.critic_target.state_dict()
        for key in target_state:
            target_state[key] = self.tau * critic_state[key] + (1.0 - self.tau) * target_state[key]
        self.critic_target.load_state_dict(target_state)

    def save(self, path: str) -> None:
        """Save agent state to checkpoint.

        Args:
            path: File path for the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "step_count": self._step_count,
            "update_count": self._update_count,
            "config": self.config,
            "n_agents": self.n_agents,
            "obs_dims": self.obs_dims,
            "action_dims": self.action_dims,
        }
        # Save actors
        for i in range(self.n_agents):
            checkpoint[f"actor_{i}"] = self.actors[i].state_dict()
            checkpoint[f"actor_optimizer_{i}"] = self.actor_optimizers[i].state_dict()
            checkpoint[f"log_alpha_{i}"] = self.log_alphas[i].detach().cpu()
            checkpoint[f"alpha_optimizer_{i}"] = self.alpha_optimizers[i].state_dict()
        # Save critic
        checkpoint["critic"] = self.critic.state_dict()
        checkpoint["critic_target"] = self.critic_target.state_dict()
        checkpoint["critic_optimizer"] = self.critic_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load agent state from checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self._step_count = checkpoint["step_count"]
        self._update_count = checkpoint["update_count"]

        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint[f"actor_{i}"])
            self.actor_optimizers[i].load_state_dict(checkpoint[f"actor_optimizer_{i}"])
            self.log_alphas[i] = checkpoint[f"log_alpha_{i}"].to(self.device).requires_grad_(True)
            # Rebuild alpha optimizer with new parameter reference
            self.alpha_optimizers[i] = torch.optim.Adam([self.log_alphas[i]], lr=self.config.get("lr", 3e-4))
            self.alpha_optimizers[i].load_state_dict(checkpoint[f"alpha_optimizer_{i}"])

        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

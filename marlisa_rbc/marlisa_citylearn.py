from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from citylearn_common import (
    CityLearnEnvConfig,
    ObsProcessor,
    _format_actions,
    _reset_env,
    evaluate_policy,
    get_n_buildings,
    make_citylearn_env,
    save_config,
    save_json,
)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from networks import GaussianActor, ValueNetwork


@dataclass
class MARLISAConfig:
    env: CityLearnEnvConfig
    episodes: int = 20
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 5
    minibatch_size: int = 256
    update_steps: int = 1024
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    eval_interval: int = 5
    eval_episodes: int = 1
    eval_max_steps: Optional[int] = None
    checkpoint_every: int = 5
    output_dir: str = "outputs/marlisa"


class MARLISAgent:
    def __init__(self, state_dim: int, cfg: MARLISAConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = GaussianActor(state_dim, action_dim=1).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

    def select_action(self, state: np.ndarray) -> Tuple[float, float]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std = self.actor(s)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return float(action.squeeze().cpu().item()), float(log_prob.squeeze().detach().cpu().item())

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        mu, std = self.actor(states)
        actions = torch.clamp(actions, -0.999, 0.999)
        z = 0.5 * torch.log((1 + actions) / (1 - actions))
        dist = Normal(mu, std)
        log_prob = dist.log_prob(z) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class MARLISABuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.joint_states: List[np.ndarray] = []
        self.per_agent_states: List[np.ndarray] = []
        self.per_agent_actions: List[np.ndarray] = []
        self.per_agent_logps: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []

    def add(self, joint_state, per_agent_states, actions, logps, reward, done, value):
        self.joint_states.append(joint_state)
        self.per_agent_states.append(per_agent_states)
        self.per_agent_actions.append(actions)
        self.per_agent_logps.append(logps)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * (1.0 - dones[step]) * next_value - values[step]
        gae = delta + gamma * gae_lambda * (1.0 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = np.array(advantages, dtype=np.float32) + np.array(values, dtype=np.float32)
    return np.array(advantages, dtype=np.float32), returns


class MARLISATrainer:
    def __init__(self, cfg: MARLISAConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(cfg.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "metrics"

        self.env = make_citylearn_env(cfg.env.schema_path, cfg.env.seed, central_agent=False)
        self.obs_processor = ObsProcessor(self.env.observation_space, normalize=cfg.env.normalize_obs)
        self.n_buildings = get_n_buildings(self.env)
        if isinstance(self.env.observation_space, (list, tuple)):
            obs_space = self.env.observation_space[0]
        else:
            obs_space = self.env.observation_space
        self.obs_dim = int(np.prod(obs_space.shape))

        self.agents = [MARLISAgent(self.obs_dim, cfg, device) for _ in range(self.n_buildings)]
        self.critic = ValueNetwork(self.obs_dim * self.n_buildings).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.buffer = MARLISABuffer()
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_eval_reward = -1e9
        self.best_saved = False

        self._init_output_dirs()
        save_config(self.output_dir / "config.json", asdict(cfg))

    def _init_output_dirs(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, episode: int):
        payload = {
            "actors": [agent.actor.state_dict() for agent in self.agents],
            "critic": self.critic.state_dict(),
            "episode": episode,
        }
        path = self.checkpoint_dir / f"episode_{episode:04d}.pt"
        torch.save(payload, path)

    def _save_best(self):
        payload = {
            "actors": [agent.actor.state_dict() for agent in self.agents],
            "critic": self.critic.state_dict(),
        }
        torch.save(payload, self.output_dir / "best_model.pt")

    def train(self):
        reward_history: List[float] = []
        csv_path = self.output_dir / "training_curves.csv"
        self._write_training_header(csv_path)

        for ep in range(1, self.cfg.episodes + 1):
            obs = _reset_env(self.env, self.cfg.env.seed + ep)
            done = False
            ep_reward = 0.0
            step_counter = 0

            while not done:
                proc_obs = self.obs_processor.transform(obs)
                actions = np.zeros(self.n_buildings, dtype=np.float32)
                logps = np.zeros(self.n_buildings, dtype=np.float32)

                for i, agent in enumerate(self.agents):
                    action, logp = agent.select_action(proc_obs[i])
                    actions[i] = action
                    logps[i] = logp

                joint_state = proc_obs.reshape(-1).astype(np.float32)
                with torch.no_grad():
                    joint_state_t = torch.tensor(joint_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    value = float(self.critic(joint_state_t).squeeze().cpu().item())

                step_out = self.env.step(_format_actions(actions, self.env.action_space))
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    next_obs, rewards, terminated, truncated, _ = step_out
                else:
                    next_obs, rewards, done, _ = step_out
                    terminated = bool(done)
                    truncated = False
                done_flag = float(terminated or truncated)
                shared_reward = float(np.mean(rewards))

                self.buffer.add(joint_state, proc_obs.copy(), actions.copy(), logps.copy(), shared_reward, done_flag, value)

                obs = next_obs
                ep_reward += shared_reward
                done = bool(terminated or truncated)
                step_counter += 1

                if self.cfg.env.max_steps is not None and step_counter >= self.cfg.env.max_steps:
                    break

                if step_counter % self.cfg.update_steps == 0:
                    self._update()

            self._update()
            reward_history.append(ep_reward)
            self.writer.add_scalar("train/episode_reward", ep_reward, ep)
            self._append_training_row(csv_path, ep, ep_reward, None)

            if ep % self.cfg.eval_interval == 0:
                eval_reward = self.evaluate(save_json_out=True)
                self.writer.add_scalar("eval/episode_reward", eval_reward, ep)
                self._append_training_row(csv_path, ep, ep_reward, eval_reward, overwrite_last=True)

                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self._save_best()
                    self.best_saved = True

            if ep % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(ep)

        self.writer.flush()
        self.writer.close()
        if not self.best_saved:
            self._save_best()
        return reward_history

    def evaluate(self, save_json_out: bool = True) -> float:
        eval_rewards = []
        metrics_out: Dict[str, float] = {}

        def policy_fn(proc_obs: np.ndarray, _t: int) -> np.ndarray:
            actions = np.zeros(self.n_buildings, dtype=np.float32)
            with torch.no_grad():
                for i, agent in enumerate(self.agents):
                    s = torch.tensor(proc_obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    mu, _ = agent.actor(s)
                    actions[i] = float(torch.tanh(mu).squeeze().cpu().item())
            return actions

        for i in range(self.cfg.eval_episodes):
            metrics, _ = evaluate_policy(
                self.env,
                policy_fn,
                self.obs_processor,
                self.cfg.eval_max_steps,
                seed=self.cfg.env.seed + 1000 + i,
            )
            eval_rewards.append(metrics.get("reward", 0.0))
            metrics_out = metrics

        avg_reward = float(np.mean(eval_rewards)) if eval_rewards else 0.0
        metrics_out["reward"] = avg_reward

        if save_json_out:
            save_json(self.output_dir / "evaluation.json", metrics_out)

        return avg_reward

    def _update(self):
        if len(self.buffer.joint_states) == 0:
            return

        joint_states = torch.tensor(np.array(self.buffer.joint_states, dtype=np.float32), device=self.device)
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)

        adv_np, ret_np = compute_gae(
            rewards,
            dones,
            values,
            last_value=0.0,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        advantages = torch.tensor(adv_np, dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns = torch.tensor(ret_np, dtype=torch.float32, device=self.device).unsqueeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = joint_states.shape[0]
        idx_all = np.arange(n)

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.cfg.minibatch_size):
                idx = idx_all[start : start + self.cfg.minibatch_size]
                batch_joint_states = joint_states[idx]

                values_pred = self.critic(batch_joint_states)
                critic_loss = nn.functional.mse_loss(values_pred, returns[idx])

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                for agent_id, agent in enumerate(self.agents):
                    batch_states = torch.tensor(
                        np.array([self.buffer.per_agent_states[i][agent_id] for i in idx], dtype=np.float32),
                        device=self.device,
                    )
                    batch_actions = torch.tensor(
                        np.array([self.buffer.per_agent_actions[i][agent_id] for i in idx], dtype=np.float32),
                        device=self.device,
                    ).unsqueeze(-1)
                    batch_old_logp = torch.tensor(
                        np.array([self.buffer.per_agent_logps[i][agent_id] for i in idx], dtype=np.float32),
                        device=self.device,
                    ).unsqueeze(-1)

                    new_logp, entropy = agent.evaluate_actions(batch_states, batch_actions)
                    ratio = torch.exp(new_logp - batch_old_logp)

                    unclipped = ratio * advantages[idx]
                    clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages[idx]
                    actor_loss = -torch.min(unclipped, clipped).mean() - self.cfg.entropy_coef * entropy.mean()

                    agent.actor_optim.zero_grad()
                    actor_loss.backward()
                    agent.actor_optim.step()

        self.buffer.reset()

    def _write_training_header(self, path: Path):
        if path.exists():
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "eval_reward"])

    def _append_training_row(self, path: Path, episode: int, reward: float, eval_reward: Optional[float], overwrite_last: bool = False):
        rows = []
        if overwrite_last and path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if len(rows) > 1 and int(rows[-1][0]) == episode:
                rows = rows[:-1]
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if rows:
                writer.writerows(rows)
            else:
                writer.writerow(["episode", "reward", "eval_reward"])
            writer.writerow([episode, f"{reward:.6f}", "" if eval_reward is None else f"{eval_reward:.6f}"])


def load_trainer(config_path: Optional[str], device: torch.device) -> MARLISATrainer:
    if config_path is None:
        raise ValueError("config_path is required")
    from citylearn_common import load_json

    cfg_data = load_json(Path(config_path))
    env_cfg = CityLearnEnvConfig(**cfg_data["env"])
    cfg = MARLISAConfig(env=env_cfg, **{k: v for k, v in cfg_data.items() if k != "env"})
    return MARLISATrainer(cfg, device)

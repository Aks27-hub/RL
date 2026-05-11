import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import MasterConfig, get_default_config
from models.networks import Actor, CentralisedCritic
from utils.buffers import MAPPORolloutBuffer
from utils.env_wrapper import DistrictEnergyEnv, EnvParams
from utils.logger import MetricsLogger

class RunningStat:
    """Running mean and standard deviation for reward normalization."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class MAPPOAgent:
    def __init__(self, cfg: MasterConfig, env: DistrictEnergyEnv):
        self.cfg = cfg
        self.env = env
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device in ["auto", "cuda"] else "cpu")
        print(f"Using device: {self.device}")

        self.n_agents = env.n_buildings
        self.obs_dim = env.state_dim
        self.global_obs_dim = env.global_state_dim
        
        # Decentralised Actors (one for each agent)
        self.actors = nn.ModuleList([
            Actor(self.obs_dim, action_dim=1, hidden_dim=cfg.mappo.hidden_dim).to(self.device)
            for _ in range(self.n_agents)
        ])
        
        # Centralised Critic (shared)
        self.critic = CentralisedCritic(
            self.global_obs_dim, 
            hidden_dim=cfg.mappo.central_hidden_dim
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=cfg.mappo.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.mappo.critic_lr, eps=1e-5)

        # Buffer
        self.buffer = MAPPORolloutBuffer(
            rollout_steps=cfg.mappo.rollout_steps,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=1,
            device=self.device
        )
        
        self.reward_stat = RunningStat()

    def get_actions(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get actions from all actors for the given observations."""
        actions = np.zeros((self.n_agents, 1), dtype=np.float32)
        log_probs = np.zeros(self.n_agents, dtype=np.float32)
        
        with torch.no_grad():
            for i in range(self.n_agents):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                a, lp, _ = self.actors[i].get_action(obs_tensor)
                actions[i] = a.item()
                log_probs[i] = lp.item()
                
        return actions, log_probs

    def get_value(self, global_obs: np.ndarray) -> float:
        """Get value from the centralised critic."""
        with torch.no_grad():
            obs_tensor = torch.tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            value = self.critic(obs_tensor)
        return value.item()

    def update(self) -> dict:
        """Perform PPO update step."""
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "kl_approx": 0.0}
        updates = 0

        for epoch in range(self.cfg.mappo.update_epochs):
            data_generator = self.buffer.get_generator(self.cfg.mappo.minibatch_size)
            
            for batch in data_generator:
                obs = batch["obs"]                # (B, N, Obs)
                global_obs = batch["global_obs"]  # (B, GlobalObs)
                actions = batch["actions"]        # (B, N, Act)
                old_log_probs = batch["log_probs"]# (B, N)
                advantages = batch["advantages"]  # (B,)
                returns = batch["returns"]        # (B,)

                # 1. Evaluate actions
                new_log_probs = torch.zeros_like(old_log_probs)
                entropies = torch.zeros_like(old_log_probs)
                
                for i in range(self.n_agents):
                    lp, ent = self.actors[i].evaluate_actions(obs[:, i, :], actions[:, i, :])
                    new_log_probs[:, i] = lp.squeeze(-1)
                    entropies[:, i] = ent.squeeze(-1)

                # Sum log probs over agents (joint action probability)
                # For MAPPO, we can treat the joint probability as the product of marginals
                joint_new_log_prob = new_log_probs.sum(dim=-1)
                joint_old_log_prob = old_log_probs.sum(dim=-1)
                
                # 2. Actor Loss
                ratio = torch.exp(joint_new_log_prob - joint_old_log_prob)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.mappo.clip_eps, 1.0 + self.cfg.mappo.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (summed over agents)
                entropy = entropies.sum(dim=-1).mean()
                
                # 3. Critic Loss
                values = self.critic(global_obs).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, returns)

                # 4. Total Loss & Backprop
                loss = actor_loss - self.cfg.mappo.entropy_coef * entropy
                
                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actors.parameters(), self.cfg.mappo.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.mappo.max_grad_norm)
                self.critic_optimizer.step()

                # Logging metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    
                metrics["actor_loss"] += actor_loss.item()
                metrics["critic_loss"] += critic_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["kl_approx"] += approx_kl.item()
                updates += 1

        # Average metrics
        for k in metrics:
            metrics[k] /= updates
            
        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "critic": self.critic.state_dict(),
            "actors": [actor.state_dict() for actor in self.actors]
        }
        torch.save(save_dict, path)
        
    def load(self, path: str):
        save_dict = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(save_dict["critic"])
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(save_dict["actors"][i])


def train():
    cfg = get_default_config()
    
    # Setup seeding
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", cfg.algorithm)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    
    # Save config
    cfg.save(os.path.join(out_dir, "config.json"))
    
    # Setup Environment
    env_params = EnvParams(
        n_buildings=cfg.env.n_buildings,
        horizon=cfg.env.horizon,
        battery_capacity_kwh=cfg.env.battery_capacity_kwh,
        battery_max_power_kw=cfg.env.battery_max_power_kw,
        seed=cfg.train.seed
    )
    env = DistrictEnergyEnv(env_params)

    # Setup Agent
    agent = MAPPOAgent(cfg, env)
    
    # Setup Logger
    logger = MetricsLogger(out_dir, algorithm=cfg.algorithm)
    logger.info("Starting MAPPO Training")
    
    start_time = time.time()
    best_reward = -float("inf")
    episodes_without_improvement = 0
    
    total_steps = 0
    
    for episode in range(1, cfg.train.max_episodes + 1):
        if time.time() - start_time > cfg.train.max_seconds:
            logger.info("Time limit reached. Stopping training.")
            break
            
        obs, _ = env.reset(seed=cfg.train.seed + episode)
        global_obs = env.global_obs(obs)
        
        episode_reward = 0
        episode_cost = 0
        episode_emissions = 0
        episode_ramp = []
        episode_peak = []
        
        done = False
        step = 0
        
        while not done:
            # Gather rollout data
            for _ in range(cfg.mappo.rollout_steps):
                if done:
                    break
                    
                actions, log_probs = agent.get_actions(obs)
                value = agent.get_value(global_obs)
                
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                # Use shared reward (first agent's reward is representative of the shared district reward)
                shared_reward = float(rewards[0])
                
                if cfg.train.reward_norm:
                    agent.reward_stat.update(np.array([shared_reward]))
                    norm_reward = (shared_reward - agent.reward_stat.mean) / (np.sqrt(agent.reward_stat.var) + 1e-8)
                    norm_reward = np.clip(norm_reward, -cfg.train.reward_norm_clip, cfg.train.reward_norm_clip)
                else:
                    norm_reward = shared_reward
                
                next_global_obs = env.global_obs(next_obs)
                
                agent.buffer.add(
                    obs, global_obs, actions, log_probs, norm_reward, value, done
                )
                
                obs = next_obs
                global_obs = next_global_obs
                
                episode_reward += shared_reward
                episode_cost += info["electricity_cost"]
                episode_emissions += info["carbon_emissions"]
                episode_ramp.append(info["ramp"])
                episode_peak.append(info["district_load"])
                
                total_steps += 1
                step += 1

            # PPO Update
            if done:
                last_val = 0.0
            else:
                last_val = agent.get_value(global_obs)
                
            agent.buffer.compute_gae(last_val, cfg.mappo.gamma, cfg.mappo.gae_lambda)
            update_metrics = agent.update()
            agent.buffer.reset()

        # Episode logging
        ep_metrics = {
            "reward": episode_reward,
            "electricity_cost": episode_cost,
            "carbon_emissions": episode_emissions,
            "ramping": np.mean(episode_ramp),
            "load_factor": np.sum(episode_peak) / (np.max(episode_peak) * len(episode_peak)),
            "daily_peak": np.max(episode_peak),
            "comfort_violation": 0.0,
            **update_metrics
        }
        
        logger.log_episode(episode, total_steps, ep_metrics)
        
        if episode % cfg.train.log_every == 0:
            logger.info(
                f"Ep {episode:3d} | Reward: {episode_reward:7.2f} | "
                f"Loss: {update_metrics['actor_loss']:6.3f} / {update_metrics['critic_loss']:6.3f}"
            )
            
        # Checkpointing & Early Stopping
        if episode_reward > best_reward + cfg.train.early_stop_delta:
            best_reward = episode_reward
            episodes_without_improvement = 0
            agent.save(os.path.join(out_dir, "best_model.pt"))
            logger.info(f"New best model saved (Reward: {best_reward:.2f})")
        else:
            episodes_without_improvement += 1
            
        if episode % cfg.train.checkpoint_every == 0:
            agent.save(os.path.join(out_dir, "checkpoints", f"model_ep{episode}.pt"))
            
        if episodes_without_improvement >= cfg.train.early_stop_patience:
            logger.info(f"Early stopping triggered after {episode} episodes.")
            break

    logger.info("Training complete.")
    logger.close()


if __name__ == "__main__":
    train()

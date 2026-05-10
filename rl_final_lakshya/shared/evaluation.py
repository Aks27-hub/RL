"""
shared/evaluation.py — Unified evaluation routine for all algorithms.

evaluate() works for both DDQN and MASAC agents via duck typing.
Only requires the agent to implement:
    agent.act(obs, deterministic=True) -> list of actions
"""

import numpy as np
import torch
from typing import Any, Dict, List

from shared.metrics import extract_metrics


def evaluate(agent: Any, env: Any, n_episodes: int, seed: int) -> Dict[str, Any]:
    """Run evaluation episodes and return standardized results.

    Args:
        agent: RL agent with .act(obs, deterministic=True) method.
        env: CityLearnWrapper environment instance.
        n_episodes: Number of evaluation episodes to run.
        seed: Random seed for reproducibility.

    Returns:
        Dict matching the evaluation.json schema.
    """
    env.set_training(False)
    all_metrics: List[Dict[str, float]] = []

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)
        obs = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                actions = agent.act(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards)
        ep_metrics = extract_metrics(env, episode_reward=episode_reward)
        all_metrics.append(ep_metrics)

    env.set_training(True)
    rewards = [m["reward"] for m in all_metrics]
    result = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_electricity_cost": float(np.mean([m["electricity_cost"] for m in all_metrics])),
        "mean_carbon_emissions": float(np.mean([m["carbon_emissions"] for m in all_metrics])),
        "mean_ramping": float(np.mean([m["ramping"] for m in all_metrics])),
        "mean_load_factor": float(np.mean([m["load_factor"] for m in all_metrics])),
        "mean_daily_peak": float(np.mean([m["daily_peak"] for m in all_metrics])),
        "mean_comfort_violation": float(np.mean([m["comfort_violation"] for m in all_metrics])),
        "num_eval_episodes": n_episodes,
    }
    return result

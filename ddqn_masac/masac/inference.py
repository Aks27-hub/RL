"""
masac/inference.py — Run single-episode inference with a trained MASAC agent.

Usage:
    python masac/inference.py --checkpoint outputs/masac/best_model.pt --render
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import set_seed, get_device, load_config
from shared.env_wrapper import make_env
from shared.metrics import extract_metrics
from masac.agent import MASACAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Run MASAC inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--render", action="store_true", help="Print step-by-step actions and rewards")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    set_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    # Load config
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(ckpt_dir), "config.json")
    config = load_config(config_path)

    # Create environment
    env = make_env(config["env_schema"], args.seed)
    env.set_training(False)

    # Reconstruct agent
    agent = MASACAgent(
        obs_dims=env.per_agent_obs_dims,
        action_dims=env.per_agent_action_dims,
        n_agents=env.n_agents,
        config=config,
        device=device,
    )
    agent.load(args.checkpoint)

    # Run one episode
    obs = env.reset()
    episode_reward = 0.0
    done = False
    step = 0
    step_log = []

    while not done:
        with torch.no_grad():
            actions = agent.act(obs, deterministic=True)
        next_obs, rewards, done, info = env.step(actions)
        step_reward = sum(rewards)
        episode_reward += step_reward

        if args.render:
            action_strs = [np.array2string(a, precision=3) for a in actions]
            print(f"Step {step:4d} | Actions: {action_strs} | "
                  f"Rewards: {[f'{r:.4f}' for r in rewards]} | "
                  f"Total: {step_reward:.4f}")

        step_log.append({
            "step": step,
            "actions": [a.tolist() for a in actions],
            "rewards": rewards,
            "step_reward": step_reward,
        })

        obs = next_obs
        step += 1

    # Final metrics
    metrics = extract_metrics(env, episode_reward=episode_reward)

    summary = {
        "algo": "masac",
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "total_steps": step,
        "episode_reward": episode_reward,
        "metrics": metrics,
        "num_step_details": len(step_log),
    }

    # Save inference result
    out_dir = os.path.dirname(args.checkpoint)
    if "checkpoints" in out_dir:
        out_dir = os.path.dirname(out_dir)
    result_path = os.path.join(out_dir, "inference_result.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nEpisode complete: {step} steps, total reward = {episode_reward:.2f}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    print(f"Inference result saved to: {result_path}")


if __name__ == "__main__":
    main()

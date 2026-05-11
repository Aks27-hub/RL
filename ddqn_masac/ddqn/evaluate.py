"""
ddqn/evaluate.py — Evaluate a trained DDQN agent.

Usage:
    python ddqn/evaluate.py --checkpoint outputs/ddqn/best_model.pt --seed 42
"""

import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import set_seed, get_device, load_config
from shared.env_wrapper import make_env
from shared.evaluation import evaluate
from ddqn.agent import DDQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DDQN agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    set_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    # Load config from same directory as checkpoint
    ckpt_dir = os.path.dirname(args.checkpoint)
    # Check parent dir for config.json
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(ckpt_dir), "config.json")
    config = load_config(config_path)

    # Create environment
    env = make_env(config["env_schema"], args.seed)

    # Reconstruct agent
    agent = DDQNAgent(
        obs_dim=env.obs_dim,
        action_dims_per_agent=env.per_agent_action_dims,
        n_agents=env.n_agents,
        config=config,
        device=device,
    )
    agent.load(args.checkpoint)

    # Run evaluation
    result = evaluate(agent, env, args.eval_episodes, args.seed)

    # Build full evaluation output
    eval_output = {
        "algo": "ddqn",
        "seed": args.seed,
        "mean_reward": result["mean_reward"],
        "std_reward": result["std_reward"],
        "mean_electricity_cost": result["mean_electricity_cost"],
        "mean_carbon_emissions": result["mean_carbon_emissions"],
        "mean_ramping": result["mean_ramping"],
        "mean_load_factor": result["mean_load_factor"],
        "mean_daily_peak": result["mean_daily_peak"],
        "mean_comfort_violation": result["mean_comfort_violation"],
        "num_eval_episodes": args.eval_episodes,
        "best_checkpoint": args.checkpoint,
        "training_wall_time_seconds": 0.0,
    }

    # Save evaluation.json
    out_dir = os.path.dirname(args.checkpoint)
    if "checkpoints" in out_dir:
        out_dir = os.path.dirname(out_dir)
    eval_path = os.path.join(out_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(json.dumps(eval_output, indent=2))
    print(f"\nEvaluation saved to: {eval_path}")


if __name__ == "__main__":
    main()

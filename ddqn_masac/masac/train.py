"""
masac/train.py — Full training loop for Multi-Agent SAC on CityLearn.

Usage:
    python masac/train.py --episodes 150 --seed 42 --device auto
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import set_seed, get_device, save_config, make_output_dirs
from shared.env_wrapper import make_env
from shared.metrics import extract_metrics, csv_header, metrics_to_csv_row
from shared.evaluation import evaluate
from shared.plotting import plot_training_curves
from masac.agent import MASACAgent

# Update every N environment steps to avoid excessive gradient updates.
# With 720 steps/episode, this yields ~72 updates/episode instead of 720.
UPDATE_EVERY = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Agent SAC on CityLearn")
    parser.add_argument("--episodes", type=int, default=150, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--schema", type=str, default="citylearn_challenge_2022_phase_1",
                        help="CityLearn schema name")
    parser.add_argument("--simulation_days", type=int, default=30,
                        help="Days to simulate per episode (30=720 steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eval_interval", type=int, default=15, help="Episodes between evaluations")
    parser.add_argument("--checkpoint_interval", type=int, default=25, help="Episodes between checkpoints")
    parser.add_argument("--eval_episodes", type=int, default=3, help="Number of eval episodes")
    parser.add_argument("--early_stopping_patience", type=int, default=30,
                        help="Eval cycles without improvement before stopping")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set CPU threads for CPU-only runs
    torch.set_num_threads(os.cpu_count() or 4)

    # Setup
    device = get_device(args.device)
    set_seed(args.seed)

    # Build config
    algo_specific = {
        "tau": 0.005,
        "alpha_init": 0.2,
        "auto_alpha": True,
        "target_entropy_scale": 1.0,
        "actor_update_freq": 1,
        "grad_clip": 5.0,
        "warmup_steps": 1000,
    }
    config = {
        "algo": "masac",
        "seed": args.seed,
        "env_schema": args.schema,
        "num_episodes": args.episodes,
        "max_steps_per_episode": None,
        "hidden_dims": [256, 256],
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "replay_buffer_size": args.buffer_size,
        "device": args.device,
        "checkpoint_interval": args.checkpoint_interval,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "early_stopping_patience": args.early_stopping_patience,
        "algo_specific": algo_specific,
    }

    # Create output directories
    dirs = make_output_dirs("masac")
    save_config(config, os.path.join(dirs["root"], "config.json"))

    # Create environment with limited simulation period for speed
    env = make_env(args.schema, args.seed, simulation_days=args.simulation_days)
    print(f"Environment: {args.schema}")
    print(f"  n_agents={env.n_agents}, obs_dim={env.obs_dim}, action_dim={env.action_dim}")
    print(f"  simulation_days={args.simulation_days} (~{args.simulation_days * 24} steps/episode)")
    print(f"  Device: {device}")

    # Create agent
    agent = MASACAgent(
        obs_dims=env.per_agent_obs_dims,
        action_dims=env.per_agent_action_dims,
        n_agents=env.n_agents,
        config=config,
        device=device,
    )

    warmup_steps = algo_specific["warmup_steps"]
    global_step = 0

    # TensorBoard
    writer = SummaryWriter(log_dir=dirs["logs"])

    # Training CSV
    csv_path = os.path.join(dirs["root"], "training_curves.csv")
    with open(csv_path, "w") as f:
        f.write(csv_header() + "\n")

    # JSONL metrics log
    jsonl_path = os.path.join(dirs["metrics"], "metrics_log.jsonl")
    jsonl_file = open(jsonl_path, "w")

    # Training state
    best_mean_reward = -float("inf")
    patience_counter = 0
    start_time = time.time()

    pbar = tqdm(range(1, args.episodes + 1), desc="MASAC Training")
    for episode in pbar:
        obs = env.reset()
        episode_reward = 0.0
        done = False
        step_in_episode = 0

        while not done:
            # During warmup, take random actions
            if global_step < warmup_steps:
                actions = []
                for i in range(env.n_agents):
                    act = np.random.uniform(-1.0, 1.0, size=env.per_agent_action_dims[i]).astype(np.float32)
                    actions.append(act)
            else:
                actions = agent.act(obs)

            next_obs, rewards, done, info = env.step(actions)
            total_reward = sum(rewards)

            # Store transition
            agent.store(obs, actions, rewards, next_obs, done)

            # Update only every UPDATE_EVERY steps after warmup
            if step_in_episode % UPDATE_EVERY == 0 and global_step >= warmup_steps and agent.replay_buffer.size >= args.batch_size:
                losses = agent.update()

            obs = next_obs
            episode_reward += total_reward
            step_in_episode += 1
            global_step += 1

        # Extract metrics
        ep_metrics = extract_metrics(env, episode_reward=episode_reward)

        # Log to JSONL (every episode — lightweight)
        ep_log = {"episode": episode, **ep_metrics}
        jsonl_file.write(json.dumps(ep_log) + "\n")
        jsonl_file.flush()

        # Append to CSV (every episode — needed for training_curves.csv completeness)
        with open(csv_path, "a") as f:
            f.write(f"{episode},{metrics_to_csv_row(ep_metrics)}\n")

        # Progress bar
        elapsed = time.time() - start_time
        pbar.set_postfix({
            "reward": f"{episode_reward:.1f}",
            "alpha0": f"{agent.alphas[0]:.3f}",
            "elapsed": f"{elapsed:.0f}s",
        })

        # TensorBoard logging at eval intervals to reduce I/O
        if episode % args.eval_interval == 0:
            writer.add_scalar("train/reward", ep_metrics["reward"], episode)
            writer.add_scalar("train/steps", step_in_episode, episode)
            for i, alpha in enumerate(agent.alphas):
                writer.add_scalar(f"train/alpha_{i}", alpha, episode)

        # Evaluation
        if episode % args.eval_interval == 0:
            eval_result = evaluate(agent, env, args.eval_episodes, args.seed)
            mean_reward = eval_result["mean_reward"]

            writer.add_scalar("eval/mean_reward", mean_reward, episode)
            writer.add_scalar("eval/std_reward", eval_result["std_reward"], episode)

            print(f"\n  [Eval ep={episode}] mean_reward={mean_reward:.2f} "
                  f"± {eval_result['std_reward']:.2f}")

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                patience_counter = 0
                agent.save(os.path.join(dirs["root"], "best_model.pt"))
                agent.save(os.path.join(dirs["checkpoints"], "best_model.pt"))
                print(f"  -> New best model! reward={mean_reward:.2f}")
            else:
                patience_counter += 1

            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping: no improvement for {args.early_stopping_patience} eval cycles")
                break

        # Checkpoint
        if episode % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(dirs["checkpoints"], f"epoch_{episode}.pt")
            agent.save(ckpt_path)

    # Training complete
    total_time = time.time() - start_time
    jsonl_file.close()

    # Final evaluation
    final_eval = evaluate(agent, env, args.eval_episodes, args.seed)
    eval_output = {
        "algo": "masac",
        "seed": args.seed,
        "mean_reward": final_eval["mean_reward"],
        "std_reward": final_eval["std_reward"],
        "mean_electricity_cost": final_eval["mean_electricity_cost"],
        "mean_carbon_emissions": final_eval["mean_carbon_emissions"],
        "mean_ramping": final_eval["mean_ramping"],
        "mean_load_factor": final_eval["mean_load_factor"],
        "mean_daily_peak": final_eval["mean_daily_peak"],
        "mean_comfort_violation": final_eval["mean_comfort_violation"],
        "num_eval_episodes": args.eval_episodes,
        "best_checkpoint": "outputs/masac/checkpoints/best_model.pt",
        "training_wall_time_seconds": total_time,
    }

    eval_path = os.path.join(dirs["root"], "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)

    # Plot training curves
    plot_training_curves(csv_path, dirs["root"])

    writer.close()
    print(f"\nTraining complete! Wall time: {total_time:.1f}s")
    print(f"Best mean reward: {best_mean_reward:.2f}")
    print(f"Results saved to: {dirs['root']}")


if __name__ == "__main__":
    main()

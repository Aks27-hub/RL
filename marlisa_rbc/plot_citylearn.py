import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CityLearn training curves")
    parser.add_argument("--csv", required=True, help="Path to training_curves.csv")
    parser.add_argument("--out", required=True, help="Path to output image")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    episodes = data["episode"]
    rewards = data["reward"]
    eval_rewards = data["eval_reward"] if "eval_reward" in data.dtype.names else None

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Train Reward")
    if eval_rewards is not None:
        plt.plot(episodes, eval_rewards, label="Eval Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MARLISA Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    main()

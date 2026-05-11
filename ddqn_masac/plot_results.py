"""
plot_results.py — Plot and compare training curves across algorithms.

Usage:
    python plot_results.py
    python plot_results.py --algos ddqn masac
    python plot_results.py --output_dir outputs
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training results across algorithms")
    parser.add_argument("--algos", nargs="+", default=["ddqn", "masac"],
                        help="Algorithms to plot")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Base output directory")
    parser.add_argument("--save_dir", type=str, default="outputs/plots",
                        help="Directory to save comparison plots")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    metric_cols = [
        "reward", "electricity_cost", "carbon_emissions",
        "ramping", "load_factor", "daily_peak", "comfort_violation"
    ]

    # Load data from each algorithm
    algo_data = {}
    for algo in args.algos:
        csv_path = os.path.join(args.output_dir, algo, "training_curves.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            algo_data[algo] = df
            print(f"Loaded {len(df)} episodes from {algo}")
        else:
            print(f"Warning: {csv_path} not found, skipping {algo}")

    if not algo_data:
        print("No data found. Train algorithms first.")
        return

    # Colors for each algorithm
    colors = {
        "ddqn": "#2196F3",
        "masac": "#FF5722",
        "mappo": "#4CAF50",
        "marlisa": "#9C27B0",
        "rbc": "#FF9800",
    }

    # Individual metric comparison plots
    for col in metric_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        for algo, df in algo_data.items():
            if col not in df.columns:
                continue
            color = colors.get(algo, "#999999")
            ax.plot(df["episode"], df[col], alpha=0.3, color=color, linewidth=0.8)
            if len(df) >= 10:
                rolling = df[col].rolling(window=10, min_periods=1).mean()
                ax.plot(df["episode"], rolling, linewidth=2.0, color=color, label=algo.upper())
            else:
                ax.plot(df["episode"], df[col], linewidth=2.0, color=color, label=algo.upper())
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Comparison: {col.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_dir, f"compare_{col}.png"), dpi=150)
        plt.close(fig)

    # Combined overview
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    for i, col in enumerate(metric_cols):
        ax = axes[i]
        for algo, df in algo_data.items():
            if col not in df.columns:
                continue
            color = colors.get(algo, "#999999")
            if len(df) >= 10:
                rolling = df[col].rolling(window=10, min_periods=1).mean()
                ax.plot(df["episode"], rolling, linewidth=2.0, color=color, label=algo.upper())
            else:
                ax.plot(df["episode"], df[col], linewidth=2.0, color=color, label=algo.upper())
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    for j in range(len(metric_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Algorithm Comparison — Training Curves", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(args.save_dir, "comparison_overview.png"), dpi=150)
    plt.close(fig)

    print(f"\nComparison plots saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

"""
shared/plotting.py — Training curve visualization.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(csv_path: str, out_dir: str) -> None:
    """Plot training curves from a training_curves.csv file.

    Generates individual plots for each metric column and a combined overview.
    Saves PNG files to out_dir.

    Args:
        csv_path: Path to training_curves.csv.
        out_dir: Directory to save plot images.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    metric_cols = [
        "reward", "electricity_cost", "carbon_emissions",
        "ramping", "load_factor", "daily_peak", "comfort_violation"
    ]

    # Individual plots
    for col in metric_cols:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["episode"], df[col], linewidth=1.0, alpha=0.6, label=col)
        # Rolling average
        if len(df) >= 10:
            rolling = df[col].rolling(window=10, min_periods=1).mean()
            ax.plot(df["episode"], rolling, linewidth=2.0, label=f"{col} (avg-10)")
        ax.set_xlabel("Episode")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_title(f"Training Curve: {col.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{col}.png"), dpi=150)
        plt.close(fig)

    # Combined overview plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(metric_cols):
        if col not in df.columns:
            continue
        ax = axes[i]
        ax.plot(df["episode"], df[col], linewidth=1.0, alpha=0.6)
        if len(df) >= 10:
            rolling = df[col].rolling(window=10, min_periods=1).mean()
            ax.plot(df["episode"], rolling, linewidth=2.0, color="red")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
    # Hide unused subplots
    for j in range(len(metric_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Training Overview", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "overview.png"), dpi=150)
    plt.close(fig)
    print(f"Plots saved to {out_dir}")

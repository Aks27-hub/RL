import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curves(log_dir: str):
    """
    Reads the training_curves.csv and generates standard plots.
    """
    csv_path = os.path.join(log_dir, "training_curves.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("CSV is empty.")
        return
        
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Reward Curve
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['reward'], alpha=0.8)
    # Smooth
    smooth_window = min(10, len(df))
    plt.plot(df['episode'], df['reward'].rolling(smooth_window).mean(), color='red')
    plt.title("Episode Reward vs. Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reward_curve.png"))
    plt.close()
    
    # 2. Key Metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(df['episode'], df['daily_peak'], color='orange')
    axes[0, 0].set_title("Daily Peak (Lower is better)")
    
    axes[0, 1].plot(df['episode'], df['ramping'], color='green')
    axes[0, 1].set_title("Avg Ramping (Lower is better)")
    
    axes[1, 0].plot(df['episode'], df['electricity_cost'], color='blue')
    axes[1, 0].set_title("Electricity Cost")
    
    axes[1, 1].plot(df['episode'], df['carbon_emissions'], color='purple')
    axes[1, 1].set_title("Carbon Emissions")
    
    for ax in axes.flatten():
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_curves.png"))
    plt.close()
    
    # 3. Losses
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(df['episode'], df['actor_loss'])
    axes[0].set_title("Actor Loss")
    
    axes[1].plot(df['episode'], df['critic_loss'])
    axes[1].set_title("Critic Loss")
    
    for ax in axes.flatten():
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curves.png"))
    plt.close()
    
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="outputs/mappo", help="Directory containing training_curves.csv")
    args = parser.parse_args()
    plot_training_curves(args.log_dir)

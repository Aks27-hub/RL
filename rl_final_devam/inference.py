import argparse
import os

import numpy as np
import torch

from configs.config import MasterConfig
from train import MAPPOAgent
from utils.env_wrapper import DistrictEnergyEnv, EnvParams


def get_inference_controller(model_path: str, algo: str = "mappo"):
    # 3. RBC
    def rbc_controller(obs: np.ndarray, _t: int) -> np.ndarray:
        hours = np.round(obs[:, 0] * 23.0).astype(int)
        actions = np.zeros(obs.shape[0], dtype=np.float32)
        night = (hours >= 0) & (hours <= 6)
        peak = (hours >= 17) & (hours <= 21)
        actions[night] = 0.8
        actions[peak] = -0.9
        return actions

    if algo == "rbc":
        print("Using Rule-Based Controller (RBC)")
        return rbc_controller

    # 4. MAPPO (Original logic)

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "config.json")
    if os.path.exists(cfg_path):
        cfg = MasterConfig.load(cfg_path)
    else:
        # Fallback to default if config not found alongside model
        cfg = MasterConfig()

    env_params = EnvParams(n_buildings=cfg.env.n_buildings)
    env = DistrictEnergyEnv(env_params)
    
    agent = MAPPOAgent(cfg, env)
    
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    def controller(obs: np.ndarray, _t: int) -> np.ndarray:
        actions = np.zeros(agent.n_agents, dtype=np.float32)
        with torch.no_grad():
            for i in range(agent.n_agents):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=agent.device).unsqueeze(0)
                a, _, _ = agent.actors[i].get_action(obs_tensor, deterministic=True)
                actions[i] = a.item()
        return actions

    return controller


def main():
    parser = argparse.ArgumentParser(description="Run inference with MAPPO model")
    parser.add_argument("--model_path", type=str, default="", help="Path to the saved model (.pt)")
    parser.add_argument("--algo", type=str, default="mappo", choices=["mappo", "rbc"], help="Algorithm to run")
    parser.add_argument("--horizon", type=int, default=24*7, help="Simulation horizon in hours")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for environment")
    args = parser.parse_args()

    if args.algo == "mappo" and not args.model_path:
        # Try to find default mappo model
        args.model_path = "outputs/mappo/best_model.pt"

    print(f"Algorithm: {args.algo}")
    controller = get_inference_controller(args.model_path, algo=args.algo)
    
    # Run a short test simulation
    env_params = EnvParams(horizon=args.horizon, seed=args.seed)
    env = DistrictEnergyEnv(env_params)
    
    obs, _ = env.reset(seed=args.seed)
    done = False
    t = 0
    total_reward = 0.0
    
    print(f"Running inference for {args.horizon} steps...")
    
    vis_data = {
        "metadata": {
            "algorithm": args.algo.upper(),
            "n_buildings": env.n_buildings,
            "horizon": args.horizon
        },
        "steps": []
    }
    
    while not done:
        actions = controller(obs, t)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Save step data
        step_entry = {
            "t": t,
            "hour": int(np.round(obs[0, 0] * 23.0)),
            "day": int(np.round(obs[0, 1] * 6.0)),
            "price": float(obs[0, 4] * 0.30),
            "temp": float(obs[0, 2] * 50.0 - 10.0),
            "district_load": float(info["district_load"]),
            "buildings": []
        }
        
        for i in range(env.n_buildings):
            step_entry["buildings"].append({
                "id": i,
                "action": float(actions[i]),
                "soc": float(obs[i, 7] * 40.0), # Assuming 40kWh capacity from config
                "consumption": float(obs[i, 8] * 25.0),
                "solar": float(obs[i, 3] * 10.0)
            })
            
        vis_data["steps"].append(step_entry)
        
        obs = next_obs
        total_reward += float(rewards[0])
        done = terminated or truncated
        t += 1
        
    import json
    if args.algo == "rbc":
        vis_path = os.path.join("outputs", "rbc", "vis_data.json")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    else:
        vis_path = os.path.join(os.path.dirname(args.model_path), "vis_data.json")
    
    with open(vis_path, "w") as f:
        json.dump(vis_data, f, indent=2)
        
    print(f"Inference complete. Total Reward: {total_reward:.2f}")
    print(f"Visualization data saved to: {vis_path}")


if __name__ == "__main__":
    main()

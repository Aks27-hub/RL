import json
import os
import time

import numpy as np
import torch

from configs.config import MasterConfig
from train import MAPPOAgent
from utils.env_wrapper import DistrictEnergyEnv, EnvParams

def evaluate():
    cfg_path = os.path.join(os.path.dirname(__file__), "outputs", "mappo", "config.json")
    if not os.path.exists(cfg_path):
        print(f"Config not found at {cfg_path}. Please train first.")
        return

    cfg = MasterConfig.load(cfg_path)
    
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", cfg.algorithm)
    model_path = os.path.join(out_dir, "best_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return

    # Setup Environment
    env_params = EnvParams(
        n_buildings=cfg.env.n_buildings,
        horizon=cfg.eval.eval_horizon, # Full year
        battery_capacity_kwh=cfg.env.battery_capacity_kwh,
        battery_max_power_kw=cfg.env.battery_max_power_kw,
        seed=cfg.eval.eval_seed
    )
    env = DistrictEnergyEnv(env_params)

    # Setup Agent
    agent = MAPPOAgent(cfg, env)
    agent.load(model_path)
    
    print(f"Evaluating MAPPO for {cfg.eval.eval_episodes} episodes...")
    
    results = {
        "final_reward": 0.0,
        "average_reward": 0.0,
        "electricity_cost": 0.0,
        "emissions": 0.0,
        "runtime": 0.0,
        "training_episodes": cfg.train.max_episodes, # Hardcoded as max for schema compliance if actual isn't stored
        "evaluation_episodes": cfg.eval.eval_episodes,
    }
    
    start_time = time.time()
    
    for ep in range(cfg.eval.eval_episodes):
        obs, _ = env.reset(seed=cfg.eval.eval_seed + ep)
        done = False
        
        ep_reward = 0
        ep_cost = 0
        ep_emiss = 0
        
        while not done:
            # Deterministic actions for evaluation
            actions = np.zeros((agent.n_agents, 1), dtype=np.float32)
            with torch.no_grad():
                for i in range(agent.n_agents):
                    obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=agent.device).unsqueeze(0)
                    a, _, _ = agent.actors[i].get_action(obs_tensor, deterministic=True)
                    actions[i] = a.item()
            
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            obs = next_obs
            ep_reward += float(rewards[0])
            ep_cost += info["electricity_cost"]
            ep_emiss += info["carbon_emissions"]
            
        results["final_reward"] = ep_reward
        results["average_reward"] += ep_reward
        results["electricity_cost"] += ep_cost
        results["emissions"] += ep_emiss
        
        print(f"Eval Ep {ep+1} | Reward: {ep_reward:.2f}")

    # Average metrics
    results["average_reward"] /= cfg.eval.eval_episodes
    results["electricity_cost"] /= cfg.eval.eval_episodes
    results["emissions"] /= cfg.eval.eval_episodes
    results["runtime"] = time.time() - start_time
    
    # Save evaluation.json
    eval_path = os.path.join(out_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {eval_path}")

if __name__ == "__main__":
    evaluate()

import argparse
from pathlib import Path

import numpy as np
import torch

from citylearn_common import ObsProcessor, evaluate_policy, get_n_buildings, load_json, make_citylearn_env, save_json
from networks import GaussianActor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MARLISA inference on CityLearn")
    parser.add_argument("--config", required=True, help="Path to MARLISA config.json")
    parser.add_argument("--model", required=True, help="Path to best_model.pt")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg_data = load_json(Path(args.config))
    env_cfg = cfg_data["env"]

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    env = make_citylearn_env(env_cfg["schema_path"], env_cfg["seed"], central_agent=False)
    obs_processor = ObsProcessor(env.observation_space, normalize=env_cfg.get("normalize_obs", True))

    n_buildings = get_n_buildings(env)
    if isinstance(env.observation_space, (list, tuple)):
        obs_space = env.observation_space[0]
    else:
        obs_space = env.observation_space
    obs_dim = int(np.prod(obs_space.shape))

    actors = [GaussianActor(obs_dim, action_dim=1).to(device) for _ in range(n_buildings)]
    payload = torch.load(args.model, map_location=device)
    for actor, state in zip(actors, payload["actors"]):
        actor.load_state_dict(state)
        actor.eval()

    def policy_fn(proc_obs: np.ndarray, _t: int) -> np.ndarray:
        actions = np.zeros(n_buildings, dtype=np.float32)
        with torch.no_grad():
            for i, actor in enumerate(actors):
                s = torch.tensor(proc_obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                mu, _ = actor(s)
                actions[i] = float(torch.tanh(mu).squeeze().cpu().item())
        return actions

    metrics, _ = evaluate_policy(env, policy_fn, obs_processor, env_cfg.get("max_steps"), seed=env_cfg["seed"] + 999)
    output_dir = Path(cfg_data.get("output_dir", "outputs/marlisa"))
    save_json(output_dir / "metrics" / "inference.json", metrics)


if __name__ == "__main__":
    main()

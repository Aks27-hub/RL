import argparse
from pathlib import Path

import numpy as np

from citylearn_common import CityLearnEnvConfig, ObsProcessor, ensure_dir, evaluate_policy, make_citylearn_env, save_json
from rbc_citylearn import RBCConfig, RBCPolicy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RBC on CityLearn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/rbc")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    env_cfg = CityLearnEnvConfig(seed=args.seed, max_steps=args.max_steps)
    env = make_citylearn_env(env_cfg.schema_path, env_cfg.seed, central_agent=False)
    obs_processor = ObsProcessor(env.observation_space, normalize=env_cfg.normalize_obs)

    observation_names = getattr(env, "observation_names", None)
    if callable(observation_names):
        observation_names = observation_names()
    if isinstance(observation_names, list) and observation_names and isinstance(observation_names[0], list):
        observation_names = observation_names[0]
    if observation_names is None:
        observation_names = [f"obs_{i}" for i in range(env.observation_space[0].shape[0])]

    policy = RBCPolicy(observation_names, RBCConfig())

    def policy_fn(proc_obs: np.ndarray, _t: int) -> np.ndarray:
        actions = np.zeros(len(proc_obs), dtype=np.float32)
        for i in range(len(proc_obs)):
            actions[i] = policy.act(proc_obs[i])
        return actions

    metrics, _ = evaluate_policy(env, policy_fn, obs_processor, env_cfg.max_steps, seed=env_cfg.seed + 999)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "metrics")
    save_json(output_dir / "evaluation.json", metrics)


if __name__ == "__main__":
    main()

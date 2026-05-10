import argparse

import torch

from citylearn_common import CityLearnEnvConfig
from marlisa_citylearn import MARLISAConfig, MARLISATrainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MARLISA on CityLearn")
    parser.add_argument("--schema", required=True, help="Path to CityLearn schema.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/marlisa")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    env_cfg = CityLearnEnvConfig(schema_path=args.schema, seed=args.seed, max_steps=args.max_steps)
    cfg = MARLISAConfig(env=env_cfg, episodes=args.episodes, output_dir=args.output_dir)

    trainer = MARLISATrainer(cfg, device)
    trainer.train()
    trainer.evaluate(save_json_out=True)


if __name__ == "__main__":
    main()

import argparse

import torch

from marlisa_citylearn import load_trainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MARLISA on CityLearn")
    parser.add_argument("--config", required=True, help="Path to MARLISA config.json")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    trainer = load_trainer(args.config, device)
    trainer.evaluate(save_json_out=True)


if __name__ == "__main__":
    main()

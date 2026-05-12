# RL: Multi-Agent Reinforcement Learning for CityLearn and Energy Management

This repository contains implementations and experiments for various multi-agent reinforcement learning (MARL) algorithms applied to energy management environments, including the CityLearn Challenge. The project is organized to support research, benchmarking, and visualization of MARL methods for smart grid and building control tasks.

## Project Structure

```
RL/
├── ddqn_masac/         # Double DQN and MASAC algorithms and results
│   ├── ddqn/           # DDQN agent, training, evaluation
│   ├── masac/          # MASAC agent, training, evaluation
│   ├── outputs/        # Model checkpoints, logs, metrics, plots
│   └── shared/         # Shared utilities (env wrappers, metrics, plotting)
├── mappo/              # MAPPO algorithm, configs, models, outputs, utils
├── marlisa_rbc/        # MARLISA and RBC baselines for CityLearn
│   ├── marlisa_citylearn.py  # Main MARLISA agent code
│   ├── rbc_citylearn.py      # Rule-based controller baseline
│   ├── ...             # Training, evaluation, plotting scripts
│   └── outputs/        # Results for MARLISA and RBC
├── env/                # Environment wrappers and requirements
│   ├── env_ddqn_masac.py
│   ├── env_mappo.py
│   ├── env_marlisa_rbc.py
│   └── requirements_*.txt
└── README.md           # Project overview (this file)
```

## Main Ideas

- **Multi-Agent RL**: Implements and compares state-of-the-art MARL algorithms (DDQN, MASAC, MAPPO, MARLISA) for energy management tasks.
- **CityLearn Challenge**: Provides agents and evaluation for the CityLearn environment, a benchmark for building energy control.
- **Baselines**: Includes rule-based controllers (RBC) for comparison.
- **Reproducibility**: Scripts for training, evaluation, inference, and plotting results.
- **Visualization**: Tools for plotting training curves, evaluation metrics, and generating dashboards.

## Getting Started

1. Clone the repository.
2. Install dependencies for the desired algorithm/environment (see `env/requirements_*.txt`).
3. Run training or evaluation scripts in the respective folders.

## Folders

- `ddqn_masac/`: Double DQN and MASAC agents, outputs, and shared utilities.
- `mappo/`: MAPPO agent, configs, models, outputs, and utilities.
- `marlisa_rbc/`: MARLISA and RBC baselines for CityLearn, with scripts and results.
- `env/`: Environment wrappers and requirements files for each setup.

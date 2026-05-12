
# RL: Multi-Agent Reinforcement Learning for CityLearn and Energy Management

## Objective

The primary objective of this project is to advance the development and evaluation of multi-agent reinforcement learning (MARL) algorithms for intelligent energy management in smart grids and building environments. By leveraging the CityLearn Challenge and other energy management scenarios, this project aims to:

- Develop scalable and robust MARL agents capable of optimizing energy consumption, storage, and distribution across multiple buildings or agents.
- Benchmark and compare the performance of different MARL algorithms and rule-based baselines in realistic, dynamic environments.
- Provide reproducible research tools and clear visualizations to facilitate understanding and further innovation in the field of energy-efficient control.

## Approach

To achieve these objectives, the project is structured as follows:

- Implements several state-of-the-art MARL algorithms (DDQN, MASAC, MAPPO, MARLISA) and integrates them with the CityLearn environment and other wrappers.
- Provides modular code for training, evaluation, and inference, allowing easy experimentation and extension.
- Includes rule-based controllers (RBC) as baselines for fair comparison.
- Offers comprehensive logging, metrics, and visualization tools to analyze agent performance and learning dynamics.
- Ensures reproducibility through organized scripts, configuration files, and environment requirements.

Through this approach, the project serves as a platform for both research and practical application of MARL in energy management, supporting the community in developing smarter, more sustainable control strategies.

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


Follow these steps to set up and run experiments with this project:

1. **Clone the repository**
	```bash
	git clone <repository-url>
	cd RL
	```

2. **Set up a Python environment**
	- It is recommended to use a virtual environment (venv, conda, or similar):
	  ```bash
	  python -m venv venv
	  source venv/bin/activate  # On Windows: venv\Scripts\activate
	  ```

3. **Install dependencies**
	- Choose the requirements file for your target algorithm/environment:
	  ```bash
	  pip install -r env/requirements_ddqn_masac.txt
	  # or
	  pip install -r env/requirements_mappo.txt
	  # or
	  pip install -r env/requirements_marlisa_rbc.txt
	  ```

4. **Run training or evaluation scripts**
	- Navigate to the relevant folder and run the desired script. For example:
	  ```bash
	  # Train MARLISA agent
	  cd marlisa_rbc
	  python train_marlisa_citylearn.py --episodes 50 --output-dir outputs/marlisa

	  # Train MAPPO agent
	  cd ../mappo
	  python train.py --config configs/config.py

	  # Train DDQN agent
	  cd ../ddqn_masac/ddqn
	  python train.py --config config.json
	  ```

5. **Visualize results**
	- Use the provided plotting scripts or notebooks to visualize training curves and evaluation metrics. For example:
	  ```bash
	  cd ../../plot_results.py
	  python plot_results.py --input-dir ddqn_masac/outputs/ddqn
	  ```

6. **(Optional) Modify configurations**
	- Edit the config files in each folder to change hyperparameters, environment settings, or output locations.

For more details, see the README files in each subfolder.

## Folders

- `ddqn_masac/`: Double DQN and MASAC agents, outputs, and shared utilities.
- `mappo/`: MAPPO agent, configs, models, outputs, and utilities.
- `marlisa_rbc/`: MARLISA and RBC baselines for CityLearn, with scripts and results.
- `env/`: Environment wrappers and requirements files for each setup.


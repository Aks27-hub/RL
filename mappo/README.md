# Multi-Agent PPO (MAPPO) for CityLearn

This repository contains a complete, modular, and merge-friendly implementation of Multi-Agent Proximal Policy Optimization (MAPPO) for the CityLearn reinforcement learning benchmark.

The implementation is designed to be fully compatible with the shared project evaluation, inference, and benchmarking pipelines (alongside DDQN, MASAC, MARLISA, and RBC).

## Features
- **Centralized Training, Decentralized Execution (CTDE)**: Individual actors for each building, with a shared centralized critic that observes the global state during training.
- **Generalized Advantage Estimation (GAE)**
- **Running Normalization** for rewards
- **Parallel Rollout Collection** internally (across buildings)
- **Fast Training**: Optimized to complete within the 1-2 hour strict budget by using lightweight networks and efficient batched tensor operations.
- **Unified Outputs**: Automatically structures checkpoints, logs, and evaluation metrics exactly as required by the shared pipeline (`outputs/mappo/...`).

## Repository Structure

```
├── configs/
│   └── config.py        # Centralized hyperparameter and settings management
├── models/
│   └── networks.py      # Actor and Centralized Critic architectures
├── utils/
│   ├── env_wrapper.py   # Shared DistrictEnergyEnv with global_obs() support
│   ├── buffers.py       # MAPPO Rollout Buffer with GAE
│   ├── logger.py        # Unified CSV and Tensorboard logging
│   └── plots.py         # Utilities to plot training curves
├── train.py             # Main training loop
├── evaluate.py          # Evaluation script generating evaluation.json
├── inference.py         # Inference script for benchmarking
├── requirements.txt     # Dependencies
└── README.md
```

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the MAPPO Agents
```bash
python train.py
```
This will run the training loop and save outputs, checkpoints, and logs to `outputs/mappo/`.

### 3. Plot Training Curves
```bash
python utils/plots.py --log_dir outputs/mappo
```

### 4. Evaluate the Trained Model
```bash
python evaluate.py
```
This evaluates the `best_model.pt` on the full 365-day test environment and saves `evaluation.json`.

### 5. Run Inference
The `inference.py` script matches the shared signature for future ensembling.
```bash
python inference.py --model_path outputs/mappo/best_model.pt
```

## Output Structure Compliance

After training and evaluation, the following structure will be generated, ready for merging:

```
outputs/
└── mappo/
    ├── config.json
    ├── training_curves.csv
    ├── evaluation.json
    ├── best_model.pt
    ├── checkpoints/
    │   ├── model_ep25.pt
    │   └── ...
    └── logs/
        ├── training.log
        └── tensorboard/
```

## Hyperparameters
All hyperparameters are centrally managed in `configs/config.py`.
Recommended defaults for short training runs (1-2h limit) are already configured:
- 300 maximum episodes (with early stopping)
- 30-day training episodes
- Hidden dimensions: Actor (256), Central Critic (512)
- PPO Epochs: 8
- GAE Lambda: 0.95

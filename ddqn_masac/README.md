# District Energy Management — Multi-Agent RL Benchmark

Reinforcement learning implementations for the [CityLearn](https://www.citylearn.net/) environment using **Double DQN (DDQN)** and **Multi-Agent Soft Actor-Critic (MASAC)**.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

**Python ≥ 3.9** and **PyTorch ≥ 2.0** are required.

---

## Training

### Double DQN
```bash
python ddqn/train.py --episodes 500 --seed 42 --device auto
```

### Multi-Agent SAC
```bash
python masac/train.py --episodes 500 --seed 42 --device auto
```

#### CLI Options (same for both)
| Argument | Default | Description |
|---|---|---|
| `--episodes` | 500 | Number of training episodes |
| `--seed` | 42 | Random seed |
| `--device` | auto | `auto`, `cpu`, or `cuda` |
| `--schema` | `citylearn_challenge_2022_phase_1` | CityLearn schema |
| `--lr` | 3e-4 | Learning rate |
| `--batch_size` | 256 | Replay buffer batch size |
| `--buffer_size` | 100000 | Replay buffer capacity |
| `--gamma` | 0.99 | Discount factor |
| `--eval_interval` | 25 | Episodes between evaluations |
| `--checkpoint_interval` | 50 | Episodes between checkpoints |
| `--eval_episodes` | 5 | Number of evaluation episodes |
| `--early_stopping_patience` | 50 | Eval cycles without improvement |

---

## Evaluation

```bash
python ddqn/evaluate.py --checkpoint outputs/ddqn/best_model.pt --seed 42
python masac/evaluate.py --checkpoint outputs/masac/best_model.pt --seed 42
```

---

## Inference

Run a single episode with step-by-step output:
```bash
python ddqn/inference.py --checkpoint outputs/ddqn/best_model.pt --render
python masac/inference.py --checkpoint outputs/masac/best_model.pt --render
```

---

## Plotting

```bash
# Plot individual algorithm curves
python plot_results.py

# Compare specific algorithms
python plot_results.py --algos ddqn masac
```

---

## TensorBoard

```bash
tensorboard --logdir outputs/
```

---

## Expected Runtime

| Hardware | DDQN (500 ep) | MASAC (500 ep) |
|---|---|---|
| 8-core CPU, 16 GB RAM | ~30–45 min | ~45–60 min |
| CPU + GPU (≤6 GB VRAM) | ~20–30 min | ~30–45 min |

Both algorithms are designed to complete within **≤ 2 hours total** on consumer hardware.

---

## Output Structure

Each algorithm produces the following layout under `outputs/<algo>/`:

```
outputs/
├── ddqn/
│   ├── checkpoints/          # epoch_N.pt + best_model.pt
│   ├── logs/                 # TensorBoard event files
│   ├── metrics/              # metrics_log.jsonl
│   ├── config.json           # Full configuration
│   ├── training_curves.csv   # Per-episode metrics (8 columns)
│   ├── evaluation.json       # Final evaluation results
│   └── best_model.pt         # Best checkpoint
└── masac/
    ├── (same structure)
```

### `training_curves.csv` columns
```
episode,reward,electricity_cost,carbon_emissions,ramping,load_factor,daily_peak,comfort_violation
```

### `evaluation.json` schema
```json
{
  "algo": "ddqn",
  "seed": 42,
  "mean_reward": 0.0,
  "std_reward": 0.0,
  "mean_electricity_cost": 0.0,
  "mean_carbon_emissions": 0.0,
  "mean_ramping": 0.0,
  "mean_load_factor": 0.0,
  "mean_daily_peak": 0.0,
  "mean_comfort_violation": 0.0,
  "num_eval_episodes": 5,
  "best_checkpoint": "outputs/ddqn/checkpoints/best_model.pt",
  "training_wall_time_seconds": 0.0
}
```

---

## Merge Notes for Team Members (MAPPO, MARLISA, RBC)

### What to import from `shared/`

All algorithms **must** use these shared utilities — no copy-pasting:

```python
from shared.utils import set_seed, get_device, save_config, load_config, make_output_dirs
from shared.env_wrapper import make_env
from shared.replay_buffer import ReplayBuffer
from shared.metrics import extract_metrics, csv_header, metrics_to_csv_row
from shared.evaluation import evaluate
from shared.plotting import plot_training_curves
```

### Agent interface contract

Every agent must implement:
```python
class YourAgent:
    def act(self, obs, deterministic=False) -> list:
        """obs: list of per-agent observation arrays.
        Returns: list of per-agent action arrays."""
        ...
    
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

This ensures `shared/evaluation.py` works via duck typing.

### CLI contract

Every `train.py` must accept:
```bash
python <algo>/train.py --episodes N --seed S --device D
```

Every `evaluate.py` must accept:
```bash
python <algo>/evaluate.py --checkpoint <path> --seed S
```

### Config schema

All `config.json` files must use the exact top-level keys defined in the spec. Algorithm-specific parameters go inside `algo_specific`.

### Output directories

Use `make_output_dirs("<algo_name>")` — never create directories manually.

---

## Reproducibility

- `set_seed(seed)` is called before environment creation, model initialization, and training
- Environment is seeded via `make_env(schema, seed)`
- `config.json` is saved before the first training step
- Checkpoint filenames follow `epoch_{episode}.pt`
- All random state (numpy, torch, python random) is seeded

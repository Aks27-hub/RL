"""
configs/config.py
=================
Central, serialisable configuration for the MAPPO pipeline.

All hyper-parameters live here so that train.py, evaluate.py and
inference.py can stay algorithm-agnostic and merge-friendly.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List


# ── Environment ───────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    n_buildings: int = 5
    horizon: int = 24 * 30          # 30-day training episode (speed ↑)
    eval_horizon: int = 24 * 365    # full-year evaluation
    battery_capacity_kwh: float = 40.0
    battery_max_power_kw: float = 10.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    reward_scale: float = 0.2
    seed: int = 42


# ── MAPPO Agent ───────────────────────────────────────────────────────────────
@dataclass
class MAPPOConfig:
    # Architecture
    hidden_dim: int = 256           # neurons per hidden layer (actor + critic)
    central_hidden_dim: int = 512   # centralised critic hidden size

    # PPO clip
    clip_eps: float = 0.2

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Optimisation
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 0.5

    # PPO epochs & batch
    update_epochs: int = 8
    minibatch_size: int = 512

    # Loss coefficients
    entropy_coef: float = 0.02
    value_coef: float = 0.5

    # Rollout length (steps collected before each PPO update)
    rollout_steps: int = 1024


# ── Training ──────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Budget: ~ 1-2 h on a mid-range GPU / CPU
    max_episodes: int = 300          # hard ceiling
    max_seconds: float = 5400.0     # 90-minute wall-clock budget

    # Checkpointing
    checkpoint_every: int = 25       # episodes
    log_every: int = 5               # episodes

    # Reproducibility
    seed: int = 42

    # Device: "auto" → cuda if available, else cpu
    device: str = "auto"

    # Early stopping
    early_stop_patience: int = 40    # episodes without improvement
    early_stop_delta: float = 0.005  # minimum relative improvement

    # Reward normalisation (running mean/std)
    reward_norm: bool = True
    reward_norm_clip: float = 10.0


# ── Evaluation ────────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    eval_episodes: int = 3
    eval_seed: int = 999
    eval_horizon: int = 24 * 365


# ── Master config ─────────────────────────────────────────────────────────────
@dataclass
class MasterConfig:
    algorithm: str = "mappo"
    env: EnvConfig = field(default_factory=EnvConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MasterConfig":
        with open(path) as f:
            d = json.load(f)
        cfg = cls()
        cfg.env = EnvConfig(**d["env"])
        cfg.mappo = MAPPOConfig(**d["mappo"])
        cfg.train = TrainConfig(**d["train"])
        cfg.eval = EvalConfig(**d["eval"])
        return cfg


def get_default_config() -> MasterConfig:
    return MasterConfig()

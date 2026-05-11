"""
shared/utils.py — Seed management, device selection, config I/O, output directory creation.

All algorithms import these utilities to ensure reproducibility and consistent output structure.
"""

import os
import json
import random
import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Must be called before env creation, model init, and training loop.
    Seeds: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all devices).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN for reproducibility (slight perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """Parse device string and return a torch.device.

    Args:
        device_str: One of 'auto', 'cpu', 'cuda', 'cuda:N', 'mps'.
                    'auto' selects CUDA if available, else CPU.

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def save_config(config: Dict[str, Any], path: str) -> None:
    """Write config dict to a JSON file.

    Args:
        config: Configuration dictionary matching the standard schema.
        path: File path for the JSON output.
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


def load_config(path: str) -> Dict[str, Any]:
    """Load config dict from a JSON file.

    Args:
        path: Path to config.json.

    Returns:
        Configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_output_dirs(algo: str, base: str = "outputs") -> Dict[str, str]:
    """Create the standard output directory tree for an algorithm.

    Layout created:
        outputs/<algo>/
        ├── checkpoints/
        ├── logs/
        ├── metrics/

    Args:
        algo: Algorithm name (e.g. 'ddqn', 'masac').
        base: Base output directory.

    Returns:
        Dict mapping directory purpose to absolute path:
        {'root', 'checkpoints', 'logs', 'metrics'}
    """
    root = os.path.join(base, algo)
    dirs = {
        "root": root,
        "checkpoints": os.path.join(root, "checkpoints"),
        "logs": os.path.join(root, "logs"),
        "metrics": os.path.join(root, "metrics"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

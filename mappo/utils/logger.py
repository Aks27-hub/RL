"""
utils/logger.py
===============
Unified, merge-compatible logging utilities for the MAPPO pipeline.

Features
--------
* CSV metrics writer (training_curves.csv)
* TensorBoard SummaryWriter wrapper (falls back gracefully if not installed)
* Console logger with episode / step tagging
* Standardised metric dict keys shared across all algorithms
"""

import csv
import logging
import os
import time
from typing import Any, Dict, List, Optional

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ── Console logger ────────────────────────────────────────────────────────────
def get_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                                datefmt="%H:%M:%S")
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    return logger


# ── CSV writer ────────────────────────────────────────────────────────────────
class CSVWriter:
    """Appends one row per episode/step to a CSV file."""

    FIELDNAMES: List[str] = [
        "episode",
        "step",
        "elapsed_s",
        "reward",
        "electricity_cost",
        "carbon_emissions",
        "ramping",
        "load_factor",
        "daily_peak",
        "comfort_violation",
        "actor_loss",
        "critic_loss",
        "entropy",
        "kl_approx",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file,
                                      fieldnames=self.FIELDNAMES,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: Dict[str, Any]):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ── TensorBoard wrapper ───────────────────────────────────────────────────────
class TBLogger:
    """
    Thin wrapper around SummaryWriter.
    All log_* calls are no-ops if TensorBoard is not installed.
    """

    def __init__(self, log_dir: str):
        self.enabled = _TB_AVAILABLE
        if self.enabled:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int):
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_dict(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.log_scalar(tag, v, step)

    def close(self):
        if self.enabled and self.writer is not None:
            self.writer.close()


# ── Unified MetricsLogger ────────────────────────────────────────────────────
class MetricsLogger:
    """Combines CSV, TensorBoard, and console logging."""

    def __init__(self, output_dir: str, algorithm: str = "mappo"):
        self.output_dir   = output_dir
        self.algorithm    = algorithm
        self.start_time   = time.time()

        metrics_dir = os.path.join(output_dir, "metrics")
        log_dir     = os.path.join(output_dir, "logs")
        tb_dir      = os.path.join(log_dir, "tensorboard")

        self.csv    = CSVWriter(os.path.join(output_dir, "training_curves.csv"))
        self.tb     = TBLogger(tb_dir)
        self.logger = get_logger(algorithm, log_dir)

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def log_episode(self, episode: int, step: int, metrics: Dict[str, Any]):
        row = {
            "episode":           episode,
            "step":              step,
            "elapsed_s":         round(self.elapsed(), 1),
            **metrics,
        }
        self.csv.write(row)
        self.tb.log_dict(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            step=episode,
            prefix="train",
        )

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def close(self):
        self.csv.close()
        self.tb.close()

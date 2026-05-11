from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class CityLearnEnvConfig:
    schema_path: str
    seed: int = 42
    max_steps: Optional[int] = None
    normalize_obs: bool = True


def make_citylearn_env(schema_path: str, seed: int, central_agent: bool = False):
    try:
        from citylearn.citylearn import CityLearnEnv
    except Exception:  # pragma: no cover - fallback import
        from citylearn import CityLearnEnv

    env = CityLearnEnv(schema=schema_path, central_agent=central_agent)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
        if hasattr(env, "seed"):
            env.seed(seed)
    return env


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path: Path, cfg: Any):
    if hasattr(cfg, "__dataclass_fields__"):
        payload = asdict(cfg)
    else:
        payload = dict(cfg)
    save_json(path, payload)


def to_np_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    return np.array(obs, dtype=np.float32)


class ObsProcessor:
    def __init__(self, obs_space: Any, normalize: bool = True):
        self.normalize = normalize
        self.low, self.high = self._extract_bounds(obs_space)

    def _extract_bounds(self, obs_space: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if isinstance(obs_space, (list, tuple)) and len(obs_space) > 0:
            obs_space = obs_space[0]
        low = getattr(obs_space, "low", None)
        high = getattr(obs_space, "high", None)
        if low is None or high is None:
            return None, None
        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)

    def transform(self, obs: Any) -> np.ndarray:
        arr = to_np_obs(obs)
        if not self.normalize or self.low is None or self.high is None:
            return arr
        low = np.broadcast_to(self.low, arr.shape)
        high = np.broadcast_to(self.high, arr.shape)
        span = np.where(np.isfinite(high - low), high - low, 1.0)
        scaled = (arr - low) / np.clip(span, 1e-6, None)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 2.0 - 1.0).astype(np.float32)


def get_n_buildings(env: Any) -> int:
    if hasattr(env, "n_buildings"):
        return int(env.n_buildings)
    if hasattr(env, "buildings"):
        return len(env.buildings)
    return len(env.observation_space)


def _aggregate_info(info: Any) -> Dict[str, float]:
    if isinstance(info, dict):
        merged = {k: float(v) for k, v in info.items() if _is_number(v)}
        for nested_key in ["costs", "metrics"]:
            nested = info.get(nested_key)
            if isinstance(nested, dict):
                for k, v in nested.items():
                    if _is_number(v):
                        merged[k] = merged.get(k, 0.0) + float(v)
        return merged
    if isinstance(info, (list, tuple)):
        merged: Dict[str, float] = {}
        for item in info:
            if isinstance(item, dict):
                for k, v in item.items():
                    if _is_number(v):
                        merged[k] = merged.get(k, 0.0) + float(v)
                for nested_key in ["costs", "metrics"]:
                    nested = item.get(nested_key)
                    if isinstance(nested, dict):
                        for k, v in nested.items():
                            if _is_number(v):
                                merged[k] = merged.get(k, 0.0) + float(v)
        return merged
    return {}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number))


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.total_reward = 0.0
        self.totals: Dict[str, float] = {
            "electricity_cost": 0.0,
            "carbon_emissions": 0.0,
            "ramping": 0.0,
            "load_factor": 0.0,
            "daily_peak": 0.0,
            "comfort_violation": 0.0,
        }
        self.district_load: List[float] = []

    def step(self, reward: float, info: Any):
        self.steps += 1
        self.total_reward += float(reward)
        info_dict = _aggregate_info(info)
        self._update_metric(info_dict, ["electricity_cost", "cost"], "electricity_cost")
        self._update_metric(info_dict, ["carbon_emissions", "carbon_intensity"], "carbon_emissions")
        self._update_metric(info_dict, ["ramping"], "ramping")
        self._update_metric(info_dict, ["load_factor"], "load_factor")
        self._update_metric(info_dict, ["daily_peak"], "daily_peak")
        self._update_metric(info_dict, ["comfort_violation", "comfort"], "comfort_violation")
        if "district_load" in info_dict:
            self.district_load.append(float(info_dict["district_load"]))

    def _update_metric(self, info: Dict[str, float], keys: Iterable[str], target: str):
        for key in keys:
            if key in info:
                self.totals[target] += float(info[key])
                return

    def as_dict(self) -> Dict[str, float]:
        if self.steps == 0:
            return {"reward": 0.0, **self.totals}
        metrics = {"reward": self.total_reward}
        metrics.update({k: float(v) for k, v in self.totals.items()})
        metrics["reward_per_step"] = self.total_reward / float(self.steps)
        return metrics


def evaluate_policy(
    env: Any,
    policy_fn,
    obs_processor: ObsProcessor,
    max_steps: Optional[int],
    seed: int,
) -> Tuple[Dict[str, float], List[float]]:
    obs = _reset_env(env, seed)
    done = False
    steps = 0
    tracker = MetricTracker()

    while not done:
        proc_obs = obs_processor.transform(obs)
        actions = _format_actions(policy_fn(proc_obs, steps))
        step_out = env.step(actions)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_obs, rewards, terminated, truncated, info = step_out
        else:
            next_obs, rewards, done, info = step_out
            terminated = bool(done)
            truncated = False
        tracker.step(float(np.mean(rewards)), info)
        obs = next_obs
        done = bool(terminated or truncated)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return tracker.as_dict(), tracker.district_load


def _reset_env(env: Any, seed: int):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()
        if hasattr(env, "seed"):
            env.seed(seed)
    if isinstance(result, tuple) and len(result) >= 1:
        return result[0]
    return result


def _format_actions(actions: Any):
    if isinstance(actions, np.ndarray):
        if actions.ndim == 1:
            return [[float(a)] for a in actions]
        return actions.tolist()
    if isinstance(actions, list):
        if len(actions) == 0:
            return actions
        if isinstance(actions[0], (float, int, np.floating, np.integer)):
            return [[float(a)] for a in actions]
    return actions

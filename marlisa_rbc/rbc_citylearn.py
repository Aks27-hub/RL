from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class RBCConfig:
    price_high: float = 0.3
    solar_high: float = 0.2
    load_high: float = 0.3
    action_charge: float = 0.7
    action_discharge: float = -0.7
    action_idle: float = 0.0


class RBCPolicy:
    def __init__(self, observation_names: List[str], cfg: RBCConfig):
        self.cfg = cfg
        self.index = self._build_index(observation_names)

    def _build_index(self, names: List[str]) -> Dict[str, int]:
        def pick(candidates: List[str]) -> int:
            for name in candidates:
                if name in names:
                    return names.index(name)
            return -1

        return {
            "price": pick(["electricity_pricing", "price", "energy_price"]),
            "solar": pick(["solar_generation", "solar_irradiance", "direct_solar_irradiance", "diffuse_solar_irradiance"]),
            "load": pick(["non_shiftable_load", "net_electricity_consumption", "total_demand"]),
        }

    def act(self, obs: np.ndarray) -> float:
        price = self._read(obs, "price")
        solar = self._read(obs, "solar")
        load = self._read(obs, "load")

        if price > self.cfg.price_high or load > self.cfg.load_high:
            return self.cfg.action_discharge
        if solar > self.cfg.solar_high and load <= self.cfg.load_high:
            return self.cfg.action_charge
        return self.cfg.action_idle

    def _read(self, obs: np.ndarray, key: str) -> float:
        idx = self.index.get(key, -1)
        if idx < 0 or idx >= obs.shape[-1]:
            return 0.0
        return float(obs[idx])

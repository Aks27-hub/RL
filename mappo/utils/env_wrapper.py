"""
utils/env_wrapper.py
====================
Thin wrapper around DistrictEnergyEnv that:
  * mirrors the same DistrictEnergyEnv from the shared project codebase
  * exposes a unified reset / step API compatible with all algorithms
  * provides a `global_obs()` method used by the centralised MAPPO critic
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class EnvParams:
    n_buildings: int = 5
    horizon: int = 24 * 30
    battery_capacity_kwh: float = 40.0
    battery_max_power_kw: float = 10.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    reward_scale: float = 0.2
    seed: int = 42


class DistrictEnergyEnv:
    """
    Cooperative multi-agent district energy environment.

    Per-building observation (state_dim = 9):
      [hour/23, day/6, temp_norm, solar_norm, price_norm,
       cooling_norm, heating_norm, soc_norm, prev_consumption_norm]

    Global observation (global_state_dim = n_buildings * state_dim):
      Concatenation of all per-building observations → used by centralised critic.

    Action per building: continuous ∈ [-1, 1]
      positive → charge, negative → discharge
    """

    def __init__(self, params: EnvParams):
        self.p = params
        self.n_buildings = params.n_buildings
        self.horizon = params.horizon
        self.state_dim = 9
        self.global_state_dim = self.state_dim * self.n_buildings
        self.action_low = -1.0
        self.action_high = 1.0

        self.rng = np.random.default_rng(params.seed)
        self.data = self._build_data()

        self.t = 0
        self.soc = np.zeros(self.n_buildings, dtype=np.float32)
        self.prev_consumption = np.zeros(self.n_buildings, dtype=np.float32)
        self.prev_district_load = 0.0
        self.running_peak = 0.0

    # ── Data generation ───────────────────────────────────────────────────────
    def _build_data(self) -> Dict[str, np.ndarray]:
        h = np.arange(self.horizon)
        hour = h % 24
        day = (h // 24) % 7

        seasonal = 12.0 * np.sin(2.0 * math.pi * h / (24.0 * 365.0) - 1.1)
        daily = 6.0 * np.sin(2.0 * math.pi * (hour - 14.0) / 24.0)
        temp = 18.0 + seasonal + daily + self.rng.normal(0.0, 1.8, size=self.horizon)

        daylight = np.clip(np.sin(math.pi * (hour - 6.0) / 12.0), 0.0, None)
        seasonal_solar = 0.5 + 0.5 * np.sin(2.0 * math.pi * h / (24.0 * 365.0) - 0.3)
        solar_base = 6.0 * daylight * np.clip(seasonal_solar, 0.1, None)

        price = np.full(self.horizon, 0.12, dtype=np.float32)
        evening_peak = (hour >= 17) & (hour <= 21)
        shoulder = ((hour >= 7) & (hour <= 10)) | ((hour >= 22) & (hour <= 23))
        price[shoulder] = 0.18
        price[evening_peak] = 0.28

        cooling = np.zeros((self.horizon, self.n_buildings), dtype=np.float32)
        heating = np.zeros((self.horizon, self.n_buildings), dtype=np.float32)
        solar = np.zeros((self.horizon, self.n_buildings), dtype=np.float32)
        base_load = np.zeros((self.horizon, self.n_buildings), dtype=np.float32)

        for b in range(self.n_buildings):
            scale = self.rng.uniform(0.8, 1.25)
            cooling[:, b] = scale * np.clip(temp - 22.0, 0.0, None) * 0.9
            heating[:, b] = scale * np.clip(18.0 - temp, 0.0, None) * 0.85
            solar_noise = self.rng.normal(0.0, 0.25, size=self.horizon)
            solar[:, b] = np.clip(scale * (solar_base + solar_noise), 0.0, None)
            occupancy = 1.0 + 0.18 * np.sin(2.0 * math.pi * (hour - 8.0) / 24.0)
            non_hvac = scale * (2.3 + occupancy + self.rng.normal(0.0, 0.2, size=self.horizon))
            base_load[:, b] = np.clip(non_hvac, 0.8, None)

        return {
            "hour": hour.astype(np.float32),
            "day": day.astype(np.float32),
            "temp": temp.astype(np.float32),
            "price": price.astype(np.float32),
            "cooling": cooling,
            "heating": heating,
            "solar": solar,
            "base_load": base_load,
        }

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        idx = min(self.t, self.horizon - 1)
        obs = np.zeros((self.n_buildings, self.state_dim), dtype=np.float32)
        for b in range(self.n_buildings):
            cooling = self.data["cooling"][idx, b]
            heating = self.data["heating"][idx, b]
            solar   = self.data["solar"][idx, b]
            obs[b] = np.array([
                self.data["hour"][idx] / 23.0,
                self.data["day"][idx] / 6.0,
                (self.data["temp"][idx] + 10.0) / 50.0,
                solar / 10.0,
                self.data["price"][idx] / 0.30,
                cooling / 20.0,
                heating / 20.0,
                self.soc[b] / self.p.battery_capacity_kwh,
                self.prev_consumption[b] / 25.0,
            ], dtype=np.float32)
        return obs

    def global_obs(self, obs: np.ndarray) -> np.ndarray:
        """Flatten all agents' observations → centralised critic input."""
        return obs.reshape(-1).astype(np.float32)

    # ── Reset / Step ──────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.data = self._build_data()

        self.t = 0
        self.soc = self.rng.uniform(
            0.30 * self.p.battery_capacity_kwh,
            0.70 * self.p.battery_capacity_kwh,
            size=self.n_buildings,
        ).astype(np.float32)
        self.prev_consumption = np.zeros(self.n_buildings, dtype=np.float32)
        self.prev_district_load = 0.0
        self.running_peak = 0.0
        return self._get_obs(), {}

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n_buildings)
        actions = np.clip(actions, self.action_low, self.action_high)

        idx = self.t
        current_consumption = np.zeros(self.n_buildings, dtype=np.float32)

        for b in range(self.n_buildings):
            cooling = self.data["cooling"][idx, b]
            heating = self.data["heating"][idx, b]
            solar   = self.data["solar"][idx, b]
            base    = self.data["base_load"][idx, b]
            net_load = max(base + cooling + heating - solar, 0.0)

            a = actions[b]
            if a >= 0.0:
                desired_charge  = a * self.p.battery_max_power_kw
                room_kwh        = self.p.battery_capacity_kwh - self.soc[b]
                max_charge_pwr  = room_kwh / max(self.p.charge_efficiency, 1e-6)
                charge_power    = min(desired_charge, max_charge_pwr)
                self.soc[b]    += charge_power * self.p.charge_efficiency
                net_load       += charge_power
            else:
                desired_discharge = (-a) * self.p.battery_max_power_kw
                available_power   = self.soc[b] * self.p.discharge_efficiency
                discharge_power   = min(desired_discharge, available_power)
                self.soc[b]      -= discharge_power / max(self.p.discharge_efficiency, 1e-6)
                net_load          = max(net_load - discharge_power, 0.0)

            current_consumption[b] = net_load

        district_load    = float(np.sum(current_consumption))
        prev_peak        = self.running_peak
        self.running_peak = max(self.running_peak, district_load)
        peak_increase    = self.running_peak - prev_peak
        ramp             = abs(district_load - self.prev_district_load)

        load_norm     = district_load   / (self.n_buildings * 25.0)
        ramp_norm     = ramp            / (self.n_buildings * 10.0)
        peak_inc_norm = peak_increase   / (self.n_buildings * 10.0)
        raw_reward    = -(0.60 * load_norm + 0.15 * ramp_norm + 1.20 * peak_inc_norm)
        reward        = self.p.reward_scale * raw_reward
        rewards       = np.full(self.n_buildings, reward, dtype=np.float32)

        self.prev_consumption  = current_consumption
        self.prev_district_load = district_load

        self.t += 1
        terminated = self.t >= self.horizon
        truncated  = False
        obs = self._get_obs() if not terminated else np.zeros((self.n_buildings, self.state_dim), dtype=np.float32)

        info = {
            "district_load": district_load,
            "running_peak":  self.running_peak,
            "peak_increase": peak_increase,
            "ramp":          ramp,
            "electricity_cost": district_load * float(self.data["price"][idx]),
            "carbon_emissions": district_load * 0.233,   # kg CO2 / kWh (approx UK grid)
            "comfort_violation": 0.0,                    # placeholder – no HVAC setpoint
            "load_factor": district_load / max(self.running_peak, 1e-6),
        }
        return obs, rewards, terminated, truncated, info

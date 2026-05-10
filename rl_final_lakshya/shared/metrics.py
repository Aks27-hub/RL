"""
shared/metrics.py — Standardized metric extraction from CityLearn environment info.

extract_metrics() always returns exactly 7 keys regardless of what the env provides.
This ensures all algorithms produce identical metric schemas for benchmarking.
"""

from typing import Any, Dict, List, Optional


def extract_metrics(env, episode_reward: Optional[float] = None) -> Dict[str, float]:
    """Extract standardized metrics from a CityLearn environment after an episode.

    Attempts to read CityLearn's built-in KPIs. Falls back to 0.0 for any
    metric not available in the current environment version.

    Args:
        env: CityLearnWrapper instance (with .env attribute pointing to raw CityLearn env).
        episode_reward: Total episode reward. If None, defaults to 0.0.

    Returns:
        Dict with exactly these 7 keys:
        - reward: float
        - electricity_cost: float
        - carbon_emissions: float
        - ramping: float
        - load_factor: float
        - daily_peak: float
        - comfort_violation: float
    """
    metrics = {
        "reward": float(episode_reward) if episode_reward is not None else 0.0,
        "electricity_cost": 0.0,
        "carbon_emissions": 0.0,
        "ramping": 0.0,
        "load_factor": 0.0,
        "daily_peak": 0.0,
        "comfort_violation": 0.0,
    }

    # Try to extract KPIs from CityLearn env
    raw_env = env.env if hasattr(env, "env") else env
    try:
        kpis = raw_env.evaluate()
        if kpis is not None:
            # CityLearn evaluate() returns a DataFrame with columns:
            # 'cost_function', 'value', optionally 'building_id'
            # We aggregate across all buildings by taking the mean.
            if hasattr(kpis, "groupby"):
                kpi_summary = kpis.groupby("cost_function")["value"].mean().to_dict()
            elif isinstance(kpis, dict):
                kpi_summary = kpis
            else:
                kpi_summary = {}

            # Map CityLearn KPI names to our standard keys
            kpi_mapping = {
                "electricity_consumption": "electricity_cost",
                "price": "electricity_cost",
                "cost": "electricity_cost",
                "carbon_emissions": "carbon_emissions",
                "emission": "carbon_emissions",
                "ramping": "ramping",
                "load_factor": "load_factor",
                "1 - load_factor": "load_factor",
                "daily_peak": "daily_peak",
                "peak_demand": "daily_peak",
                "annual_peak": "daily_peak",
                "thermal_resilience": "comfort_violation",
                "discomfort": "comfort_violation",
                "unmet_hours": "comfort_violation",
            }

            for kpi_name, standard_key in kpi_mapping.items():
                if kpi_name in kpi_summary:
                    val = kpi_summary[kpi_name]
                    if isinstance(val, (int, float)) and not (val != val):  # check for NaN
                        metrics[standard_key] = float(val)

    except Exception:
        # If KPI extraction fails, return defaults (all 0.0 except reward)
        pass

    return metrics


def metrics_to_csv_row(metrics: Dict[str, float]) -> str:
    """Convert a metrics dict to a CSV row string (no newline).

    Args:
        metrics: Dict with the 7 standard keys.

    Returns:
        Comma-separated string of values in column order.
    """
    columns = [
        "reward", "electricity_cost", "carbon_emissions",
        "ramping", "load_factor", "daily_peak", "comfort_violation"
    ]
    return ",".join(str(metrics.get(col, 0.0)) for col in columns)


def csv_header() -> str:
    """Return the standard CSV header for training_curves.csv.

    Returns:
        Header string: 'episode,reward,...,comfort_violation'
    """
    return "episode,reward,electricity_cost,carbon_emissions,ramping,load_factor,daily_peak,comfort_violation"

"""
shared/env_wrapper.py — CityLearn environment factory with observation/action normalization.

Provides make_env() which returns a wrapped CityLearn environment with:
- Running mean/std observation normalization (updated during training, frozen during eval)
- Action scaling from agent outputs [-1, 1] to environment action bounds
- Unified API exposing obs_dim, action_dim, n_agents, action_space, is_discrete
- Transparent handling of single-agent and multi-agent obs/action structures
"""

import numpy as np
from citylearn.citylearn import CityLearnEnv
from typing import Any, Dict, List, Optional, Tuple, Union


class RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # small epsilon to avoid division by zero

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a batch of observations.

        Args:
            batch: Array of shape (batch_size, *shape) or (*shape,).
        """
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics.

        Args:
            x: Observation array.

        Returns:
            Normalized observation.
        """
        return (x - self.mean.astype(np.float32)) / (np.sqrt(self.var).astype(np.float32) + 1e-8)


class CityLearnWrapper:
    """Wrapper around CityLearn environment providing normalized obs/actions.

    Attributes:
        obs_dim: Observation dimension per agent.
        action_dim: Action dimension per agent.
        n_agents: Number of agents (buildings).
        is_discrete: Always False for CityLearn (continuous actions).
        action_space: List of action spaces, one per agent.
    """

    def __init__(self, env: CityLearnEnv, seed: int = 42):
        self.env = env
        self._seed = seed
        self.n_agents = len(env.buildings)
        self.is_discrete = False

        # Determine obs and action dims from the first reset
        obs = env.reset()
        if isinstance(obs, list):
            self._obs_list = True
            self.obs_dim = len(obs[0])
            self._per_agent_obs_dims = [len(o) for o in obs]
        else:
            self._obs_list = False
            self.obs_dim = len(obs)
            self._per_agent_obs_dims = [len(obs)]

        # Action dims
        self.action_space = env.action_space
        if isinstance(self.action_space, list):
            self.action_dim = self.action_space[0].shape[0]
            self._per_agent_action_dims = [s.shape[0] for s in self.action_space]
        else:
            self.action_dim = self.action_space.shape[0]
            self._per_agent_action_dims = [self.action_space.shape[0]]

        # Running normalization per agent
        self._obs_rms = [RunningMeanStd(shape=(dim,)) for dim in self._per_agent_obs_dims]
        self._training = True

        # Store initial obs for re-use
        self._initial_obs = obs

    @property
    def per_agent_obs_dims(self) -> List[int]:
        """Return list of observation dimensions, one per agent."""
        return self._per_agent_obs_dims

    @property
    def per_agent_action_dims(self) -> List[int]:
        """Return list of action dimensions, one per agent."""
        return self._per_agent_action_dims

    def set_training(self, training: bool) -> None:
        """Set whether to update running statistics (True during training, False during eval).

        Args:
            training: If True, update running mean/std on each observation.
        """
        self._training = training

    def _normalize_obs(self, obs_list: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize observations using running mean/std.

        Args:
            obs_list: List of per-agent observation arrays.

        Returns:
            List of normalized per-agent observation arrays.
        """
        normalized = []
        for i, obs in enumerate(obs_list):
            obs = np.asarray(obs, dtype=np.float32)
            if self._training:
                self._obs_rms[i].update(obs)
            normalized.append(self._obs_rms[i].normalize(obs))
        return normalized

    def _to_obs_list(self, obs: Any) -> List[np.ndarray]:
        """Convert env observations to a list of per-agent arrays.

        Args:
            obs: Raw observations from CityLearn (list or single array).

        Returns:
            List of numpy arrays, one per agent.
        """
        if isinstance(obs, list):
            return [np.asarray(o, dtype=np.float32) for o in obs]
        else:
            return [np.asarray(obs, dtype=np.float32)]

    def _scale_actions(self, actions: List[np.ndarray]) -> List[np.ndarray]:
        """Scale agent actions from [-1, 1] to environment action bounds.

        Args:
            actions: List of per-agent action arrays in [-1, 1].

        Returns:
            List of per-agent action arrays scaled to env bounds.
        """
        scaled = []
        action_spaces = self.action_space if isinstance(self.action_space, list) else [self.action_space]
        for i, (act, space) in enumerate(zip(actions, action_spaces)):
            act = np.asarray(act, dtype=np.float32)
            low = space.low.astype(np.float32)
            high = space.high.astype(np.float32)
            # Map [-1, 1] → [low, high]
            scaled_act = low + (act + 1.0) * 0.5 * (high - low)
            scaled_act = np.clip(scaled_act, low, high)
            scaled.append(scaled_act)
        return scaled

    def reset(self) -> List[np.ndarray]:
        """Reset the environment and return normalized observations.

        Returns:
            List of normalized per-agent observation arrays.
        """
        obs = self.env.reset()
        obs_list = self._to_obs_list(obs)
        return self._normalize_obs(obs_list)

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            actions: List of per-agent actions in [-1, 1].

        Returns:
            Tuple of (obs_list, rewards, done, info):
                obs_list: List of normalized per-agent observations.
                rewards: List of per-agent rewards.
                done: Whether the episode is finished.
                info: Info dict from the environment.
        """
        scaled_actions = self._scale_actions(actions)
        obs, rewards, done, info = self.env.step(scaled_actions)
        obs_list = self._to_obs_list(obs)
        obs_list = self._normalize_obs(obs_list)

        # Ensure rewards is a list
        if not isinstance(rewards, list):
            rewards = [float(rewards)] * self.n_agents

        rewards = [float(r) for r in rewards]

        return obs_list, rewards, done, info

    def get_metadata(self) -> Dict[str, Any]:
        """Return environment metadata for logging.

        Returns:
            Dict with env configuration details.
        """
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "per_agent_obs_dims": self._per_agent_obs_dims,
            "per_agent_action_dims": self._per_agent_action_dims,
            "is_discrete": self.is_discrete,
        }


def make_env(schema_name: str = "citylearn_challenge_2022_phase_1", seed: int = 42,
             simulation_days: int = 30) -> CityLearnWrapper:
    """Create a wrapped CityLearn environment.

    Args:
        schema_name: CityLearn schema dataset name.
        seed: Random seed for the environment.
        simulation_days: Number of days to simulate per episode.
                         30 days = 720 timesteps (captures diurnal/weekly cycles).
                         Set to 365 for full-year episodes.

    Returns:
        CityLearnWrapper instance with normalization and unified API.
    """
    env = CityLearnEnv(schema=schema_name, central_agent=False)
    # Limit episode length by modifying episode_time_steps in the schema
    max_steps = simulation_days * 24
    total_steps = env.schema.get("episode_time_steps", None) or env.time_steps
    if max_steps < total_steps:
        env.schema["episode_time_steps"] = max_steps
        env.episode_time_steps = max_steps
    np.random.seed(seed)
    wrapper = CityLearnWrapper(env, seed=seed)
    return wrapper

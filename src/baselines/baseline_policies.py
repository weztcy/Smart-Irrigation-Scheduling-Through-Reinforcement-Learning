"""Conventional baseline irrigation policies for fair comparison against RL."""
from typing import Tuple, Optional, Any, Dict
import numpy as np


def get_action_index(volume_mm: float, action_volumes: Optional[Dict[int, float]] = None) -> int:
    """Find the closest discrete action index for a desired volume in mm."""
    if action_volumes is None:
        action_volumes = {0: 0.0, 1: 5.0, 2: 10.0, 3: 20.0}
    closest_action = min(action_volumes.keys(), key=lambda a: abs(action_volumes[a] - volume_mm))
    return int(closest_action)


class FixedSchedulePolicy:
    """Conventional fixed schedule: irrigates a fixed volume at regular calendar intervals."""

    def __init__(self, interval_days: int = 3, volume_mm: float = 10.0, action_volumes: Optional[Dict[int, float]] = None):
        self.interval_days = int(interval_days)
        self.volume_mm = float(volume_mm)
        self.action_volumes = action_volumes or {0: 0.0, 1: 5.0, 2: 10.0, 3: 20.0}
        self.action_idx = get_action_index(self.volume_mm, self.action_volumes)
        self.day_counter = 0

    def reset(self) -> None:
        self.day_counter = 0

    def predict(self, observation: np.ndarray, state: Optional[Any] = None, episode_start: Optional[Any] = None, deterministic: bool = True) -> Tuple[int, Optional[Any]]:
        # Observation format: obs[6] = day_norm
        self.day_counter += 1
        # Irrigate every interval_days (e.g., day 1, 4, 7... or day 3, 6, 9...)
        if self.day_counter % self.interval_days == 0:
            return self.action_idx, state
        return 0, state


class ThresholdBasedPolicy:
    """Rule-based / sensor-based: triggers irrigation whenever soil moisture falls below a static threshold."""

    def __init__(self, lower_threshold_ratio: float = 0.50, volume_mm: float = 10.0, action_volumes: Optional[Dict[int, float]] = None):
        self.lower_threshold_ratio = float(lower_threshold_ratio)
        self.volume_mm = float(volume_mm)
        self.action_volumes = action_volumes or {0: 0.0, 1: 5.0, 2: 10.0, 3: 20.0}
        self.action_idx = get_action_index(self.volume_mm, self.action_volumes)

    def reset(self) -> None:
        pass

    def predict(self, observation: np.ndarray, state: Optional[Any] = None, episode_start: Optional[Any] = None, deterministic: bool = True) -> Tuple[int, Optional[Any]]:
        # Observation format: obs[0] = soil_moisture_ratio
        soil_moisture_ratio = float(observation[0])
        if soil_moisture_ratio < self.lower_threshold_ratio:
            return self.action_idx, state
        return 0, state


class SmartThresholdPolicy:
    """Adaptive threshold policy: considers current moisture and avoids watering if heavy rain is present."""

    def __init__(self, lower_threshold_ratio: float = 0.55, volume_mm: float = 10.0, rain_threshold_skip_mm: float = 5.0, action_volumes: Optional[Dict[int, float]] = None):
        self.lower_threshold_ratio = float(lower_threshold_ratio)
        self.volume_mm = float(volume_mm)
        self.rain_threshold_skip_mm = float(rain_threshold_skip_mm)
        self.action_volumes = action_volumes or {0: 0.0, 1: 5.0, 2: 10.0, 3: 20.0}
        self.action_idx = get_action_index(self.volume_mm, self.action_volumes)

    def reset(self) -> None:
        pass

    def predict(self, observation: np.ndarray, state: Optional[Any] = None, episode_start: Optional[Any] = None, deterministic: bool = True) -> Tuple[int, Optional[Any]]:
        # Observation: obs[0] = soil_moisture_ratio, obs[2] = rainfall_norm (Rain / 50.0)
        soil_moisture_ratio = float(observation[0])
        rain_mm = float(observation[2]) * 50.0

        if rain_mm >= self.rain_threshold_skip_mm:
            # Skip watering because natural rain satisfies soil moisture demand
            return 0, state

        if soil_moisture_ratio < self.lower_threshold_ratio:
            return self.action_idx, state
        return 0, state


class RandomPolicy:
    """Random baseline: uniformly samples actions from the action space."""

    def __init__(self, num_actions: int = 4, seed: Optional[int] = None):
        self.num_actions = num_actions
        self.rng = np.random.RandomState(seed)

    def reset(self) -> None:
        pass

    def predict(self, observation: np.ndarray, state: Optional[Any] = None, episode_start: Optional[Any] = None, deterministic: bool = True) -> Tuple[int, Optional[Any]]:
        return int(self.rng.choice(self.num_actions)), state


class NoIrrigationPolicy:
    """Control baseline: never irrigates (rainfed agriculture)."""

    def __init__(self):
        pass

    def reset(self) -> None:
        pass

    def predict(self, observation: np.ndarray, state: Optional[Any] = None, episode_start: Optional[Any] = None, deterministic: bool = True) -> Tuple[int, Optional[Any]]:
        return 0, state

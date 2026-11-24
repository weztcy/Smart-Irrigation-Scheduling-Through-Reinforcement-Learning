"""Weather simulation model for agricultural irrigation environment."""
from typing import Dict, Any, Tuple, Optional
import numpy as np


class WeatherModel:
    """Simulates daily weather dynamics including temperature, precipitation, and reference evapotranspiration (ET0).
    
    Uses a 3-state Markov chain (Sunny, Cloudy, Rainy) combined with physical
    evapotranspiration approximations.
    """

    STATE_NAMES = {0: "Sunny", 1: "Cloudy", 2: "Rainy"}

    def __init__(self, scenario_config: Optional[Dict[str, Any]] = None, scenario_name: str = "normal", seed: Optional[int] = None):
        """Initialize weather model with a specific scenario configuration."""
        self.scenario_name = scenario_name
        self.rng = np.random.RandomState(seed)

        # Default fallback parameters if config not provided
        self.config = scenario_config or self._default_scenario(scenario_name)
        
        self.temp_mean = float(self.config.get("temp_mean", 28.0))
        self.temp_std = float(self.config.get("temp_std", 3.0))
        self.rain_prob = float(self.config.get("rain_prob", 0.25))
        self.rain_amount_mean = float(self.config.get("rain_amount_mean", 12.0))
        self.rain_amount_std = float(self.config.get("rain_amount_std", 6.0))
        
        # 3x3 transition matrix: P[current_state, next_state]
        default_matrix = [
            [0.65, 0.25, 0.10],  # From Sunny
            [0.35, 0.45, 0.20],  # From Cloudy
            [0.30, 0.40, 0.30]   # From Rainy
        ]
        self.transition_matrix = np.array(self.config.get("transition_matrix", default_matrix), dtype=np.float64)

        # State tracking
        self.current_state: int = 0
        self.current_temp: float = self.temp_mean
        self.current_rain: float = 0.0
        self.current_et0: float = 4.5

    def _default_scenario(self, name: str) -> Dict[str, Any]:
        """Provides default parameters for predefined weather scenarios."""
        scenarios = {
            "normal": {
                "temp_mean": 28.0, "temp_std": 3.0, "rain_prob": 0.25,
                "rain_amount_mean": 12.0, "rain_amount_std": 6.0,
                "transition_matrix": [[0.65, 0.25, 0.10], [0.35, 0.45, 0.20], [0.30, 0.40, 0.30]]
            },
            "dry": {
                "temp_mean": 33.0, "temp_std": 2.5, "rain_prob": 0.08,
                "rain_amount_mean": 5.0, "rain_amount_std": 3.0,
                "transition_matrix": [[0.80, 0.16, 0.04], [0.60, 0.35, 0.05], [0.50, 0.40, 0.10]]
            },
            "wet": {
                "temp_mean": 24.0, "temp_std": 2.5, "rain_prob": 0.55,
                "rain_amount_mean": 18.0, "rain_amount_std": 8.0,
                "transition_matrix": [[0.35, 0.40, 0.25], [0.20, 0.45, 0.35], [0.15, 0.40, 0.45]]
            },
            "heatwave": {
                "temp_mean": 36.5, "temp_std": 2.5, "rain_prob": 0.10,
                "rain_amount_mean": 8.0, "rain_amount_std": 4.0,
                "transition_matrix": [[0.85, 0.12, 0.03], [0.65, 0.30, 0.05], [0.55, 0.35, 0.10]]
            }
        }
        return scenarios.get(name, scenarios["normal"])

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)

    def reset(self, initial_state: Optional[int] = None) -> Tuple[int, float, float, float]:
        """Reset the weather sequence at the beginning of an episode."""
        if initial_state is not None:
            self.current_state = int(initial_state)
        else:
            # Stationary or initial prior: 60% sunny, 30% cloudy, 10% rainy
            self.current_state = int(self.rng.choice([0, 1, 2], p=[0.60, 0.30, 0.10]))

        self.current_temp = self._sample_temperature(self.current_state)
        self.current_rain = self._sample_rainfall(self.current_state)
        self.current_et0 = self._calculate_et0(self.current_state, self.current_temp)

        return self.current_state, self.current_temp, self.current_rain, self.current_et0

    def step(self) -> Tuple[int, float, float, float]:
        """Transition to next day's weather state and generate meteorological variables."""
        prob_dist = self.transition_matrix[self.current_state]
        self.current_state = int(self.rng.choice([0, 1, 2], p=prob_dist))
        
        self.current_temp = self._sample_temperature(self.current_state)
        self.current_rain = self._sample_rainfall(self.current_state)
        self.current_et0 = self._calculate_et0(self.current_state, self.current_temp)

        return self.current_state, self.current_temp, self.current_rain, self.current_et0

    def _sample_temperature(self, state: int) -> float:
        """Sample daily mean temperature adjusted by weather condition."""
        # Sunny days are warmer; rainy days are cooler
        state_temp_offset = {0: 1.5, 1: -0.5, 2: -2.5}
        base_temp = self.temp_mean + state_temp_offset.get(state, 0.0)
        temp = float(self.rng.normal(base_temp, self.temp_std))
        return float(np.clip(temp, 10.0, 48.0))

    def _sample_rainfall(self, state: int) -> float:
        """Sample daily rainfall in millimeters."""
        if state == 2:  # Rainy state
            # Positive rainfall sampled from truncated normal or lognormal
            rain = float(self.rng.normal(self.rain_amount_mean, self.rain_amount_std))
            return float(np.clip(rain, 1.0, 80.0))
        elif state == 1:  # Cloudy state: small chance of drizzle
            if self.rng.rand() < 0.15:
                return float(self.rng.uniform(0.5, 3.0))
            return 0.0
        else:  # Sunny state
            return 0.0

    def _calculate_et0(self, state: int, temp: float) -> float:
        """Calculate Reference Evapotranspiration (ET0) in mm/day.
        
        Uses an empirical formulation aligned with Hargreaves and FAO-56 principles:
        ET0 scales positively with temperature and solar radiation (weather state).
        """
        # Solar radiation factor by sky condition
        solar_factor = {0: 1.25, 1: 0.85, 2: 0.45}
        
        # Base ET0 increases linearly with temperature: ET0 ~ 0.16 * T + offset
        # Typical tropical/subtropical values: 2.0 mm (cool/rainy) to 7.5 mm (hot/sunny)
        base_et0 = max(1.0, 0.15 * (temp - 5.0))
        et0 = base_et0 * solar_factor.get(state, 1.0)
        return float(np.clip(et0, 0.8, 10.0))

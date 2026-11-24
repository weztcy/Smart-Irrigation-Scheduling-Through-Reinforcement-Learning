"""Gymnasium-compliant Smart Irrigation Environment."""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.environment.weather_model import WeatherModel


class SmartIrrigationEnv(gym.Env):
    """Agricultural simulation environment for smart irrigation scheduling.
    
    Models soil moisture dynamics, crop evapotranspiration, rainfall, and irrigation impact
    over a seasonal growth cycle (MDP).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: Optional[Dict[str, Any]] = None, scenario: str = "normal", seed: Optional[int] = None):
        super().__init__()
        self.config = config or {}
        env_cfg = self.config.get("environment", {})
        
        # Horizon & soil parameters
        self.horizon_days = int(env_cfg.get("horizon_days", 90))
        self.soil_capacity_mm = float(env_cfg.get("soil_capacity_mm", 100.0))
        self.wilting_point_ratio = float(env_cfg.get("wilting_point_ratio", 0.20))
        self.field_capacity_ratio = float(env_cfg.get("field_capacity_ratio", 0.80))
        self.saturation_ratio = float(env_cfg.get("saturation_ratio", 1.00))
        self.initial_moisture_min = float(env_cfg.get("initial_moisture_min", 0.40))
        self.initial_moisture_max = float(env_cfg.get("initial_moisture_max", 0.65))
        self.drainage_rate = float(env_cfg.get("drainage_rate", 0.50))
        self.runoff_coefficient = float(env_cfg.get("runoff_coefficient", 0.15))

        # Reward weights
        reward_cfg = env_cfg.get("reward", {})
        self.target_in_range_bonus = float(reward_cfg.get("target_in_range_bonus", 1.0))
        self.deficit_penalty_weight = float(reward_cfg.get("deficit_penalty_weight", 2.5))
        self.excess_penalty_weight = float(reward_cfg.get("excess_penalty_weight", 1.5))
        self.water_penalty_weight = float(reward_cfg.get("water_penalty_weight", 0.04))
        self.rain_waste_penalty_weight = float(reward_cfg.get("rain_waste_penalty_weight", 0.05))

        # Crop growth stages
        default_stages = [
            {"name": "Initial", "start_day": 0, "end_day": 20, "kc": 0.45, "target_min": 0.45, "target_max": 0.65, "stress_multiplier": 1.0},
            {"name": "Development", "start_day": 21, "end_day": 50, "kc": 0.85, "target_min": 0.55, "target_max": 0.75, "stress_multiplier": 1.2},
            {"name": "Mid-Season (Flowering)", "start_day": 51, "end_day": 75, "kc": 1.15, "target_min": 0.65, "target_max": 0.85, "stress_multiplier": 2.0},
            {"name": "Late-Season (Ripening)", "start_day": 76, "end_day": 90, "kc": 0.60, "target_min": 0.45, "target_max": 0.65, "stress_multiplier": 0.8}
        ]
        self.stages = env_cfg.get("stages", default_stages)

        # Action definition: irrigation volume in mm
        raw_actions = env_cfg.get("action_volumes", {0: 0.0, 1: 5.0, 2: 10.0, 3: 20.0})
        self.action_volumes = {int(k): float(v) for k, v in raw_actions.items()}
        self.num_actions = len(self.action_volumes)

        # Action & Observation Spaces
        self.action_space = spaces.Discrete(self.num_actions)
        # Observation features:
        # [0] soil_moisture_ratio (0.0 to 1.0)
        # [1] temperature_norm ((T - 15) / 30)
        # [2] rainfall_norm (Rain / 50)
        # [3] weather_state_norm (0.0=Sunny, 0.5=Cloudy, 1.0=Rainy)
        # [4] growth_stage_norm (stage_idx / 3.0)
        # [5] crop_water_demand_norm (ETc / 10.0)
        # [6] day_norm (day / horizon)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Weather generator
        weather_cfg = self.config.get("weather", {})
        scenario_dict = weather_cfg.get("scenarios", {}).get(scenario, None)
        self.scenario = scenario
        self.weather_model = WeatherModel(scenario_config=scenario_dict, scenario_name=scenario, seed=seed)

        # Simulation state
        self.current_day: int = 0
        self.soil_moisture_mm: float = 50.0
        self.weather_state: int = 0
        self.temperature: float = 28.0
        self.rainfall: float = 0.0
        self.et0: float = 4.5
        self.cumulative_water_used: float = 0.0
        self.total_irrigations: int = 0
        self.stress_deficit_days: int = 0
        self.excess_water_days: int = 0
        self.target_compliance_days: int = 0

        self.rng = np.random.RandomState(seed)

    def _get_stage_info(self, day: int) -> Dict[str, Any]:
        """Retrieve crop growth stage metadata for the specified day."""
        for idx, stage in enumerate(self.stages):
            if stage["start_day"] <= day <= stage["end_day"]:
                stage_data = dict(stage)
                stage_data["stage_idx"] = idx
                return stage_data
        # Fallback to last stage
        last_stage = dict(self.stages[-1])
        last_stage["stage_idx"] = len(self.stages) - 1
        return last_stage

    def _get_observation(self) -> np.ndarray:
        """Construct normalized observation vector for the agent."""
        stage_info = self._get_stage_info(self.current_day)
        kc = stage_info["kc"]
        etc = kc * self.et0

        obs = np.array([
            np.clip(self.soil_moisture_mm / self.soil_capacity_mm, 0.0, 1.0),
            np.clip((self.temperature - 15.0) / 30.0, 0.0, 1.0),
            np.clip(self.rainfall / 50.0, 0.0, 1.0),
            np.clip(self.weather_state / 2.0, 0.0, 1.0),
            np.clip(stage_info["stage_idx"] / 3.0, 0.0, 1.0),
            np.clip(etc / 10.0, 0.0, 1.0),
            np.clip(self.current_day / max(1, self.horizon_days), 0.0, 1.0)
        ], dtype=np.float32)

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.weather_model.seed(seed)

        # Allow scenario override in options
        if options and "scenario" in options:
            self.scenario = options["scenario"]
            weather_cfg = self.config.get("weather", {})
            scenario_dict = weather_cfg.get("scenarios", {}).get(self.scenario, None)
            self.weather_model = WeatherModel(scenario_config=scenario_dict, scenario_name=self.scenario, seed=seed)

        self.current_day = 0
        
        # Sample realistic initial soil moisture
        init_ratio = float(self.rng.uniform(self.initial_moisture_min, self.initial_moisture_max))
        self.soil_moisture_mm = init_ratio * self.soil_capacity_mm

        # Initialize weather for day 0
        self.weather_state, self.temperature, self.rainfall, self.et0 = self.weather_model.reset()

        # Reset cumulative tracking stats
        self.cumulative_water_used = 0.0
        self.total_irrigations = 0
        self.stress_deficit_days = 0
        self.excess_water_days = 0
        self.target_compliance_days = 0

        obs = self._get_observation()
        info = self._get_info(action=0, reward=0.0, etc=self._get_stage_info(0)["kc"] * self.et0)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance simulation by one day with given irrigation action."""
        if isinstance(action, (np.ndarray, np.generic)):
            action = int(np.asarray(action).item())
        else:
            action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {list(self.action_volumes.keys())}")

        action_mm = self.action_volumes[action]
        if action_mm > 0:
            self.total_irrigations += 1
            self.cumulative_water_used += action_mm

        stage_info = self._get_stage_info(self.current_day)
        kc = stage_info["kc"]
        etc = kc * self.et0
        target_min = stage_info["target_min"]
        target_max = stage_info["target_max"]
        stress_multiplier = stage_info["stress_multiplier"]

        # Effective rainfall calculation (accounting for runoff during intense rain)
        effective_rain = self.rainfall * (1.0 - self.runoff_coefficient) if self.rainfall > 10.0 else self.rainfall

        # Soil moisture dynamics: Balance = S_prev + Irrigation + EffectiveRain - ETc
        soil_water_pre_drain = self.soil_moisture_mm + action_mm + effective_rain - etc
        
        # Deep percolation drainage if above field capacity
        field_capacity_mm = self.field_capacity_ratio * self.soil_capacity_mm
        drainage = 0.0
        if soil_water_pre_drain > field_capacity_mm:
            excess_water = soil_water_pre_drain - field_capacity_mm
            drainage = excess_water * self.drainage_rate
            soil_water_next = soil_water_pre_drain - drainage
        else:
            soil_water_next = soil_water_pre_drain

        # Physical clamping between 0 and saturation
        saturation_mm = self.saturation_ratio * self.soil_capacity_mm
        self.soil_moisture_mm = float(np.clip(soil_water_next, 0.0, saturation_mm))
        moisture_ratio = self.soil_moisture_mm / self.soil_capacity_mm

        # Evaluate plant health conditions
        in_target = (target_min <= moisture_ratio <= target_max)
        is_deficit = (moisture_ratio < target_min)
        is_overwatered = (moisture_ratio > target_max)

        if in_target:
            self.target_compliance_days += 1
        if is_deficit:
            self.stress_deficit_days += 1
        if is_overwatered:
            self.excess_water_days += 1

        # Calculate Reward components
        # 1. Target range reward
        if in_target:
            # Centered parabolic incentive for staying near the middle of target
            target_mid = 0.5 * (target_min + target_max)
            tolerance = max(0.01, 0.5 * (target_max - target_min))
            dist_norm = abs(moisture_ratio - target_mid) / tolerance
            r_target = self.target_in_range_bonus * (1.0 - 0.25 * (dist_norm ** 2))
        else:
            r_target = 0.0

        # 2. Deficit penalty (scaled by stage vulnerability)
        pen_deficit = 0.0
        if is_deficit:
            deficit_gap = target_min - moisture_ratio
            # Severe penalty if moisture drops below permanent wilting point
            if moisture_ratio < self.wilting_point_ratio:
                deficit_gap += (self.wilting_point_ratio - moisture_ratio) * 1.5
            pen_deficit = self.deficit_penalty_weight * deficit_gap * stress_multiplier

        # 3. Excess moisture penalty (waterlogging, root anoxia)
        pen_excess = 0.0
        if is_overwatered:
            excess_gap = moisture_ratio - target_max
            pen_excess = self.excess_penalty_weight * excess_gap

        # 4. Water consumption cost
        pen_water = self.water_penalty_weight * action_mm

        # 5. Inefficiency penalty: irrigating when heavy rain is occurring
        pen_rain_waste = 0.0
        if action_mm > 0 and self.rainfall > 5.0:
            pen_rain_waste = self.rain_waste_penalty_weight * action_mm

        # Net step reward
        reward = float(r_target - pen_deficit - pen_excess - pen_water - pen_rain_waste)

        step_day = self.current_day
        # Advance day
        self.current_day += 1
        terminated = bool(self.current_day >= self.horizon_days)
        truncated = False

        # Prepare step metadata
        info = self._get_info(action=action, reward=reward, etc=etc)
        info.update({
            "step_day": step_day,
            "day": self.current_day,
            "effective_rain_mm": effective_rain,
            "drainage_mm": drainage,
            "pen_deficit": pen_deficit,
            "pen_excess": pen_excess,
            "pen_water": pen_water,
            "pen_rain_waste": pen_rain_waste,
            "r_target": r_target
        })

        # Advance weather for the next step (if not terminated)
        if not terminated:
            self.weather_state, self.temperature, self.rainfall, self.et0 = self.weather_model.step()

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _get_info(self, action: int, reward: float, etc: float) -> Dict[str, Any]:
        """Compile comprehensive step metadata."""
        stage_info = self._get_stage_info(self.current_day)
        moisture_ratio = self.soil_moisture_mm / self.soil_capacity_mm
        target_min = stage_info["target_min"]
        target_max = stage_info["target_max"]

        return {
            "day": self.current_day,
            "stage_name": stage_info["name"],
            "stage_idx": stage_info["stage_idx"],
            "kc": stage_info["kc"],
            "target_min": target_min,
            "target_max": target_max,
            "soil_moisture_mm": self.soil_moisture_mm,
            "soil_moisture_ratio": moisture_ratio,
            "temperature_c": self.temperature,
            "weather_state": self.weather_model.STATE_NAMES.get(self.weather_state, "Unknown"),
            "rainfall_mm": self.rainfall,
            "et0_mm": self.et0,
            "etc_mm": etc,
            "action_idx": action,
            "action_volume_mm": self.action_volumes.get(action, 0.0),
            "step_reward": reward,
            "in_target": (target_min <= moisture_ratio <= target_max),
            "is_deficit": (moisture_ratio < target_min),
            "is_overwatered": (moisture_ratio > target_max),
            "cumulative_water_used_mm": self.cumulative_water_used,
            "total_irrigations": self.total_irrigations,
            "target_compliance_days": self.target_compliance_days,
            "stress_deficit_days": self.stress_deficit_days,
            "excess_water_days": self.excess_water_days,
            "scenario": self.scenario
        }

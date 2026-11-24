"""Unit tests for Smart Irrigation Environment and Weather Model."""
import pytest
import numpy as np
import yaml

from src.environment.weather_model import WeatherModel
from src.environment.irrigation_env import SmartIrrigationEnv


@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_weather_model_initialization_and_reset():
    """Verify weather model initializes properly and produces expected distributions."""
    wm = WeatherModel(scenario_name="normal", seed=42)
    state, temp, rain, et0 = wm.reset(initial_state=0)
    
    assert state in [0, 1, 2]
    assert 10.0 <= temp <= 48.0
    assert rain >= 0.0
    assert 0.8 <= et0 <= 10.0


def test_weather_model_transitions():
    """Verify weather transitions run smoothly over 90 days."""
    wm = WeatherModel(scenario_name="normal", seed=101)
    wm.reset()
    for _ in range(90):
        state, temp, rain, et0 = wm.step()
        assert state in [0, 1, 2]
        assert 10.0 <= temp <= 48.0
        assert rain >= 0.0
        assert 0.8 <= et0 <= 10.0


def test_environment_init_and_reset(config):
    """Verify SmartIrrigationEnv initializes, resets, and returns valid observation space."""
    env = SmartIrrigationEnv(config=config, scenario="normal", seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (7,)
    assert env.observation_space.contains(obs)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert info["day"] == 0
    assert info["soil_moisture_mm"] > 0.0


def test_invalid_action_handling(config):
    """Verify invalid action raises ValueError."""
    env = SmartIrrigationEnv(config=config, scenario="normal", seed=42)
    env.reset(seed=42)
    with pytest.raises(ValueError):
        env.step(99)


def test_physical_moisture_bounds(config):
    """Verify soil moisture stays strictly bounded between 0 and soil capacity."""
    env = SmartIrrigationEnv(config=config, scenario="normal", seed=42)
    env.reset(seed=42)
    
    # Force heavy irrigations
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(action=3)  # Heavy 20mm
        assert 0.0 <= info["soil_moisture_mm"] <= env.soil_capacity_mm
        assert np.isfinite(reward)
        if terminated:
            break


def test_full_episode_run(config):
    """Verify episode runs to 90 days horizon without errors and terminates cleanly."""
    env = SmartIrrigationEnv(config=config, scenario="normal", seed=42)
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    steps = 0
    terminated = False
    
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert env.observation_space.contains(obs)
        total_reward += reward
        steps += 1
    
    assert steps == 90
    assert terminated is True
    assert info["day"] == 90

"""Simulation tests for water balance and reward logic."""
import pytest
import numpy as np
import yaml

from src.environment.irrigation_env import SmartIrrigationEnv


@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_reward_penalizes_deficit(config):
    """Verify that zero irrigation under dry conditions triggers deficit penalty."""
    env = SmartIrrigationEnv(config=config, scenario="dry", seed=42)
    env.reset(seed=42)
    
    # Run without any irrigation
    step_rewards = []
    for _ in range(40):
        obs, reward, terminated, truncated, info = env.step(0)
        step_rewards.append(reward)
        if terminated:
            break

    # Later steps should incur deficit penalties
    assert min(step_rewards) < 0.0


def test_drainage_behavior(config):
    """Verify excess water above field capacity is drained."""
    env = SmartIrrigationEnv(config=config, scenario="normal", seed=42)
    env.reset(seed=42)
    
    # Set moisture near saturation
    env.soil_moisture_mm = 95.0
    obs, reward, terminated, truncated, info = env.step(3)  # Add 20mm
    
    assert info["drainage_mm"] > 0.0
    assert info["soil_moisture_mm"] <= env.soil_capacity_mm


def test_scenarios_diversity(config):
    """Verify dry and wet scenarios yield distinct rainfall distributions."""
    env_dry = SmartIrrigationEnv(config=config, scenario="dry", seed=42)
    env_wet = SmartIrrigationEnv(config=config, scenario="wet", seed=42)
    
    env_dry.reset(seed=42)
    env_wet.reset(seed=42)
    
    dry_rain = [env_dry.step(0)[4]["rainfall_mm"] for _ in range(50)]
    wet_rain = [env_wet.step(0)[4]["rainfall_mm"] for _ in range(50)]
    
    assert sum(wet_rain) > sum(dry_rain)

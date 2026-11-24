"""Comprehensive evaluation module comparing RL Agent against baseline policies."""
import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.environment.irrigation_env import SmartIrrigationEnv
from src.baselines.baseline_policies import (
    FixedSchedulePolicy,
    ThresholdBasedPolicy,
    SmartThresholdPolicy,
    RandomPolicy,
    NoIrrigationPolicy
)


def run_episode(policy, env, deterministic: bool = True):
    """Execute a single complete episode with given policy and record daily trajectory."""
    obs, info = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    trajectory = []
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        trajectory.append(dict(step_info))

    df_traj = pd.DataFrame(trajectory)
    
    # Calculate episode-level performance metrics
    total_water = float(df_traj["action_volume_mm"].sum())
    mean_moisture = float(df_traj["soil_moisture_ratio"].mean())
    target_compliance_pct = float(df_traj["in_target"].mean() * 100.0)
    deficit_days = int(df_traj["is_deficit"].sum())
    excess_days = int(df_traj["is_overwatered"].sum())
    num_irrigations = int((df_traj["action_volume_mm"] > 0).sum())
    
    # Water Use Efficiency: Target compliance days achieved per 10 mm of applied irrigation
    # If total_water == 0, avoid division by zero
    compliance_days = int(df_traj["in_target"].sum())
    wue = (compliance_days / (total_water / 10.0)) if total_water > 0 else (compliance_days * 1.0)

    metrics = {
        "cumulative_reward": round(total_reward, 3),
        "total_water_mm": round(total_water, 2),
        "mean_moisture_ratio": round(mean_moisture, 3),
        "target_compliance_pct": round(target_compliance_pct, 2),
        "deficit_stress_days": deficit_days,
        "excess_water_days": excess_days,
        "num_irrigations": num_irrigations,
        "water_use_efficiency": round(wue, 3)
    }

    return metrics, df_traj


def evaluate_all_policies(config_path: str = "config/config.yaml", model_path: str = "models/ppo_irrigation_agent.zip", episodes_per_seed: int = 5):
    """Run rigorous multi-scenario, multi-seed evaluation across all agents and baselines."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("evaluation", {})
    seeds = eval_cfg.get("seeds", [101, 202, 303, 404, 505])
    scenarios = eval_cfg.get("scenarios", ["normal", "dry", "wet", "heatwave"])

    # Load RL agent
    print(f"Loading trained RL model from {model_path}...")
    rl_agent = PPO.load(model_path)

    # Initialize policies dictionary
    action_vols = config.get("environment", {}).get("action_volumes", {0: 0.0, 1: 5.0, 2: 10.0, 3: 20.0})
    base_cfg = config.get("baselines", {})
    
    policies = {
        "RL Agent (PPO)": rl_agent,
        "Fixed Schedule (3-Day)": FixedSchedulePolicy(
            interval_days=base_cfg.get("fixed_schedule", {}).get("interval_days", 3),
            volume_mm=base_cfg.get("fixed_schedule", {}).get("volume_mm", 10.0),
            action_volumes=action_vols
        ),
        "Threshold-Based": ThresholdBasedPolicy(
            lower_threshold_ratio=base_cfg.get("threshold_based", {}).get("lower_threshold_ratio", 0.50),
            volume_mm=base_cfg.get("threshold_based", {}).get("irrigation_volume_mm", 10.0),
            action_volumes=action_vols
        ),
        "Smart Threshold": SmartThresholdPolicy(
            lower_threshold_ratio=base_cfg.get("smart_threshold", {}).get("lower_threshold_ratio", 0.55),
            volume_mm=base_cfg.get("smart_threshold", {}).get("irrigation_volume_mm", 10.0),
            rain_threshold_skip_mm=base_cfg.get("smart_threshold", {}).get("rain_threshold_skip_mm", 5.0),
            action_volumes=action_vols
        ),
        "No Irrigation": NoIrrigationPolicy()
    }

    all_records = []
    representative_trajectories = {}

    print(f"Starting evaluation across {len(scenarios)} scenarios and {len(seeds)} seeds...")
    for scenario in scenarios:
        for p_name, policy in policies.items():
            for seed in seeds:
                env = SmartIrrigationEnv(config=config, scenario=scenario, seed=seed)
                metrics, df_traj = run_episode(policy, env, deterministic=True)
                
                record = {
                    "scenario": scenario,
                    "policy": p_name,
                    "seed": seed,
                    **metrics
                }
                all_records.append(record)

                # Store first seed trajectory for visualization
                traj_key = f"{scenario}_{p_name}"
                if traj_key not in representative_trajectories:
                    representative_trajectories[traj_key] = df_traj.to_dict(orient="records")

    df_results = pd.DataFrame(all_records)
    
    # Aggregate summary: Mean and Std per Scenario and Policy
    summary_df = df_results.groupby(["scenario", "policy"]).agg({
        "cumulative_reward": ["mean", "std"],
        "total_water_mm": ["mean", "std"],
        "target_compliance_pct": ["mean", "std"],
        "deficit_stress_days": ["mean", "std"],
        "excess_water_days": ["mean", "std"],
        "num_irrigations": ["mean", "std"],
        "water_use_efficiency": ["mean", "std"]
    }).round(2)

    os.makedirs("outputs/metrics", exist_ok=True)
    
    # Save raw episode records
    raw_csv_path = "outputs/metrics/evaluation_episodes.csv"
    df_results.to_csv(raw_csv_path, index=False)
    print(f"Saved raw episode records to {raw_csv_path}")

    # Save summary dataframe
    summary_csv_path = "outputs/metrics/evaluation_summary.csv"
    summary_df.to_csv(summary_csv_path)
    print(f"Saved summary metrics to {summary_csv_path}")

    # Save representative trajectories for plotting
    traj_json_path = "outputs/metrics/representative_trajectories.json"
    with open(traj_json_path, "w") as f:
        json.dump(representative_trajectories, f, indent=2)
    print(f"Saved representative trajectories to {traj_json_path}")

    print("\n--- Evaluation Summary Overview (Normal Scenario) ---")
    normal_subset = df_results[df_results["scenario"] == "normal"].groupby("policy").mean(numeric_only=True)
    print(normal_subset[["cumulative_reward", "total_water_mm", "target_compliance_pct", "deficit_stress_days", "excess_water_days", "water_use_efficiency"]])

    return df_results, summary_df, representative_trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Agent against Baselines")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, default="models/ppo_irrigation_agent.zip", help="Path to trained PPO model")
    args = parser.parse_args()

    evaluate_all_policies(config_path=args.config, model_path=args.model)

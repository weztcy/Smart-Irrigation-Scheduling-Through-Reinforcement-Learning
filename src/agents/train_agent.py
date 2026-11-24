"""Training pipeline for Reinforcement Learning Smart Irrigation Agent."""
import os
import time
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.irrigation_env import SmartIrrigationEnv


class TrainingMetricsCallback(BaseCallback):
    """Logs episode rewards and lengths during training for plotting learning curves."""

    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_record = []
        self._current_ep_reward = 0.0
        self._current_ep_len = 0

    def _on_step(self) -> bool:
        # Check reward and done from env
        # locals()['rewards'] and locals()['dones']
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        
        self._current_ep_reward += float(rewards[0])
        self._current_ep_len += 1

        if dones[0]:
            self.episode_rewards.append(self._current_ep_reward)
            self.episode_lengths.append(self._current_ep_len)
            self.timesteps_record.append(self.num_timesteps)
            self._current_ep_reward = 0.0
            self._current_ep_len = 0

        return True


def train_agent(config_path: str = "config/config.yaml", total_timesteps: int = None, seed: int = None):
    """Train PPO agent on SmartIrrigationEnv and save weights, configs, and training logs."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cfg_seed = seed if seed is not None else int(config.get("seed", 42))
    train_cfg = config.get("training", {})
    timesteps = total_timesteps if total_timesteps is not None else int(train_cfg.get("total_timesteps", 100000))

    # Reproducibility
    np.random.seed(cfg_seed)
    torch.manual_seed(cfg_seed)

    # Prepare directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    print(f"--- Starting RL Training ---")
    print(f"Algorithm: PPO")
    print(f"Total Timesteps: {timesteps}")
    print(f"Random Seed: {cfg_seed}")

    # Create and wrap environment
    raw_env = SmartIrrigationEnv(config=config, scenario="normal", seed=cfg_seed)
    env = Monitor(raw_env)

    # Hyperparameters
    learning_rate = float(train_cfg.get("learning_rate", 0.0003))
    n_steps = int(train_cfg.get("n_steps", 1024))
    batch_size = int(train_cfg.get("batch_size", 64))
    n_epochs = int(train_cfg.get("n_epochs", 10))
    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    clip_range = float(train_cfg.get("clip_range", 0.2))
    ent_coef = float(train_cfg.get("ent_coef", 0.01))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=cfg_seed,
        verbose=1
    )

    callback = TrainingMetricsCallback()

    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")

    # Save model
    model_save_path = "models/ppo_irrigation_agent.zip"
    model.save(model_save_path)
    print(f"Model saved successfully to: {model_save_path}")

    # Verify model reloadability
    loaded_model = PPO.load(model_save_path)
    test_obs, _ = env.reset()
    test_act, _ = loaded_model.predict(test_obs, deterministic=True)
    print(f"Model reload verified! Sample prediction: action {test_act} for obs shape {test_obs.shape}")

    # Save training metrics history
    df_history = pd.DataFrame({
        "timestep": callback.timesteps_record,
        "episode_reward": callback.episode_rewards,
        "episode_length": callback.episode_lengths
    })
    
    # Calculate rolling moving average for smoothed curves
    if len(df_history) > 10:
        df_history["reward_rolling_mean"] = df_history["episode_reward"].rolling(window=min(25, len(df_history)), min_periods=1).mean()
    else:
        df_history["reward_rolling_mean"] = df_history["episode_reward"]

    csv_log_path = "outputs/logs/training_history.csv"
    df_history.to_csv(csv_log_path, index=False)
    print(f"Training history saved to: {csv_log_path}")

    # Save summary metadata
    summary = {
        "algorithm": "PPO",
        "total_timesteps": timesteps,
        "episodes_completed": len(callback.episode_rewards),
        "seed": cfg_seed,
        "training_duration_seconds": round(duration, 2),
        "mean_final_reward": round(float(np.mean(callback.episode_rewards[-20:]) if len(callback.episode_rewards) >= 20 else np.mean(callback.episode_rewards)), 3),
        "min_reward": round(float(np.min(callback.episode_rewards)), 3) if callback.episode_rewards else 0.0,
        "max_reward": round(float(np.max(callback.episode_rewards)), 3) if callback.episode_rewards else 0.0,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "net_arch": [128, 128]
        }
    }
    with open("outputs/metrics/training_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Training summary saved to outputs/metrics/training_summary.json")

    return model, df_history, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Irrigation Agent")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    train_agent(config_path=args.config, total_timesteps=args.timesteps, seed=args.seed)

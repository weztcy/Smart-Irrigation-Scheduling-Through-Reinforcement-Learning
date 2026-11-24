"""Generates all 8 mandatory scientific figures for evaluation and reporting."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.4
})

FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_training_reward_curve(history_csv: str = "outputs/logs/training_history.csv", save_path: str = f"{FIGURES_DIR}/training_reward_curve.png"):
    """Figure 1: Training Reward Curve with rolling average."""
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Raw episode rewards with low alpha
    ax.scatter(df["timestep"], df["episode_reward"], color="#6baed6", alpha=0.35, s=15, label="Raw Episode Reward")
    
    # Rolling mean trendline
    window = min(30, max(5, len(df) // 15))
    rolling_mean = df["episode_reward"].rolling(window=window, min_periods=1).mean()
    ax.plot(df["timestep"], rolling_mean, color="#08519c", linewidth=2.5, label=f"Moving Average (window={window})")

    ax.set_title("PPO Agent Training Convergence: Reward Curve over Timesteps", pad=12, fontweight="bold")
    ax.set_xlabel("Total Training Timesteps")
    ax.set_ylabel("Cumulative Episode Reward")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    
    # Format x-axis with comma separators (e.g. 20,000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Generated Figure 1: {save_path}")


def plot_soil_moisture_over_time(trajectories_json: str = "outputs/metrics/representative_trajectories.json", save_path: str = f"{FIGURES_DIR}/soil_moisture_over_time.png"):
    """Figure 2: Soil Moisture Dynamics Over 90-Day Horizon."""
    with open(trajectories_json, "r") as f:
        trajs = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Extract target band from RL agent run
    rl_data = pd.DataFrame(trajs.get("normal_RL Agent (PPO)", []))
    days = rl_data["day"].values

    # Target optimal moisture band
    ax.fill_between(days, rl_data["target_min"] * 100, rl_data["target_max"] * 100, 
                    color="#c7e9c0", alpha=0.5, label="Optimal Target Band (FAO Stage-Specific)")

    # Severe stress threshold (wilting point = 20%)
    ax.axhline(20.0, color="#de2d26", linestyle=":", linewidth=1.5, label="Permanent Wilting Point (20%)")
    # Field capacity line (80%)
    ax.axhline(80.0, color="#3182bd", linestyle="--", linewidth=1.2, label="Field Capacity (80%)")

    # Plot trajectories for policies
    colors = {
        "normal_RL Agent (PPO)": ("#006d2c", "-", 2.5, "RL Agent (PPO)"),
        "normal_Threshold-Based": ("#ff7f00", "--", 1.8, "Threshold-Based (50%)"),
        "normal_Fixed Schedule (3-Day)": ("#756bb1", "-.", 1.8, "Fixed Schedule (3-Day)"),
        "normal_No Irrigation": ("#969696", ":", 1.5, "No Irrigation (Rainfed)")
    }

    for key, (color, ls, lw, label) in colors.items():
        if key in trajs:
            df = pd.DataFrame(trajs[key])
            ax.plot(df["day"], df["soil_moisture_ratio"] * 100, color=color, linestyle=ls, linewidth=lw, label=label)

    ax.set_title("Soil Moisture Trajectory Across 90-Day Crop Growth Season (Normal Weather)", pad=12, fontweight="bold")
    ax.set_xlabel("Day of Growth Season")
    ax.set_ylabel("Soil Moisture Content (% Capacity)")
    ax.set_ylim(5, 95)
    ax.set_xlim(1, 90)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Generated Figure 2: {save_path}")


def plot_irrigation_action_timeline(trajectories_json: str = "outputs/metrics/representative_trajectories.json", save_path: str = f"{FIGURES_DIR}/irrigation_action_timeline.png"):
    """Figure 3: Irrigation Events and Action Depths Timeline."""
    with open(trajectories_json, "r") as f:
        trajs = json.load(f)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, dpi=300)

    policies = [
        ("normal_RL Agent (PPO)", "RL Agent (PPO)", "#006d2c"),
        ("normal_Threshold-Based", "Threshold-Based Policy", "#ff7f00"),
        ("normal_Fixed Schedule (3-Day)", "Fixed Schedule Policy", "#756bb1")
    ]

    for idx, (key, title, color) in enumerate(policies):
        ax = axes[idx]
        if key in trajs:
            df = pd.DataFrame(trajs[key])
            ax.bar(df["day"], df["action_volume_mm"], color=color, width=0.8, alpha=0.85, label="Applied Irrigation (mm)")
            
            # Secondary axis for rainfall
            ax2 = ax.twinx()
            ax2.plot(df["day"], df["rainfall_mm"], color="#3182bd", linestyle=":", alpha=0.5, label="Natural Rainfall (mm)")
            ax2.set_ylabel("Rain (mm)", color="#3182bd", fontsize=9)
            ax2.set_ylim(0, 45)
            ax2.tick_params(axis='y', labelcolor="#3182bd")

            total_applied = df["action_volume_mm"].sum()
            event_count = (df["action_volume_mm"] > 0).sum()
            ax.set_title(f"{title} (Total: {total_applied:.1f} mm across {event_count} irrigations)", fontsize=11, fontweight="bold", loc="left")
            ax.set_ylabel("Irrig (mm)", fontsize=10)
            ax.set_ylim(0, 25)
            ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Day of Growth Season")
    fig.suptitle("Daily Irrigation Action Timeline Comparison (vs Natural Rainfall)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Generated Figure 3: {save_path}")


def plot_water_usage_comparison(episodes_csv: str = "outputs/metrics/evaluation_episodes.csv", save_path: str = f"{FIGURES_DIR}/water_usage_comparison.png"):
    """Figure 4: Total Water Usage Comparison Across Policies."""
    df = pd.read_csv(episodes_csv)
    # Filter to normal scenario for fair primary comparison
    df_normal = df[df["scenario"] == "normal"]
    agg = df_normal.groupby("policy")["total_water_mm"].agg(["mean", "std"]).reset_index()

    # Sort policies for clean presentation
    policy_order = ["No Irrigation", "RL Agent (PPO)", "Smart Threshold", "Threshold-Based", "Fixed Schedule (3-Day)"]
    agg["order"] = agg["policy"].map(lambda x: policy_order.index(x) if x in policy_order else 99)
    agg = agg.sort_values("order")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars = ax.bar(agg["policy"], agg["mean"], yerr=agg["std"], capsize=5, 
                  color=["#bdbdbd", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"], 
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    # Add numeric labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f} mm",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Total Seasonal Water Consumption per Policy (Normal Weather, 90 Days)", pad=12, fontweight="bold")
    ax.set_ylabel("Total Irrigation Water Applied (mm)")
    ax.set_xlabel("Policy / Method")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Generated Figure 4: {save_path}")


def plot_moisture_target_compliance(episodes_csv: str = "outputs/metrics/evaluation_episodes.csv", save_path: str = f"{FIGURES_DIR}/moisture_target_compliance.png"):
    """Figure 5: Target Moisture Compliance (% of Season in Optimal Range)."""
    df = pd.read_csv(episodes_csv)
    df_normal = df[df["scenario"] == "normal"]
    agg = df_normal.groupby("policy")["target_compliance_pct"].agg(["mean", "std"]).reset_index()

    policy_order = ["RL Agent (PPO)", "Smart Threshold", "Threshold-Based", "Fixed Schedule (3-Day)", "No Irrigation"]
    agg["order"] = agg["policy"].map(lambda x: policy_order.index(x) if x in policy_order else 99)
    agg = agg.sort_values("order")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars = ax.bar(agg["policy"], agg["mean"], yerr=agg["std"], capsize=5,
                  color=["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#bdbdbd"],
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Crop Moisture Target Compliance (% of Days in Optimal Range)", pad=12, fontweight="bold")
    ax.set_ylabel("Compliance Rate (%)")
    ax.set_xlabel("Policy / Method")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Generated Figure 5: {save_path}")


def plot_underwatering_and_overwatering(episodes_csv: str = "outputs/metrics/evaluation_episodes.csv", save_path: str = f"{FIGURES_DIR}/underwatering_and_overwatering.png"):
    """Figure 6: Under-watering vs Over-watering Days Comparison."""
    df = pd.read_csv(episodes_csv)
    df_normal = df[df["scenario"] == "normal"]
    agg = df_normal.groupby("policy")[["deficit_stress_days", "excess_water_days"]].mean().reset_index()

    policy_order = ["RL Agent (PPO)", "Smart Threshold", "Threshold-Based", "Fixed Schedule (3-Day)", "No Irrigation"]
    agg["order"] = agg["policy"].map(lambda x: policy_order.index(x) if x in policy_order else 99)
    agg = agg.sort_values("order")

    x = np.arange(len(agg))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    b1 = ax.bar(x - width/2, agg["deficit_stress_days"], width, label="Under-watering / Deficit Days", color="#d73027", alpha=0.85, edgecolor="black")
    b2 = ax.bar(x + width/2, agg["excess_water_days"], width, label="Over-watering / Excess Days", color="#4575b4", alpha=0.85, edgecolor="black")

    for bar in list(b1) + list(b2):
        height = bar.get_height()
        if height > 0.5:
            ax.annotate(f"{height:.1f}d",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_title("Stress Frequency: Under-watering vs Over-watering Days (90 Days Total)", pad=12, fontweight="bold")
    ax.set_ylabel("Number of Stress Days")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["policy"], rotation=15, ha="right")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Generated Figure 6: {save_path}")


def plot_weather_irrigation_relationship(trajectories_json: str = "outputs/metrics/representative_trajectories.json", save_path: str = f"{FIGURES_DIR}/weather_irrigation_relationship.png"):
    """Figure 7: Relationship Between Weather Conditions and RL Irrigation Actions."""
    with open(trajectories_json, "r") as f:
        trajs = json.load(f)

    rl_data = pd.DataFrame(trajs.get("normal_RL Agent (PPO)", []))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Action vs Rainfall
    scatter = ax1.scatter(rl_data["rainfall_mm"], rl_data["action_volume_mm"], 
                          c=rl_data["soil_moisture_ratio"] * 100, cmap="viridis", s=45, alpha=0.85, edgecolors="none")
    ax1.set_title("RL Irrigation Action vs Daily Rainfall", fontweight="bold")
    ax1.set_xlabel("Daily Rainfall (mm)")
    ax1.set_ylabel("RL Chosen Irrigation (mm)")
    cbar1 = fig.colorbar(scatter, ax=ax1)
    cbar1.set_label("Soil Moisture (% Capacity)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. Action distribution by weather state
    weather_irr = rl_data.groupby("weather_state")["action_volume_mm"].mean().reset_index()
    colors = {"Sunny": "#e66101", "Cloudy": "#998ec3", "Rainy": "#542788"}
    bar_colors = [colors.get(w, "#4575b4") for w in weather_irr["weather_state"]]

    bars = ax2.bar(weather_irr["weather_state"], weather_irr["action_volume_mm"], color=bar_colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f} mm/d", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_title("Average Daily Irrigation by Weather State", fontweight="bold")
    ax2.set_xlabel("Weather Condition")
    ax2.set_ylabel("Mean Irrigation Depth (mm/day)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Adaptive Behavior: RL Agent Interaction with Climate Dynamics", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Generated Figure 7: {save_path}")


def plot_robustness_comparison(episodes_csv: str = "outputs/metrics/evaluation_episodes.csv", save_path: str = f"{FIGURES_DIR}/robustness_comparison.png"):
    """Figure 8: Robustness Comparison Across Climate Scenarios."""
    df = pd.read_csv(episodes_csv)
    # Focus on RL Agent vs Key Baselines
    target_policies = ["RL Agent (PPO)", "Smart Threshold", "Threshold-Based", "Fixed Schedule (3-Day)"]
    df_sub = df[df["policy"].isin(target_policies)]

    # Compute target compliance per scenario
    agg = df_sub.groupby(["scenario", "policy"])["target_compliance_pct"].mean().unstack()
    scenarios_order = ["normal", "dry", "wet", "heatwave"]
    agg = agg.reindex([s for s in scenarios_order if s in agg.index])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(agg.index))
    width = 0.2
    
    palette = {
        "RL Agent (PPO)": "#2ca02c",
        "Smart Threshold": "#1f77b4",
        "Threshold-Based": "#ff7f0e",
        "Fixed Schedule (3-Day)": "#9467bd"
    }

    for i, col in enumerate(target_policies):
        if col in agg.columns:
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, agg[col], width, label=col, color=palette.get(col, "#333333"), edgecolor="black", alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 5:
                    ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_title("Robustness Evaluation: Target Compliance Across Climate Scenarios", pad=12, fontweight="bold")
    ax.set_ylabel("Moisture Target Compliance (%)")
    ax.set_xlabel("Climate Scenario")
    ax.set_xticks(x)
    scenario_labels = [s.capitalize() for s in agg.index]
    ax.set_xticklabels(scenario_labels)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Generated Figure 8: {save_path}")


def generate_all_visualizations(history_csv: str = "outputs/logs/training_history.csv",
                                episodes_csv: str = "outputs/metrics/evaluation_episodes.csv",
                                trajectories_json: str = "outputs/metrics/representative_trajectories.json"):
    """Execute generation of all 8 mandatory scientific figures."""
    print("--- Generating Scientific Visualizations ---")
    plot_training_reward_curve(history_csv)
    plot_soil_moisture_over_time(trajectories_json)
    plot_irrigation_action_timeline(trajectories_json)
    plot_water_usage_comparison(episodes_csv)
    plot_moisture_target_compliance(episodes_csv)
    plot_underwatering_and_overwatering(episodes_csv)
    plot_weather_irrigation_relationship(trajectories_json)
    plot_robustness_comparison(episodes_csv)
    print("All 8 figures successfully generated in outputs/figures/!")


if __name__ == "__main__":
    generate_all_visualizations()

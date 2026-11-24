"""Smart Irrigation Scheduling Through Reinforcement Learning - Streamlit Application."""
import os
import json
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.environment.irrigation_env import SmartIrrigationEnv
from src.baselines.baseline_policies import (
    FixedSchedulePolicy,
    ThresholdBasedPolicy,
    SmartThresholdPolicy,
    NoIrrigationPolicy
)
from src.evaluation.evaluate import run_episode

# Configure page settings
st.set_page_config(
    page_title="Smart Irrigation RL Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom sleek dark mode CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-highlight {
        color: #10b981;
    }
    .badge-rl {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid #10b981;
    }
    .badge-fixed {
        background-color: rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid #64748b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config_and_model():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_path = "models/ppo_irrigation_agent.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        model = None
    return config, model


config, model = load_config_and_model()

# Sidebar Navigation
st.sidebar.image("https://images.unsplash.com/photo-1586771107445-d3ca888129ff?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.title("🌱 Smart Irrigation RL")
st.sidebar.caption("Precision Agriculture via Deep RL")

page_options = [
    "🌿 Live Simulation & Comparison",
    "📊 Benchmark & Robustness Gallery",
    "⚙️ System Architecture & Hydrology MDP"
]

# Robust query parameter routing: ?p=1 (or simulation), ?p=2 (or benchmark), ?p=3 (or architecture)
qp = st.query_params
p_val = str(qp.get("page", qp.get("p", ""))).lower()

if "bench" in p_val or "gallery" in p_val or p_val == "2":
    initial_index = 1
elif "arch" in p_val or "model" in p_val or "mdp" in p_val or p_val == "3":
    initial_index = 2
else:
    initial_index = 0

page = st.sidebar.radio(
    "Navigation Menu",
    page_options,
    index=initial_index
)

# ---------------- PAGE 1: LIVE SIMULATION & COMPARISON ----------------
if page == "🌿 Live Simulation & Comparison":
    st.header("🌿 Interactive 90-Day Irrigation Simulation & Baseline Benchmark")
    st.markdown("""
    Explore how the **Reinforcement Learning Agent (PPO)** makes adaptive daily irrigation decisions compared to 
    traditional agricultural heuristics under dynamic weather conditions.
    """)

    # Controls
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        scenario_choice = st.selectbox("Climate Scenario", ["normal", "dry", "wet", "heatwave"], index=0,
                                       help="Select climatic regime for the 90-day seasonal crop cycle.")
    with col_c2:
        seed_choice = st.number_input("Random Seed", min_value=1, max_value=9999, value=101, step=1,
                                      help="Controls stochastic weather patterns and initial moisture state.")
    with col_c3:
        initial_moisture_override = st.slider("Initial Soil Moisture (%)", 20, 80, 50, step=5,
                                              help="Override initial soil water fraction at day 0.")
    with col_c4:
        run_button = st.button("🚀 Run Comparative Simulation", type="primary", use_container_width=True)

    if model is None:
        st.error("Trained RL model not found in `models/ppo_irrigation_agent.zip`. Please run training first.")
    else:
        # Run simulation for all policies
        with st.spinner("Running 90-day simulation across all policies..."):
            policies = {
                "RL Agent (PPO)": model,
                "Fixed Schedule (3-Day)": FixedSchedulePolicy(interval_days=3, volume_mm=10.0),
                "Threshold-Based": ThresholdBasedPolicy(lower_threshold_ratio=0.50, volume_mm=10.0),
                "Smart Threshold": SmartThresholdPolicy(lower_threshold_ratio=0.55, volume_mm=10.0, rain_threshold_skip_mm=5.0),
                "No Irrigation (Rainfed)": NoIrrigationPolicy()
            }

            results = {}
            trajectories = {}

            for p_name, policy in policies.items():
                env = SmartIrrigationEnv(config=config, scenario=scenario_choice, seed=seed_choice)
                env.soil_moisture_mm = (initial_moisture_override / 100.0) * env.soil_capacity_mm
                metrics, df_traj = run_episode(policy, env, deterministic=True)
                results[p_name] = metrics
                trajectories[p_name] = df_traj

        # Top KPI cards
        rl_res = results["RL Agent (PPO)"]
        fix_res = results["Fixed Schedule (3-Day)"]
        water_saved_pct = ((fix_res["total_water_mm"] - rl_res["total_water_mm"]) / max(1.0, fix_res["total_water_mm"])) * 100.0

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Water Conservation</div>
                <div class="metric-value metric-highlight">{water_saved_pct:+.1f}%</div>
                <small style="color: #94a3b8">vs Fixed Schedule ({rl_res['total_water_mm']:.1f} vs {fix_res['total_water_mm']:.1f} mm)</small>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Target Moisture Compliance</div>
                <div class="metric-value">{rl_res['target_compliance_pct']:.1f}%</div>
                <small style="color: #94a3b8">{rl_res['target_compliance_pct'] - fix_res['target_compliance_pct']:+.1f}% vs Fixed Schedule</small>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Drought Deficit Days</div>
                <div class="metric-value">{rl_res['deficit_stress_days']} <span style="font-size: 1rem; color: #94a3b8">days</span></div>
                <small style="color: #94a3b8">{fix_res['deficit_stress_days']} days on Fixed Schedule</small>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Cumulative Reward</div>
                <div class="metric-value metric-highlight">{rl_res['cumulative_reward']:.1f}</div>
                <small style="color: #94a3b8">Overall multi-objective efficiency</small>
            </div>
            """, unsafe_allow_html=True)

        # Plot dynamic soil moisture trajectory
        st.subheader("📈 Daily Soil Moisture Dynamics & Crop Target Bands")
        rl_traj = trajectories["RL Agent (PPO)"]
        days = rl_traj["day"].values

        fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
        fig.patch.set_facecolor("#0b0f19")
        ax.set_facecolor("#1e293b")

        # FAO stage target band
        ax.fill_between(days, rl_traj["target_min"] * 100, rl_traj["target_max"] * 100,
                        color="#10b981", alpha=0.25, label="FAO Target Band (Optimal Growth)")

        # Wilting point & Field capacity
        ax.axhline(20.0, color="#ef4444", linestyle=":", linewidth=1.2, label="Wilting Point (20%)")
        ax.axhline(80.0, color="#38bdf8", linestyle="--", linewidth=1.0, label="Field Capacity (80%)")

        # Policy curves
        ax.plot(days, rl_traj["soil_moisture_ratio"] * 100, color="#10b981", linewidth=2.5, label="RL Agent (PPO)")
        ax.plot(days, trajectories["Threshold-Based"]["soil_moisture_ratio"] * 100, color="#f59e0b", linestyle="--", linewidth=1.5, label="Threshold-Based (50%)")
        ax.plot(days, trajectories["Fixed Schedule (3-Day)"]["soil_moisture_ratio"] * 100, color="#a855f7", linestyle="-.", linewidth=1.5, label="Fixed Schedule (3-Day)")
        ax.plot(days, trajectories["No Irrigation (Rainfed)"]["soil_moisture_ratio"] * 100, color="#64748b", linestyle=":", linewidth=1.2, label="Rainfed (No Irrig)")

        ax.set_xlabel("Day in Crop Season", color="#f1f5f9", fontsize=10)
        ax.set_ylabel("Soil Moisture (% Capacity)", color="#f1f5f9", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xlim(1, 90)
        ax.tick_params(colors="#94a3b8", which="both")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
        ax.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=8)

        st.pyplot(fig)
        plt.close(fig)

        # Comparative Summary Table
        st.subheader("📋 Performance Scorecard Across Policies")
        summary_rows = []
        for p_name, m in results.items():
            summary_rows.append({
                "Policy": p_name,
                "Total Water Applied (mm)": m["total_water_mm"],
                "Target Compliance (%)": f"{m['target_compliance_pct']:.1f}%",
                "Deficit Days": m["deficit_stress_days"],
                "Excess Water Days": m["excess_water_days"],
                "Irrigations": m["num_irrigations"],
                "Water Use Efficiency": m["water_use_efficiency"],
                "Episode Reward": m["cumulative_reward"]
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # Daily Decision Log inspection
        with st.expander("🔍 Inspect Detailed Daily Decision Log (RL Agent)"):
            st.dataframe(rl_traj[["day", "stage_name", "weather_state", "temperature_c", "rainfall_mm", "etc_mm", "action_volume_mm", "soil_moisture_ratio", "in_target", "step_reward"]], use_container_width=True)


# ---------------- PAGE 2: BENCHMARK GALLERY ----------------
elif page == "📊 Benchmark & Robustness Gallery":
    st.header("📊 Scientific Benchmark & Robustness Gallery")
    st.markdown("""
    Pre-computed experimental results across **4 climate regimes** (*Normal, Dry, Wet, Heatwave*) 
    evaluated over multiple random seeds.
    """)

    figures_meta = [
        ("training_reward_curve.png", "Figure 1: PPO Training Convergence Curve",
         "Demonstrates stable policy improvement and convergence over 100,000 timesteps as rolling episode reward improves from negative stress penalties to a steady positive plateau (~+50.0)."),
        ("soil_moisture_over_time.png", "Figure 2: Soil Moisture Trajectory Across 90 Days",
         "Depicts how the RL agent anticipates stage-specific evapotranspiration and maintains soil moisture tightly within optimal growth bounds, preventing both water stress and leaching."),
        ("irrigation_action_timeline.png", "Figure 3: Irrigation Actions Timeline vs Rainfall",
         "Illustrates the timing and depth of irrigation actions chosen by the RL agent compared to fixed and threshold heuristics, highlighting reactive avoidance of irrigating during precipitation events."),
        ("water_usage_comparison.png", "Figure 4: Total Water Consumption Comparison",
         "Compares aggregate irrigation water consumed across 90 days. RL consumes significantly less water than fixed schedules while preserving plant health."),
        ("moisture_target_compliance.png", "Figure 5: Target Moisture Compliance Rate",
         "Shows the percentage of total season days maintained within stage-specific optimal water bands. RL achieves superior compliance across all tests."),
        ("underwatering_and_overwatering.png", "Figure 6: Under-watering vs Over-watering Days",
         "Contrasts the frequency of drought stress days and saturated over-watering days, highlighting the multi-objective balance of the RL policy."),
        ("weather_irrigation_relationship.png", "Figure 7: Adaptive Behavior vs Meteorological Dynamics",
         "Shows how the RL policy dynamically adjusts irrigation depth in response to rainfall and sunny vs cloudy weather states."),
        ("robustness_comparison.png", "Figure 8: Climate Robustness Across Scenarios",
         "Demonstrates policy robustness under unseen or extreme climate conditions: Normal, Dry (Drought), Wet (Monsoon), and Heatwave.")
    ]

    for fname, title, desc in figures_meta:
        fpath = os.path.join("outputs/figures", fname)
        st.subheader(title)
        if os.path.exists(fpath):
            st.image(fpath, caption=desc, use_container_width=True)
        else:
            st.warning(f"Figure file `{fpath}` not found.")
        st.divider()


# ---------------- PAGE 3: ARCHITECTURE & METHODOLOGY ----------------
elif page == "⚙️ System Architecture & Hydrology MDP":
    st.header("⚙️ System Architecture & Hydrological MDP Formulation")
    st.markdown("""
    ### 1. Markov Decision Process (MDP) Definition
    The smart irrigation problem is formulated as a discrete-time Markov Decision Process $\\mathcal{M} = \\langle \\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma \\rangle$:

    - **State Space $\\mathcal{S} \\in \\mathbb{R}^7$:**
      1. Normalized Soil Moisture Ratio ($S_t / S_{\\text{capacity}}$)
      2. Normalized Temperature ($(T_t - 15) / 30$)
      3. Normalized Daily Rainfall ($P_t / 50$)
      4. Weather State (0=Sunny, 0.5=Cloudy, 1.0=Rainy)
      5. Crop Growth Stage Index ($k / 3$)
      6. Crop Evapotranspiration Demand ($ET_{c,t} / 10$)
      7. Normalized Day Index ($t / T$)

    - **Action Space $\\mathcal{A} = \\{0, 1, 2, 3\\}$:**
      - $a=0$: No irrigation (0.0 mm)
      - $a=1$: Light irrigation (5.0 mm)
      - $a=2$: Medium irrigation (10.0 mm)
      - $a=3$: Heavy irrigation (20.0 mm)

    - **Soil Water Balance Dynamics:**
      $$S_{t+1} = \\text{clip}\\left(S_t + I_t + P_{\\text{eff}, t} - ET_{c,t} - \\text{Drainage}_t, 0, S_{\\text{cap}}\\right)$$
      Where $ET_{c,t} = K_c(\\text{stage}) \\times ET_{0,t}$ and $P_{\\text{eff}, t}$ is precipitation adjusted for surface runoff.

    - **Reward Function $\\mathcal{R}(s, a)$:**
      $$R_t = R_{\\text{target}} - \\lambda_{\\text{deficit}} \\cdot \\Delta_{\\text{deficit}} \\cdot M_{\\text{stage}} - \\lambda_{\\text{excess}} \\cdot \\Delta_{\\text{excess}} - \\lambda_{\\text{water}} \\cdot I_t - \\lambda_{\\text{waste}} \\cdot \\text{Waste}_t$$
    """)

    st.subheader("🌾 FAO-56 Crop Growth Stage Parameters")
    stages_data = [
        {"Stage": "1. Initial", "Days": "0 - 20", "Crop Coeff (Kc)": 0.45, "Target Moisture": "45% - 65%", "Stress Weight": "1.0x (Moderate)"},
        {"Stage": "2. Development", "Days": "21 - 50", "Crop Coeff (Kc)": 0.85, "Target Moisture": "55% - 75%", "Stress Weight": "1.2x (Substantial)"},
        {"Stage": "3. Mid-Season (Flowering)", "Days": "51 - 75", "Crop Coeff (Kc)": 1.15, "Target Moisture": "65% - 85%", "Stress Weight": "2.0x (Critical)"},
        {"Stage": "4. Late-Season (Ripening)", "Days": "76 - 90", "Crop Coeff (Kc)": 0.60, "Target Moisture": "45% - 65%", "Stress Weight": "0.8x (Low)"}
    ]
    st.table(pd.DataFrame(stages_data))

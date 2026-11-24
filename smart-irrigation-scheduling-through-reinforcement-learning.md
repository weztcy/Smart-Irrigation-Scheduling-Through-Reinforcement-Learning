# Portfolio Project Data — Smart Irrigation Scheduling Through Reinforcement Learning

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/smart_irrigation_scheduling_through/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image08.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "smart-irrigation-scheduling-through-reinforcement-learning",
  "code": "PRO-33 // ML-18",
  "title": "Smart Irrigation Scheduling Through Reinforcement Learning",
  "category": "machine-learning",
  "categoryLabel": "Reinforcement Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Smart-Irrigation-Scheduling-Through-Reinforcement-Learning",
  "license": "MIT License",
  "banner": "/images/projects/smart_irrigation_scheduling_through/banner.webp",
  "summary": "PPO reinforcement learning controller optimizing 90-day crop irrigation, saving 59.67% water while achieving 72.45%–91.56% target moisture compliance.",
  "projectDescription": "An end-to-end Reinforcement Learning system formulating 90-day irrigation as an FAO-56 Markov Decision Process. A PPO actor-critic agent learns weather-aware scheduling policies, cutting water use by 59.67% over fixed schedules and sustaining over 91% compliance in droughts.",
  "architectureSummary": "Config YAML -> 3-State Markov Weather & FAO-56 Gymnasium Env -> PPO Actor-Critic MLP -> Multi-Seed Evaluation & Streamlit Dashboard",
  "techStack": [
    "Python",
    "Gymnasium",
    "Stable-Baselines3",
    "PyTorch",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "PyYAML",
    "Pytest",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Reinforcement Learning Control Pipeline",
    "targetPlatform": "Python Runtime & Streamlit Dashboard",
    "primaryDomain": "Precision Agriculture & Hydrology",
    "architecturePipeline": "PPO Actor-Critic over FAO-56 MDP",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Sequential Irrigation Control Pipeline",
    "learningParadigm": "Reinforcement Learning",
    "modelDomain": "Agro-Hydrological Decision Control",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Adaptive Irrigation Control",
      "Multi-Scenario Robustness",
      "Water Conservation Policy"
    ]
  },
  "mlProblemContext": "Fixed calendar irrigation and static sensor thresholds fail to anticipate phenological crop water demand and incoming rain, leading to severe water waste, root-zone waterlogging, or drought stress during sensitive flowering windows. Optimizing daily irrigation requires sequential, weather-aware trade-offs across an entire 90-day growing season.",
  "mlDatasetOverview": {
    "profileBadge": "Gymnasium MDP Simulation",
    "source": "Synthetic FAO-56 Agro-Hydrological Simulator (SmartIrrigationEnv & WeatherModel)",
    "modality": "Sequential State-Action Trajectory",
    "scale": "1,115 Training Episodes (100,000 Steps) & 100 Evaluation Episodes (9,000 Days)",
    "inputShape": "7 Normalized State Features (float32, [0.0, 1.0])",
    "dataFormat": "Gymnasium Box / Discrete + CSV & JSON Logs",
    "description": "Experiences are generated dynamically over 90-day crop cycles spanning four phenological stages (Initial, Development, Mid-Season Flowering, Late-Season Ripening) and four stochastic Markov weather regimes (Normal, Dry, Wet, Heatwave), with evaluation logs persisted across 5 seeds.",
    "provenanceNote": "Self-contained synthetic agro-hydrological simulator parameterized by FAO-56 crop coefficients in config/config.yaml; no external private dataset required."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "7D State & 4-Action MDP",
    "spaceNote": "All observations are clipped to [0, 1] without future weather leakage.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Discrete Daily Irrigation Action (0, 5, 10, 20 mm)",
      "type": "Discrete Control Policy (4 Actions)",
      "distribution": "0.0 mm (No Irrigation), 5.0 mm (Light), 10.0 mm (Medium), 20.0 mm (Heavy)"
    },
    "inputSignals": [
      {
        "name": "Soil Moisture Ratio",
        "type": "Continuous Normalized",
        "importanceOrRole": "Primary Root-Zone State",
        "description": "Root-zone water storage divided by 100.0 mm soil capacity, bounded between wilting point (20%) and saturation (100%)."
      },
      {
        "name": "Crop Water Demand (ETc)",
        "type": "Continuous Normalized",
        "importanceOrRole": "Phenological Evapotranspiration Driver",
        "description": "Daily crop evapotranspiration Kc(stage) * ET0 scaled by 10.0 mm/day, reflecting stage-specific water consumption."
      },
      {
        "name": "Growth Stage Index",
        "type": "Ordinal Normalized",
        "importanceOrRole": "Anticipatory Target & Stress Weight",
        "description": "Encodes the 4 FAO-56 crop phases (Initial through Late-Season) whose target moisture bands rise to 65%–85% during flowering."
      },
      {
        "name": "Daily Rainfall & Weather State",
        "type": "Continuous & Categorical Normalized",
        "importanceOrRole": "Precipitation & Solar Radiation Signal",
        "description": "Normalized daily rainfall (0–50 mm) and Markov sky condition (Sunny, Cloudy, Rainy) used to skip unnecessary watering."
      },
      {
        "name": "Normalized Air Temperature & Season Day",
        "type": "Continuous Normalized",
        "importanceOrRole": "Thermal & Horizon Context",
        "description": "Daily air temperature ((T - 15) / 30) and normalized seasonal day index (t / 90) across the finite 90-day horizon."
      }
    ],
    "dataCharacteristicsSummary": "Each 90-day episode couples Markov-chain weather transitions with mass-conserving water balance dynamics (runoff above 10 mm rain and 50% daily drainage above 80% field capacity). Non-stationary Kc values (0.45 to 1.15) and 2.0x flowering stress penalties create tight temporal dependencies across seasons."
  },
  "mlLearningApproach": {
    "paradigm": "Episodic Reinforcement Learning (Proximal Policy Optimization)",
    "methodologyDescription": "Daily irrigation is formulated as a finite-horizon Markov Decision Process solved via PPO with a clipped surrogate objective and Generalized Advantage Estimation. Separate 128x128 Tanh actor and critic networks learn anticipatory watering policies directly from multi-objective reward feedback.",
    "justification": "PPO stabilizes discrete policy updates under stochastic rainfall transitions compared to unclipped policy gradients and learns forward-looking pre-irrigation before high-Kc flowering stages, outperforming reactive threshold heuristics."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Markov Weather & ET0 Simulation",
      "description": "Generate daily sky state (Sunny, Cloudy, Rainy) via 3x3 scenario transition matrices, sample state-conditioned temperature and precipitation, and compute Hargreaves-inspired ET0.",
      "badge": "Weather Model"
    },
    {
      "step": "02",
      "title": "FAO-56 Water Balance & Runoff Update",
      "description": "Apply 15% surface runoff attenuation for intense rainfall (>10 mm), update root-zone storage with irrigation and ETc, and drain 50% of excess water above 80% field capacity.",
      "badge": "Hydrology Physics"
    },
    {
      "step": "03",
      "title": "Leakage-Free Observation Normalization",
      "description": "Clip soil moisture, temperature, rainfall, weather state, growth stage, ETc demand, and season day strictly to [0.0, 1.0] float32 vectors prior to policy inference.",
      "badge": "State Scaling"
    }
  ],
  "mlDataPreparationNote": "Observations expose only current-day meteorological and phenological state variables with zero future rainfall or temperature leakage; evaluation uses five held-out seeds across four climate regimes.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Proximal Policy Optimization (PPO)",
    "architectureSummary": "Stable-Baselines3 MlpPolicy using decoupled actor (pi=[128, 128]) and critic (vf=[128, 128]) multi-layer perceptrons with Tanh activations mapping 7 normalized state features to 4 discrete irrigation logits and scalar value estimates.",
    "configurationBadge": "PPO 100K Steps",
    "hyperparameters": [
      {
        "name": "total_timesteps",
        "value": "100,000",
        "description": "Total environment interactions (~1,115 full 90-day crop episodes)."
      },
      {
        "name": "learning_rate",
        "value": "0.0003",
        "description": "Adam optimizer step size for actor and critic updates."
      },
      {
        "name": "n_steps / batch_size",
        "value": "1,024 / 64",
        "description": "Rollout buffer horizon and mini-batch size across 10 optimization epochs."
      },
      {
        "name": "gamma / gae_lambda",
        "value": "0.99 / 0.95",
        "description": "Long-horizon discount factor and GAE advantage smoothing parameter."
      },
      {
        "name": "clip_range / ent_coef",
        "value": "0.2 / 0.01",
        "description": "Surrogate policy clipping bound and entropy bonus for exploration."
      }
    ],
    "objectiveCriterion": "Clipped Surrogate Policy Objective + Value MSE - Entropy Bonus",
    "optimizationSolver": "Adam Optimizer (Stable-Baselines3 PyTorch)",
    "trainingProcess": "Trained on the normal weather regime (seed 42) wrapped with Monitor for 108.42 seconds on CPU. Episode returns climbed from -51.896 during early random exploration to a converged plateau around +48.214 (peak +74.413) across 1,115 completed episodes, with checkpoint weights verified via reloadable inference."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Deterministic Multi-Scenario Multi-Seed Benchmark",
    "description": "Evaluated deterministically across 4 climate regimes (Normal, Dry, Wet, Heatwave) and 5 independent seeds ([101, 202, 303, 404, 505]) totaling 100 episodes (9,000 simulated days) against Fixed Schedule, Threshold-Based, Smart Threshold, and No Irrigation baselines.",
    "metrics": [
      {
        "label": "NORMAL COMPLIANCE",
        "value": "72.45%",
        "benchmark": "Baseline: 32.67%"
      },
      {
        "label": "WATER SAVINGS",
        "value": "59.67%",
        "benchmark": "Baseline: 300.0 mm"
      },
      {
        "label": "DRY COMPLIANCE",
        "value": "91.56%",
        "benchmark": "Target > 85.0%"
      },
      {
        "label": "HEATWAVE COMPLIANCE",
        "value": "91.11%",
        "benchmark": "Target > 85.0%"
      },
      {
        "label": "NORMAL DEFICIT DAYS",
        "value": "1.8 days",
        "benchmark": "Target < 5.0 days"
      },
      {
        "label": "NORMAL REWARD",
        "value": "+54.63",
        "benchmark": "Baseline: +31.67"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Interactive 90-Day Simulation & Baseline Benchmark Dashboard",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/smart_irrigation_scheduling_through/image01.webp",
      "telemetryNote": "Displays +66.7% water savings and 68.9% target compliance on Normal seed 101."
    },
    {
      "title": "Scientific Benchmark Gallery & PPO Training View",
      "badge": "Streamlit Gallery",
      "imageUrl": "/images/projects/smart_irrigation_scheduling_through/image02.webp",
      "telemetryNote": "Embeds pre-computed 300 DPI plots and convergence summaries across 4 climates."
    },
    {
      "title": "System Architecture & Hydrological MDP View",
      "badge": "MDP Docs",
      "imageUrl": "/images/projects/smart_irrigation_scheduling_through/image03.webp",
      "telemetryNote": "Documents the 7D state space, 4 discrete actions, water balance, and FAO-56 table."
    },
    {
      "title": "Soil Moisture Trajectory Across 90-Day Crop Season",
      "badge": "Trajectory Plot",
      "imageUrl": "/images/projects/smart_irrigation_scheduling_through/image04.webp",
      "telemetryNote": "PPO tracks the shifting 45%–85% FAO optimal band while avoiding wilting stress."
    },
    {
      "title": "PPO Agent Training Convergence Curve",
      "badge": "Learning Curve",
      "imageUrl": "/images/projects/smart_irrigation_scheduling_through/image05.webp",
      "telemetryNote": "30-episode moving average climbs from -46.8 to +48.214 over 100,000 steps."
    },
    {
      "title": "Daily Irrigation Action Timeline vs. Rainfall",
      "badge": "Action Timeline",
      "imageUrl": "/images/projects/smart_irrigation_scheduling_through/image06.webp",
      "telemetryNote": "PPO applies 100.0 mm across 13 events while skipping rainy days unlike Fixed Schedule."
    },
    {
      "title": "Adaptive Behavior: Irrigation vs. Weather Dynamics",
      "badge": "Policy Analysis",
      "imageUrl": "/images/projects/smart_irrigation_scheduling_through/image07.webp",
      "telemetryNote": "Mean irrigation scales from 0.00 mm/d (Rainy) to 0.54 (Cloudy) and 2.16 mm/d (Sunny)."
    },
    {
      "title": "Multi-Metric & Cross-Climate Robustness Summary",
      "badge": "Robustness Grid",
      "imageUrl": "/images/projects/smart_irrigation_scheduling_through/image08.webp",
      "telemetryNote": "Sustains 92% Dry and 91% Heatwave compliance while cutting deficit days to 1.8d."
    }
  ],
  "mlVisualizationTelemetryNote": "Training converged smoothly on Normal weather (seed 42, final mean reward +48.214) and generalized cleanly to unseen evaluation seeds across Normal (+54.63), Dry (+65.87), Heatwave (+61.40), and Wet (+16.68) regimes.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Policy inspection shows the PPO agent leverages phenological stage and ETc features to pre-irrigate ahead of the sensitive Mid-Season flowering band (65%–85%) while withholding irrigation whenever daily rainfall exceeds zero, achieving a superior water-compliance Pareto frontier.",
    "keyDrivers": [
      {
        "factor": "Crop Growth Stage & Kc Demand",
        "impact": "Triggers proactive pre-irrigation before flowering (Kc=1.15, 2.0x stress penalty) and tapers watering during ripening."
      },
      {
        "factor": "Daily Rainfall & Weather State",
        "impact": "Suppresses irrigation to 0.00 mm/day on Rainy days, avoiding the 0.05/mm rain-waste penalty and saving 98% water in Wet regimes."
      },
      {
        "factor": "Root-Zone Moisture Deficit Margin",
        "impact": "Maintains moisture near stage midpoints for parabolic target rewards while preventing drops below the 20% wilting point."
      }
    ],
    "limitations": [
      {
        "title": "Single-Bucket Homogeneous Soil Model",
        "description": "Models the root zone as a single 100 mm reservoir rather than multi-layer vertical infiltration via Richards equations."
      },
      {
        "title": "Idealized Sensor & Salinity Assumptions",
        "description": "Assumes noise-free daily soil moisture readings and omits irrigation salinity buildup or physical hardware actuation delays."
      },
      {
        "title": "Synthetic Weather Generator Calibration",
        "description": "Training and evaluation rely on Markov weather parameters rather than historical meteorological station time series."
      }
    ],
    "futureImprovements": [
      {
        "title": "Multi-Horizon Weather Forecast Inputs",
        "target": "Integrate 3-to-5-day precipitation and temperature forecasts into observations to reduce excess-water days."
      },
      {
        "title": "Multi-Layer Soil & Sensor Noise Modeling",
        "target": "Adopt multi-layer root-zone hydrology with realistic sensor drift and domain randomization."
      },
      {
        "title": "Edge Deployment for Drip Actuation",
        "target": "Export quantized ONNX/TFLite policy weights for hardware-in-the-loop ESP32 or Raspberry Pi valve control."
      }
    ],
    "discoveryNote": "All simulation parameters, unit tests (9 tests in tests/), and policy rollouts are reproducibly configured via config/config.yaml."
  },
  "mlModelArtifacts": [
    {
      "name": "ppo_irrigation_agent.zip",
      "format": "ZIP"
    },
    {
      "name": "training_history.csv",
      "format": "CSV"
    },
    {
      "name": "training_summary.json",
      "format": "JSON"
    },
    {
      "name": "evaluation_summary.csv",
      "format": "CSV"
    },
    {
      "name": "evaluation_episodes.csv",
      "format": "CSV"
    },
    {
      "name": "representative_trajectories.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Reinforcement Learning",
    "modelArchitecture": "PPO Actor-Critic MLP",
    "coreFramework": "Stable-Baselines3 PyTorch",
    "validationMethod": "Multi-Scenario Multi-Seed"
  }
}
```

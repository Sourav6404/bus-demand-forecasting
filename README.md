# BusDemand AI – Bus Passenger-Demand Forecasting

BusDemand AI is an enterprise-grade machine learning platform engineered to forecast intercity bus passenger demand. Built around behavioural synthetic mobility simulation across major South Indian urban corridors, the system simulates multi-year passenger demand dynamics under complex socioeconomic, seasonal, calendar, and price elasticity factors.

---

## 1. Project Overview

Intercity passenger bus operations require precise demand forecasting to optimize fleet scheduling, pricing strategies, and route frequency. BusDemand AI models passenger demand dynamics using high-resolution synthetic simulation grounded in gravity-model transportation theory and behavioral economics.

### Core Specifications
- **Language & Runtime:** Python 3.10+
- **Data Engine:** Pandas & NumPy
- **Simulation Horizon:** 3 Years (Daily granularity, 1,096 simulation days: `2024-01-01` to `2026-12-31`)
- **Departure Frequency:** 4 scheduled departures per route per day:
  - `DEP_01`: Early Morning (`06:00`)
  - `DEP_02`: Midday (`11:30`)
  - `DEP_03`: Evening (`17:00`)
  - `DEP_04`: Night Sleeper (`22:30`)
- **Initial Geography:** 8 major South Indian urban hubs across Karnataka, Tamil Nadu, and Kerala
  - **Bengaluru (BLR)** – Karnataka (Tier 1)
  - **Mysuru (MYS)** – Karnataka (Tier 2)
  - **Chennai (MAA)** – Tamil Nadu (Tier 1)
  - **Coimbatore (CBE)** – Tamil Nadu (Tier 2)
  - **Kochi (COK)** – Kerala (Tier 2)
  - **Kozhikode (CCJ)** – Kerala (Tier 2)
  - **Thrissur (TCR)** – Kerala (Tier 2)
  - **Thiruvananthapuram (TVM)** – Kerala (Tier 2)
- **Route Network:** 20 directed intercity bus routes connecting the 8 hubs across 10 strategic bidirectional corridors. All route distances, journey durations, commercial speeds, and route demand multipliers are synthetic scenario modeling assumptions calibrated for prototype simulation.

---

## 2. Project Objectives

1. **High-Fidelity Synthetic Simulation:** Develop an advanced behavioural demand simulator capturing macroeconomic indices, seasonal cycles, festival surges (Diwali, Pongal, Onam), weekday/weekend splits, and fare sensitivity.
2. **Feature Engineering Pipeline:** Extract spatio-temporal, lag, rolling statistical, weather, and calendar features tailored for corridor-level mobility.
3. **Forecasting Model Benchmark:** Evaluate and benchmark time-series and gradient boosting architectures (LightGBM, XGBoost, baseline models) against business KPIs (MAE, RMSE, WAPE).
4. **Operational API & Serving:** Expose production-ready REST endpoints for route-level and departure-level demand predictions and confidence intervals.

---

## 3. Architecture & Data Strategy

For complete details on the behavioral demand equations, spatial interaction gravity models, and socioeconomic index formulations, see:
- [Synthetic Data Strategy & Behavioral Assumptions](docs/synthetic_data_strategy.md)
- [Project Configuration Module](src/config.py)

```
BusDemand AI
├── Data Pipeline (Synthetic Simulation & Validation)
│   ├── City Profiles Generator (Socioeconomic, Tourism, Mobility Indices)
│   ├── Intercity Route Network (Distance, Travel Time, Tolls, Road Quality)
│   └── Behavioural Demand Simulator (3-year Horizon, 4 Departures/Day)
├── Feature Engineering Engine
│   ├── Temporal & Calendar Features (Diwali, Pongal, Onam, Long Weekends)
│   ├── Origin-Destination Spatial Embeddings
│   └── Rolling Demand Lags & Volatility
├── Predictive Modeling & Evaluation
│   ├── Baseline Time-Series Models
│   ├── Gradient Boosted Trees (LightGBM / XGBoost)
│   ├── Cross-Validation (Corridor & Time-split CV)
│   └── Interpretability (SHAP, Feature Attribution)
└── Serving & Deployment
    ├── FastAPI REST Endpoints
    └── Batch Inference & Reporting Engine
```

---

## 4. Setup Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone & Navigate to the Repository
```bash
git clone <repository_url>
cd "BusDemand AI – Bus Passenger-Demand Forecasting"
```

### 2. Set Up a Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Reproducibility & Configuration
All simulation components reference a centralized deterministic seed (`RANDOM_SEED = 42`). Verify your configuration setup with:
```bash
python -c "import src.config as cfg; print(f'Config OK: {cfg.SIMULATION_YEARS} years, {len(cfg.TARGET_CITIES)} cities, seed={cfg.DEFAULT_RANDOM_SEED}')"
```

### 5. Generate Reference & Simulation Datasets
Generate the validated reference datasets and synthetic passenger demand:
```bash
# 1. Generate 8 scenario-based city profiles
python src/data/generate_city_profiles.py

# 2. Generate 20 directed intercity route profiles
python src/data/generate_route_profiles.py

# 3. Generate 3-year calendar features & official holiday registry (1,096 days)
python src/data/generate_calendar_features.py

# 4. Generate 87,680 synthetic trip-level passenger demand records (Stage 6)
python src/data/generate_trip_demand.py

# 5. Generate EDA report and Jupyter notebook
python src/data/generate_eda_report.py

# 6. Run full automated test suite (59 tests)
pytest tests/ -v
```

---

## 5. Repository Structure

```
├── docs/               <- Strategy, assumptions, and architectural documentation
│   └── synthetic_data_strategy.md
├── data/
│   ├── raw/            <- Immutable raw datasets and base configuration profiles
│   │   ├── city_profiles.csv       <- 8 South Indian hub profiles & mobility indices
│   │   ├── route_profiles.csv      <- 20 directed intercity route features
│   │   ├── calendar_features.csv   <- 1,096-day calendar matrix (2024-2026)
│   │   └── holiday_events.csv      <- 165 official and cultural holiday events
│   ├── generated/      <- Intermediate simulation artifacts
│   └── processed/      <- Processed modeling matrices
│       └── trip_demand.csv         <- 87,680 scheduled trip passenger demand records
├── src/
│   ├── __init__.py
│   ├── config.py       <- Centralized project configuration & seed control
│   ├── data/           <- Data generation, ingestion, validation scripts
│   │   ├── generate_city_profiles.py
│   │   ├── generate_route_profiles.py
│   │   ├── generate_calendar_features.py
│   │   ├── generate_trip_demand.py
│   │   └── generate_eda_report.py
│   ├── features/       <- Feature extraction, encoders, lag transformers
│   ├── models/         <- Model definitions, training scripts, tuning routines
│   ├── evaluation/     <- Performance metrics, backtesting, residual diagnostics
│   └── api/            <- REST API endpoints, schemas, request/response models
├── notebooks/          <- Exploratory data analysis and experimental prototypes
│   └── 01_trip_demand_eda.ipynb    <- Interactive trip passenger demand EDA
├── artifacts/          <- Serialized models, encoders, and evaluation metadata
├── reports/            <- Generated performance reports, figures, and metrics
│   └── trip_demand_eda.md          <- Comprehensive Stage 6 EDA summary report
└── tests/              <- Unit and integration test suite (59 tests)
    ├── test_city_profiles.py
    ├── test_route_profiles.py
    ├── test_calendar_features.py
    └── test_trip_demand.py
```

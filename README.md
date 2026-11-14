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

---

## 5. Repository Structure

```
├── docs/               <- Strategy, assumptions, and architectural documentation
│   └── synthetic_data_strategy.md
├── data/
│   ├── raw/            <- Immutable raw datasets and base configuration profiles
│   ├── generated/      <- Synthetic simulation outputs (corridor demand, routes)
│   └── processed/      <- Cleaned, feature-engineered matrices ready for modeling
├── src/
│   ├── __init__.py
│   ├── config.py       <- Centralized project configuration & seed control
│   ├── data/           <- Data generation, ingestion, validation scripts
│   ├── features/       <- Feature extraction, encoders, lag transformers
│   ├── models/         <- Model definitions, training scripts, tuning routines
│   ├── evaluation/     <- Performance metrics, backtesting, residual diagnostics
│   └── api/            <- REST API endpoints, schemas, request/response models
├── notebooks/          <- Exploratory data analysis and experimental prototypes
├── artifacts/          <- Serialized models, encoders, and evaluation metadata
├── reports/            <- Generated performance reports, figures, and metrics
└── tests/              <- Unit and integration test suite
```

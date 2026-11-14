# BusDemand AI – Bus Passenger-Demand Forecasting

BusDemand AI is an enterprise-grade machine learning platform engineered to forecast intercity bus passenger demand. Built around behavioural synthetic mobility simulation across major South Indian urban corridors, the system simulates multi-year passenger demand dynamics under complex socioeconomic, seasonal, calendar, and price elasticity factors.

---

## 1. Project Overview

Intercity passenger bus operations require precise demand forecasting to optimize fleet scheduling, pricing strategies, and route frequency. BusDemand AI models passenger demand dynamics using high-resolution synthetic simulation grounded in gravity-model transportation theory and behavioral economics.

### Core Specifications
- **Language & Runtime:** Python 3.10+
- **Data Engine:** Pandas & NumPy
- **Simulation Horizon:** 3 Years (Daily granularity, 1,096 simulation days)
- **Departure Frequency:** 4 scheduled departures per route per day (Morning, Midday, Evening, Night)
- **Initial Geography:** 8 major South Indian urban hubs across Karnataka, Tamil Nadu, and Kerala
  - Bengaluru (KA)
  - Mysuru (KA)
  - Chennai (TN)
  - Coimbatore (TN)
  - Kochi (KL)
  - Kozhikode (KL)
  - Thrissur (KL)
  - Thiruvananthapuram (KL)

---

## 2. Project Objectives

1. **High-Fidelity Synthetic Simulation:** Develop an advanced behavioural demand simulator capturing macroeconomic indices, seasonal cycles, festival surges, weekday/weekend splits, and fare sensitivity.
2. **Feature Engineering Pipeline:** Extract spatio-temporal, lag, rolling statistical, weather, and calendar features tailored for corridor-level mobility.
3. **Forecasting Model Benchmark:** Evaluate and benchmark time-series and gradient boosting architectures (e.g., LightGBM/XGBoost, baseline ARIMA/Prophet) against business KPIs (MAE, RMSE, WAPE).
4. **Operational API & Serving:** Expose production-ready REST endpoints for route-level and departure-level demand predictions and confidence intervals.

---

## 3. Planned Architecture

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

## 4. Repository Structure

```
├── data/
│   ├── raw/            <- Immutable raw datasets and base configuration profiles
│   ├── generated/      <- Synthetic simulation outputs (corridor demand, routes)
│   └── processed/      <- Cleaned, feature-engineered matrices ready for modeling
├── src/
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

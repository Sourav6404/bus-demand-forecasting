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
3. **Forecasting Model Benchmark:** Develop and benchmark machine learning models — Historical Baseline, Random Forest, and HistGradientBoosting — using chronological splits against business KPIs (MAE, RMSE, WAPE). **Stage 7 complete:** HistGradientBoosting achieves **21.10% WAPE** on held-out test data, a **19.4% improvement** over the historical baseline.
4. **Operational API & Serving [STAGE 8 COMPLETE]:** Expose production-ready FastAPI REST endpoints and interactive web interface for route-level and departure-level demand predictions 24 hours prior to scheduled departure.

---

## 3. Architecture & Data Strategy

For complete details on the behavioral demand equations, spatial interaction gravity models, and socioeconomic index formulations, see:
- [Synthetic Data Strategy & Behavioral Assumptions](docs/synthetic_data_strategy.md)
- [Project Configuration Module](src/config.py)
- [Stage 7 Model Evaluation Report](reports/model_evaluation_report.md)

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
├── Predictive Modeling & Evaluation [STAGE 7 COMPLETE]
│   ├── Historical Route × Slot Baseline (hierarchical fallback)
│   ├── Random Forest Regressor (MAE: 8.058, WAPE: 21.53% val)
│   ├── HistGradientBoosting Regressor ★ WINNER (MAE: 7.331, WAPE: 21.10% test)
│   ├── Chronological Train/Val/Test Splits (58,480 / 14,480 / 14,720 trips)
│   └── Permutation Feature Importance & Per-Route Diagnostics
└── Serving & Deployment [STAGE 8 COMPLETE]
    ├── FastAPI REST Endpoints (/health, /routes, /departure-slots, /predict)
    ├── Feature Builder (25-feature matrix from raw reference datasets)
    ├── Predictor Service (loads best_model_pipeline.joblib once on boot)
    └── Interactive Web UI (HTML/CSS/JS dashboard served at /)
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
Generate the validated reference datasets, synthetic passenger demand, and model artifacts:
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

# 6. Train and evaluate all Stage 7 models (Baseline, RF, HistGradientBoosting)
#    Saves best_model_pipeline.joblib and evaluation_metrics.json to artifacts/
python src/models/train_models.py

# 7. Start the FastAPI inference service and interactive web UI (Stage 8)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# 8. Run full automated test suite (103 tests across all stages)
python -m pytest tests/ -v
```

---

## 5. REST Inference API & Interactive Interface (Stage 8)

The FastAPI inference service loads the trained `artifacts/best_model_pipeline.joblib` artifact **once at application startup** via a lifespan manager, reconstructs the 25 required model features dynamically from reference datasets, and predicts passenger volume 24 hours prior to departure.

### Starting the Server
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Web Interface:** `http://127.0.0.1:8000/`
- **Swagger Interactive API Documentation:** `http://127.0.0.1:8000/docs`
- **ReDoc API Documentation:** `http://127.0.0.1:8000/redoc`

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service and model status, scikit-learn version, and horizon bounds |
| `GET` | `/routes` | 20 supported directed corridors with distance, duration, and mobility scores |
| `GET` | `/departure-slots` | 4 daily scheduled departure slots (`06:00`, `11:30`, `17:00`, `22:30`) |
| `POST` | `/predict` | 24-hour ahead demand forecast with operational capacity status |
| `GET` | `/` | Responsive HTML5/CSS/JavaScript single-page forecasting dashboard |

### Example Prediction Request & Response

#### Request (`POST /predict`):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "route_id": "R_BLR_COK",
    "departure_date": "2026-12-20",
    "departure_time": "22:30"
  }'
```

#### Response (`HTTP 200 OK`):
```json
{
  "route_id": "R_BLR_COK",
  "origin_city": "Bengaluru",
  "destination_city": "Kochi",
  "departure_datetime": "2026-12-20T22:30:00",
  "prediction_cutoff": "2026-12-19T22:30:00",
  "predicted_passenger_demand": 49.0,
  "bus_capacity": 45,
  "capacity_status": "capacity_exceeded",
  "estimated_unserved_demand": 4.0,
  "model_name": "HistGradientBoostingRegressor"
}
```

#### Operational Logic:
- `prediction_cutoff`: Calculated as `departure_datetime - 24 hours`. Calendar and planned holiday features are assumed known at this cutoff.
- `capacity_status`: `"within_capacity"` if `predicted_passenger_demand <= bus_capacity`, else `"capacity_exceeded"`.
- `estimated_unserved_demand`: `max(0, predicted_passenger_demand - bus_capacity)`.

---

## 6. Repository Structure

```
├── docs/               <- Strategy, assumptions, and architectural documentation
│   ├── synthetic_data_strategy.md
│   └── future_forecasting_architecture.md <- Architecture: historical vs 2027 calendar separation
├── data/
│   ├── raw/            <- Immutable raw datasets and base configuration profiles
│   │   ├── city_profiles.csv       <- 8 South Indian hub profiles & mobility indices
│   │   ├── route_profiles.csv      <- 20 directed intercity route features
│   │   ├── calendar_features.csv   <- 1,096-day calendar matrix (2024-2026, immutable)
│   │   └── holiday_events.csv      <- 165 official and cultural holiday events
│   ├── future/         <- Future operational calendar partition
│   │   └── 2027_calendar_features.csv <- 365-day calendar matrix for 2027 inference
│   ├── generated/      <- Intermediate simulation artifacts
│   └── processed/      <- Processed modeling matrices (git-ignored)
│       └── trip_demand.csv         <- 87,680 scheduled trip passenger demand records (2024-2026)
├── src/
│   ├── __init__.py
│   ├── config.py       <- Centralized project configuration & seed control
│   ├── data/           <- Data generation, ingestion, validation scripts
│   │   ├── generate_city_profiles.py
│   │   ├── generate_route_profiles.py
│   │   ├── generate_calendar_features.py
│   │   ├── generate_trip_demand.py
│   │   └── generate_eda_report.py
│   ├── models/         <- Stage 7: Model training, preprocessing, evaluation
│   │   ├── baseline.py             <- HistoricalRouteSlotBaseline with fallback
│   │   ├── preprocessing.py        <- ColumnTransformer pipelines & feature specs
│   │   ├── evaluate_models.py      <- MAE, RMSE, WAPE metrics & diagnostics
│   │   └── train_models.py         <- End-to-end training & artifact serialization
│   ├── api/            <- Stage 8: REST inference API & interactive UI
│   │   ├── main.py                 <- FastAPI app, exception handlers, web UI
│   │   ├── schemas.py              <- Pydantic v2 request/response schemas
│   │   ├── predictor.py            <- Model lifecycle, predict, capacity logic
│   │   └── feature_builder.py      <- 25-feature row builder from reference data
│   ├── features/       <- Feature extraction, encoders, lag transformers
│   ├── evaluation/     <- Performance metrics, backtesting, residual diagnostics
├── notebooks/          <- Exploratory data analysis and experimental prototypes
│   ├── 01_trip_demand_eda.ipynb    <- Interactive trip passenger demand EDA
│   └── 02_model_training_and_evaluation.ipynb  <- Stage 7 model training walkthrough
├── artifacts/          <- Serialized models & evaluation metadata (joblib git-ignored)
│   └── evaluation_metrics.json     <- Model comparison metrics (tracked)
├── reports/            <- Generated performance reports, figures, and metrics
│   ├── trip_demand_eda.md          <- Comprehensive Stage 6 EDA summary report
│   └── model_evaluation_report.md  <- Stage 7 model training & evaluation report
└── tests/              <- Unit and integration test suite (103 tests)
    ├── test_city_profiles.py       <- Stage 3 tests (10 tests)
    ├── test_route_profiles.py      <- Stage 4 tests (17 tests)
    ├── test_calendar_features.py   <- Stage 5 tests (17 tests)
    ├── test_trip_demand.py         <- Stage 6 tests (15 tests)
    ├── test_model_pipeline.py      <- Stage 7 tests (14 tests)
    ├── test_api.py                 <- Stage 8 tests (22 tests)
    └── test_future_calendar.py     <- Future calendar & data separation tests (8 tests)
```

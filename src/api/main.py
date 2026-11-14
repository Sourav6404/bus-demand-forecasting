"""FastAPI REST Inference Service and Interactive Interface for BusDemand AI.

Exposes production endpoints for route discovery, schedule discovery, health checks,
and 24-hour ahead passenger demand forecasting using the trained HistGradientBoosting pipeline.
"""

from contextlib import asynccontextmanager
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional
import sklearn

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.feature_builder import (
    DateOutOfRangeError,
    FeatureBuilder,
    OperationalCutoffError,
    RouteNotFoundError,
    UnsupportedDepartureTimeError,
)
from src.api.predictor import (
    ModelArtifactNotFoundError,
    ModelLoadError,
    ModelPredictor,
)
from src.api.schemas import (
    DepartureSlotsResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RoutesResponse,
)
from src.config import ARTIFACTS_DIR

# Global shared instances (lifecycle-managed)
predictor: Optional[ModelPredictor] = None
feature_builder: Optional[FeatureBuilder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager: initialize services once at startup."""
    global predictor, feature_builder
    # 1. Initialize feature builder with reference datasets
    try:
        feature_builder = FeatureBuilder()
    except Exception as e:
        print(f"[FATAL] Failed to initialize FeatureBuilder: {e}")
        feature_builder = None

    # 2. Load trained machine learning pipeline once
    try:
        predictor = ModelPredictor(auto_load=True)
        print(f"[INFO] Loaded model pipeline: {predictor.model_name} from {predictor.artifact_path}")
    except (ModelArtifactNotFoundError, ModelLoadError) as e:
        print(f"[WARNING] Model artifact initialization failed: {e}")
        # Allow app to boot so health endpoint can report the degraded status
        predictor = ModelPredictor(auto_load=False)
        predictor._load_error = str(e)

    yield

    # Teardown logic
    predictor = None
    feature_builder = None


app = FastAPI(
    title="BusDemand AI – Intercity Bus Passenger Demand Forecasting API",
    description=(
        "Production inference service for predicting intercity bus passenger demand "
        "24 hours prior to scheduled departure across major South Indian corridors."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local web interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Custom Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RouteNotFoundError)
async def route_not_found_handler(request: Request, exc: RouteNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc), "error_type": "RouteNotFoundError"},
    )


@app.exception_handler(DateOutOfRangeError)
async def date_out_of_range_handler(request: Request, exc: DateOutOfRangeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "DateOutOfRangeError"},
    )


@app.exception_handler(UnsupportedDepartureTimeError)
async def unsupported_time_handler(request: Request, exc: UnsupportedDepartureTimeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "UnsupportedDepartureTimeError"},
    )


@app.exception_handler(OperationalCutoffError)
async def operational_cutoff_handler(request: Request, exc: OperationalCutoffError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "OperationalCutoffError"},
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and model health check",
    tags=["Monitoring"],
)
def get_health() -> HealthResponse:
    """Check API service status and verify that the ML model pipeline is loaded and ready."""
    is_ready = predictor is not None and predictor.is_loaded()
    model_info = predictor.get_model_info() if predictor else {}

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        service="BusDemand AI Inference API",
        version="1.0.0",
        model_loaded=is_ready,
        model_name=model_info.get("model_name"),
        model_artifact_path=model_info.get("artifact_path"),
        supported_horizon={
            "start_date": "2024-01-01",
            "historical_end_date": "2026-12-31",
            "operational_horizon_end_date": "2027-12-31",
            "total_calendar_days": "1461",
        },
        environment={
            "scikit_learn_version": sklearn.__version__,
            "python_version": os.sys.version.split()[0],
        },
    )


@app.get(
    "/routes",
    response_model=RoutesResponse,
    summary="Get supported route network",
    tags=["Metadata"],
)
def get_routes() -> RoutesResponse:
    """Retrieve all 20 supported directed intercity corridors with distance, duration, and mobility scores."""
    if feature_builder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reference data service is currently initializing or unavailable.",
        )
    routes = feature_builder.get_supported_routes()
    return RoutesResponse(total_routes=len(routes), routes=routes)


@app.get(
    "/departure-slots",
    response_model=DepartureSlotsResponse,
    summary="Get supported departure slots",
    tags=["Metadata"],
)
def get_departure_slots() -> DepartureSlotsResponse:
    """Retrieve the four standard daily scheduled departure slots, timing windows, and operational cutoff bounds."""
    if feature_builder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reference data service is currently initializing or unavailable.",
        )
    slots = feature_builder.get_departure_slots()
    timing_info = feature_builder.get_earliest_departure_info()
    return DepartureSlotsResponse(
        total_slots=len(slots),
        slots=slots,
        server_time=timing_info["server_time"],
        earliest_departure_datetime=timing_info["earliest_departure_datetime"],
        min_departure_date=timing_info["min_departure_date"],
        max_departure_date=timing_info["max_departure_date"],
        timezone=timing_info["timezone"],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Forecast trip passenger demand",
    tags=["Forecasting"],
)
def predict_passenger_demand(request: PredictionRequest) -> PredictionResponse:
    """Generate 24-hour ahead passenger demand forecast for a scheduled bus departure.

    Accepts corridor route_id, departure_date (YYYY-MM-DD), and departure_time (HH:MM).
    Strictly enforces operational 24-hour advance notice and upper calendar bounds (2026-12-31).
    Constructs the 25 model features from raw reference datasets, executes the
    trained HistGradientBoosting pipeline, and computes operational capacity indicators.
    """
    if predictor is None or not predictor.is_loaded():
        err = predictor._load_error if predictor else "Predictor not initialized."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Forecasting model is not loaded: {err}",
        )

    if feature_builder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature builder is not initialized.",
        )

    # 1. Construct model feature DataFrame and metadata (validates 24h cutoff & calendar bounds)
    features_df, meta = feature_builder.build_feature_dataframe(
        route_id=request.route_id,
        departure_date=request.departure_date,
        departure_time=request.departure_time,
        allow_historical=request.allow_historical,
    )

    # 2. Execute inference through model pipeline
    bus_capacity = meta["bus_capacity"]
    pred_demand, capacity_status, unserved = predictor.predict(
        features_df=features_df,
        bus_capacity=bus_capacity,
    )

    return PredictionResponse(
        route_id=meta["route_id"],
        origin_city=meta["origin_city"],
        destination_city=meta["destination_city"],
        departure_datetime=meta["departure_datetime"],
        prediction_cutoff=meta["prediction_cutoff"],
        predicted_passenger_demand=pred_demand,
        bus_capacity=bus_capacity,
        capacity_status=capacity_status,
        estimated_unserved_demand=unserved,
        model_name=predictor.model_name,
        calendar_source=meta.get("calendar_source", "historical"),
    )


@app.post(
    "/predict/historical",
    response_model=PredictionResponse,
    summary="Historical prediction testing endpoint",
    tags=["Testing"],
)
def predict_passenger_demand_historical(request: PredictionRequest) -> PredictionResponse:
    """Historical testing endpoint that evaluates historical trips within the simulation dataset (2024-2026) without enforcing the operational 24-hour advance notice cutoff."""
    request_copy = request.model_copy(update={"allow_historical": True})
    return predict_passenger_demand(request_copy)


# ---------------------------------------------------------------------------
# Interactive Web Interface (HTML/CSS/JS)
# ---------------------------------------------------------------------------

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BusDemand AI - Passenger Demand Forecasting</title>
    <style>
        :root {
            --bg-color: #f8f9fa;
            --surface-color: #ffffff;
            --text-primary: #1f2937;
            --text-secondary: #4b5563;
            --text-muted: #6b7280;
            --border-color: #d1d5db;
            --border-subtle: #e5e7eb;
            --accent-green: #2e7d32;
            --accent-green-hover: #256629;
            --warning-bg: #fffbeb;
            --warning-text: #b45309;
            --warning-border: #fde68a;
            --success-bg: #f0fdf4;
            --success-text: #166534;
            --success-border: #bbf7d0;
            --error-bg: #fef2f2;
            --error-text: #991b1b;
            --error-border: #fecaca;
            --font-stack: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: var(--font-stack);
            font-size: 14px;
            line-height: 1.5;
            padding: 24px 16px;
        }

        .container {
            max-width: 860px;
            margin: 0 auto;
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 24px 32px;
        }

        /* Header */
        header {
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 20px;
        }

        .logo {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }

        .subtitle {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Main Page Intro */
        h1.page-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        p.page-description {
            color: var(--text-secondary);
            font-size: 13px;
            margin-bottom: 20px;
        }

        /* Form */
        form {
            display: flex;
            flex-direction: column;
            gap: 14px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 24px;
        }

        .form-row {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        @media (max-width: 560px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .container {
                padding: 16px 18px;
            }
        }

        label {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        select, input[type="date"] {
            background-color: var(--surface-color);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 8px 10px;
            font-size: 14px;
            font-family: inherit;
            width: 100%;
        }

        select:focus, input[type="date"]:focus {
            outline: none;
            border-color: var(--accent-green);
            box-shadow: 0 0 0 1px var(--accent-green);
        }

        .btn-submit {
            background-color: var(--accent-green);
            color: #ffffff;
            border: 1px solid var(--accent-green-hover);
            border-radius: 4px;
            padding: 9px 20px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            align-self: flex-start;
            transition: background-color 0.1s ease;
        }

        .btn-submit:hover {
            background-color: var(--accent-green-hover);
        }

        .btn-submit:disabled {
            background-color: #9ca3af;
            border-color: #9ca3af;
            cursor: not-allowed;
        }

        /* Messages */
        .error-message {
            background-color: var(--error-bg);
            color: var(--error-text);
            border: 1px solid var(--error-border);
            border-radius: 4px;
            padding: 10px 12px;
            font-size: 13px;
            margin-bottom: 16px;
            display: none;
        }

        .initial-instruction {
            color: var(--text-muted);
            font-size: 13px;
            padding: 8px 0;
        }

        /* Results */
        .results-section h2 {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
        }

        .results-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .results-table tr {
            border-bottom: 1px solid var(--border-subtle);
        }

        .results-table tr:last-child {
            border-bottom: none;
        }

        .results-table th {
            text-align: left;
            padding: 9px 4px;
            color: var(--text-secondary);
            font-weight: 500;
            width: 36%;
        }

        .results-table td {
            padding: 9px 4px;
            color: var(--text-primary);
            font-weight: 500;
        }

        .field-note {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 400;
            margin-left: 4px;
        }

        .status-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 500;
        }

        .status-within {
            background-color: var(--success-bg);
            color: var(--success-text);
            border: 1px solid var(--success-border);
        }

        .status-exceeded {
            background-color: var(--warning-bg);
            color: var(--warning-text);
            border: 1px solid var(--warning-border);
        }

        /* Demand vs Capacity Chart */
        .capacity-chart-container {
            margin-top: 18px;
            padding: 14px 16px;
            background-color: #f9fafb;
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .chart-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .chart-pct {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .chart-bar-bg {
            height: 14px;
            background-color: #e5e7eb;
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }

        .chart-bar-fill {
            height: 100%;
            background-color: var(--accent-green);
            transition: width 0.3s ease;
        }

        .chart-bar-fill.exceeded {
            background-color: var(--warning-text);
        }

        .chart-axis {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 6px;
        }

        .sim-note {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 16px;
            font-style: italic;
        }

        .fractional-note {
            font-size: 12px;
            color: var(--text-secondary);
            margin: 12px 0 6px 0;
            padding: 8px 12px;
            background-color: #f3f4f6;
            border-left: 3px solid var(--accent-green);
            border-radius: 2px;
        }

        /* Footer */
        footer {
            margin-top: 24px;
            padding-top: 12px;
            border-top: 1px solid var(--border-subtle);
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }

        footer a {
            color: var(--text-secondary);
            text-decoration: underline;
        }

        footer a:hover {
            color: var(--text-primary);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">BusDemand AI</div>
            <div class="subtitle">Bus Passenger Demand Forecasting</div>
        </header>

        <main>
            <h1 class="page-title">Passenger Demand Forecast</h1>
            <p class="page-description">Estimate passenger demand for a scheduled bus route using historical demand patterns.</p>

            <div id="errorBanner" class="error-message"></div>

            <form id="predictionForm">
                <div class="form-row">
                    <label for="routeSelect">Route</label>
                    <select id="routeSelect" required>
                        <option value="">Loading routes...</option>
                    </select>
                </div>

                <div class="form-grid">
                    <div class="form-row">
                        <label for="dateInput">Departure Date</label>
                        <input type="date" id="dateInput" value="2027-01-15" min="2024-01-01" max="2027-12-31" required>
                    </div>

                    <div class="form-row">
                        <label for="slotSelect">Departure Time</label>
                        <select id="slotSelect" required>
                            <option value="">Loading departure slots...</option>
                        </select>
                    </div>
                </div>

                <div>
                    <button type="submit" id="predictBtn" class="btn-submit">Forecast</button>
                </div>
            </form>

            <section class="results-section">
                <h2>Forecast</h2>

                <div id="initialInstruction" class="initial-instruction">
                    Enter trip details to view the demand forecast.
                </div>

                <div id="resultsContent" style="display: none;">
                    <table class="results-table">
                        <tbody>
                            <tr>
                                <th>Estimated Passenger Demand</th>
                                <td><span id="resDemand">0.0</span> passengers <span class="field-note">(Expected passengers (model estimate))</span></td>
                            </tr>
                            <tr>
                                <th>Bus Capacity</th>
                                <td><span id="resCapacity">45</span> passengers</td>
                            </tr>
                            <tr>
                                <th>Capacity Status</th>
                                <td><span id="resStatus" class="status-pill status-within">Within capacity</span></td>
                            </tr>
                            <tr>
                                <th>Capacity Utilisation</th>
                                <td><span id="resUtilisation">0.0%</span></td>
                            </tr>
                            <tr>
                                <th>Estimated Unserved Demand</th>
                                <td><span id="resUnserved">0.0</span> passengers <span class="field-note">(predicted demand above available coach capacity)</span></td>
                            </tr>
                            <tr>
                                <th>Forecast Cutoff</th>
                                <td><span id="resCutoff">-</span></td>
                            </tr>
                            <tr>
                                <th>Scheduled Departure</th>
                                <td><span id="resDeparture">-</span></td>
                            </tr>
                            <tr>
                                <th>Corridor</th>
                                <td><span id="resCorridor">-</span></td>
                            </tr>
                            <tr>
                                <th>Calendar Source</th>
                                <td><span id="resCalSource">-</span></td>
                            </tr>
                        </tbody>
                    </table>

                    <p class="fractional-note">Fractional values represent estimated average demand, not a literal fraction of a passenger.</p>

                    <div class="capacity-chart-container">
                        <div class="chart-header">
                            <span class="chart-title">Demand vs. Capacity Comparison</span>
                            <span id="chartUtilisation" class="chart-pct">0% Utilisation</span>
                        </div>
                        <div class="chart-bar-bg">
                            <div id="chartBarFill" class="chart-bar-fill" style="width: 0%;"></div>
                        </div>
                        <div class="chart-axis">
                            <span>0 passengers</span>
                            <span>Coach Capacity: <span id="axisCapacity">45</span> seats</span>
                            <span>Predicted: <span id="axisDemand">0.0</span> pax</span>
                        </div>
                    </div>

                    <p class="sim-note">Note: Future 2027 forecasts are model inferences based on historical synthetic demand patterns and are not independently validated against actual 2027 passenger observations.</p>
                </div>
            </section>
        </main>

        <footer>
            <span>BusDemand AI Reference API</span>
            <span>
                <a href="/docs" target="_blank">API Documentation</a> &bull;
                <a href="/health" target="_blank">Health Status</a>
            </span>
        </footer>
    </div>

    <script>
        const errorBanner = document.getElementById("errorBanner");
        const predictionForm = document.getElementById("predictionForm");
        const predictBtn = document.getElementById("predictBtn");
        const routeSelect = document.getElementById("routeSelect");
        const dateInput = document.getElementById("dateInput");
        const slotSelect = document.getElementById("slotSelect");
        const initialInstruction = document.getElementById("initialInstruction");
        const resultsContent = document.getElementById("resultsContent");

        let timingConfig = {
            serverTime: null,
            earliestDeparture: null,
            minDate: "2026-11-16",
            maxDate: "2027-12-31",
            slots: []
        };

        function showError(msg) {
            errorBanner.textContent = msg;
            errorBanner.style.display = "block";
        }

        function clearError() {
            errorBanner.style.display = "none";
        }

        function updateDatePickerBounds() {
            if (!timingConfig.earliestDeparture) return;

            dateInput.max = timingConfig.maxDate;

            // Determine if the selected departure slot on minDate is >= earliestDeparture (+24h)
            const selectedSlotTime = slotSelect.value || "22:30";
            const minDateStr = timingConfig.minDate;

            const [sH, sM] = selectedSlotTime.split(":").map(Number);
            const candidateDt = new Date(minDateStr + "T00:00:00");
            candidateDt.setHours(sH, sM, 0, 0);

            let effectiveMinDate = minDateStr;
            if (candidateDt < timingConfig.earliestDeparture) {
                // For this specific slot, minDate is less than 24h ahead, so minDate advances by 1 day
                const nextDay = new Date(minDateStr + "T00:00:00");
                nextDay.setDate(nextDay.getDate() + 1);
                const y = nextDay.getFullYear();
                const m = String(nextDay.getMonth() + 1).padStart(2, "0");
                const d = String(nextDay.getDate()).padStart(2, "0");
                effectiveMinDate = `${y}-${m}-${d}`;
            }

            dateInput.min = effectiveMinDate;

            if (!dateInput.value || dateInput.value < effectiveMinDate) {
                dateInput.value = effectiveMinDate;
            } else if (dateInput.value > timingConfig.maxDate) {
                dateInput.value = timingConfig.maxDate;
            }
        }

        async function loadRoutes() {
            try {
                const res = await fetch("/routes");
                if (!res.ok) throw new Error("Failed to load routes catalog");
                const data = await res.json();
                routeSelect.innerHTML = "";
                data.routes.forEach(r => {
                    const opt = document.createElement("option");
                    opt.value = r.route_id;
                    opt.textContent = `${r.route_id} (${r.origin_city} - ${r.destination_city}, ${r.distance_km} km)`;
                    routeSelect.appendChild(opt);
                });
                routeSelect.value = "R_BLR_COK";
            } catch (err) {
                showError("Unable to load route catalog: " + err.message);
            }
        }

        async function loadSlots() {
            try {
                const res = await fetch("/departure-slots");
                if (!res.ok) throw new Error("Failed to load departure slots");
                const data = await res.json();

                timingConfig.serverTime = data.server_time ? new Date(data.server_time) : new Date();
                timingConfig.earliestDeparture = data.earliest_departure_datetime ? new Date(data.earliest_departure_datetime) : new Date(Date.now() + 24 * 3600 * 1000);
                timingConfig.minDate = data.min_departure_date || "2026-11-16";
                timingConfig.maxDate = data.max_departure_date || "2026-12-31";
                timingConfig.slots = data.slots || [];

                slotSelect.innerHTML = "";
                data.slots.forEach(s => {
                    const opt = document.createElement("option");
                    opt.value = s.time;
                    opt.textContent = `${s.time} (${s.name})`;
                    slotSelect.appendChild(opt);
                });
                slotSelect.value = "22:30";

                updateDatePickerBounds();
            } catch (err) {
                showError("Unable to load departure slots: " + err.message);
            }
        }

        slotSelect.addEventListener("change", updateDatePickerBounds);

        predictionForm.addEventListener("submit", async function(e) {
            e.preventDefault();
            clearError();

            const selectedRoute = routeSelect.value;
            const selectedDate = dateInput.value;
            const selectedTime = slotSelect.value;

            if (!selectedRoute) {
                showError("Please select a route.");
                return;
            }
            if (!selectedDate) {
                showError("Please select a departure date.");
                return;
            }
            if (!selectedTime) {
                showError("Please select a departure time.");
                return;
            }

            // Client-side validations
            if (selectedDate > timingConfig.maxDate) {
                showError(`Departure date (${selectedDate}) exceeds the maximum supported calendar horizon (${timingConfig.maxDate}).`);
                return;
            }
            if (selectedDate < "2024-01-01") {
                showError(`Departure date (${selectedDate}) is outside the supported calendar range (2024-01-01 to ${timingConfig.maxDate}).`);
                return;
            }

            if (timingConfig.earliestDeparture) {
                const [h, m] = selectedTime.split(":").map(Number);
                const depDt = new Date(selectedDate + "T00:00:00");
                depDt.setHours(h, m, 0, 0);

                if (timingConfig.serverTime && depDt < timingConfig.serverTime) {
                    showError(`The selected departure datetime (${selectedDate} ${selectedTime}) is in the past. Operational forecasts require a departure datetime at least 24 hours in the future.`);
                    return;
                }

                if (depDt < timingConfig.earliestDeparture) {
                    const earliestStr = timingConfig.earliestDeparture.toLocaleString();
                    showError(`The selected departure (${selectedDate} ${selectedTime}) is less than 24 hours ahead of the current time. Operational forecasting requires a minimum 24-hour advance booking notice (earliest allowable departure: ${earliestStr}).`);
                    return;
                }
            }

            predictBtn.disabled = true;
            predictBtn.textContent = "Calculating...";

            const payload = {
                route_id: selectedRoute,
                departure_date: selectedDate,
                departure_time: selectedTime
            };

            try {
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();

                if (!res.ok) {
                    showError(data.detail || "Prediction request could not be processed.");
                    return;
                }

                initialInstruction.style.display = "none";
                resultsContent.style.display = "block";

                const demand = data.predicted_passenger_demand;
                const capacity = data.bus_capacity;
                const utilisationPct = (demand / capacity) * 100;

                // 1. Populate table fields dynamically
                document.getElementById("resDemand").textContent = demand.toFixed(1);
                document.getElementById("resCapacity").textContent = capacity;
                document.getElementById("resUtilisation").textContent = utilisationPct.toFixed(1) + "%";
                document.getElementById("resUnserved").textContent = data.estimated_unserved_demand.toFixed(1);
                document.getElementById("resCutoff").textContent = data.prediction_cutoff.replace("T", " ");
                document.getElementById("resDeparture").textContent = data.departure_datetime.replace("T", " ");
                document.getElementById("resCorridor").textContent = `${data.origin_city} to ${data.destination_city} (${data.route_id})`;
                const calLabel = data.calendar_source === "future" ? "Future Forecast Calendar (2027)" : "Historical Reference Calendar (2024-2026)";
                document.getElementById("resCalSource").textContent = calLabel;

                // 2. Status Pill
                const statusEl = document.getElementById("resStatus");
                const barFill = document.getElementById("chartBarFill");
                const chartUtilisation = document.getElementById("chartUtilisation");
                const axisCapacity = document.getElementById("axisCapacity");
                const axisDemand = document.getElementById("axisDemand");

                chartUtilisation.textContent = utilisationPct.toFixed(1) + "% Capacity Utilisation";
                axisCapacity.textContent = capacity;
                axisDemand.textContent = demand.toFixed(1);

                // Bar fill scale: cap at 100% visually
                const fillWidth = Math.min(100, Math.round(utilisationPct));
                barFill.style.width = fillWidth + "%";

                if (data.capacity_status === "capacity_exceeded") {
                    statusEl.textContent = "Exceeds capacity";
                    statusEl.className = "status-pill status-exceeded";
                    barFill.className = "chart-bar-fill exceeded";
                } else {
                    statusEl.textContent = "Within capacity";
                    statusEl.className = "status-pill status-within";
                    barFill.className = "chart-bar-fill";
                }
            } catch (err) {
                showError("Network or server communication error: " + err.message);
            } finally {
                predictBtn.disabled = false;
                predictBtn.textContent = "Forecast";
            }
        });

        // Initialize form selects
        loadRoutes();
        loadSlots();
    </script>
</body>
</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Interactive web interface",
    tags=["Interface"],
)
def serve_interactive_frontend() -> HTMLResponse:
    """Serve the interactive web interface for trip selection and real-time demand forecasting."""
    return HTMLResponse(content=FRONTEND_HTML, status_code=200)

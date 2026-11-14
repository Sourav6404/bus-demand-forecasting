"""Unit and integration tests for BusDemand AI FastAPI inference service.

Verifies:
1. Health, routes, and departure slots endpoints.
2. Valid prediction inference with real loaded model.
3. 24-hour forecast cutoff timing.
4. Input validation (route existence, date bounds, slot times).
5. Operational capacity and unserved demand calculations.
6. Error handling for missing artifacts and malformed payloads.
7. Interactive web frontend delivery.
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.feature_builder import (
    FeatureBuilder,
    RouteNotFoundError,
    DateOutOfRangeError,
    OperationalCutoffError,
    UnsupportedDepartureTimeError,
)
from src.api.predictor import (
    ModelArtifactNotFoundError,
    ModelLoadError,
    ModelPredictor,
)
from src.config import (
    DEFAULT_BUS_CAPACITY,
    OPERATIONAL_HORIZON_END_DATE,
    PROCESSED_DATA_DIR,
    SIMULATION_END_DATE,
    SIMULATION_START_DATE,
    get_current_server_time,
    get_operating_timezone,
)
from src.models.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    split_chronological,
    split_dataframes,
)


@pytest.fixture(scope="module")
def client():
    """TestClient context manager that triggers app lifespan (loading model once)."""
    with TestClient(app) as test_client:
        yield test_client


class TestHealthAndMetadataEndpoints:
    """Verify health check and discovery endpoints."""

    def test_health_endpoint_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "HistGradientBoostingRegressor"
        assert "scikit_learn_version" in data["environment"]
        assert data["supported_horizon"]["total_calendar_days"] == "1461"
        assert data["supported_horizon"]["historical_end_date"] == "2026-12-31"
        assert data["supported_horizon"]["operational_horizon_end_date"] == "2027-12-31"

    def test_routes_endpoint(self, client):
        response = client.get("/routes")
        assert response.status_code == 200
        data = response.json()
        assert data["total_routes"] == 20
        assert len(data["routes"]) == 20

        # Verify key corridor properties
        blr_cok = next((r for r in data["routes"] if r["route_id"] == "R_BLR_COK"), None)
        assert blr_cok is not None
        assert blr_cok["origin_city"] == "Bengaluru"
        assert blr_cok["destination_city"] == "Kochi"
        assert blr_cok["distance_km"] == 540.0
        assert blr_cok["bus_capacity"] == 45

    def test_departure_slots_endpoint(self, client):
        response = client.get("/departure-slots")
        assert response.status_code == 200
        data = response.json()
        assert data["total_slots"] == 4
        slot_ids = [s["slot_id"] for s in data["slots"]]
        assert slot_ids == ["DEP_01", "DEP_02", "DEP_03", "DEP_04"]
        times = [s["time"] for s in data["slots"]]
        assert "06:00" in times
        assert "22:30" in times


class TestPredictionEndpoint:
    """Verify live model inference and operational calculations."""

    def test_valid_prediction_request(self, client):
        payload = {
            "route_id": "R_BLR_COK",
            "departure_date": "2026-12-20",
            "departure_time": "22:30",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Schema checks
        assert data["route_id"] == "R_BLR_COK"
        assert data["origin_city"] == "Bengaluru"
        assert data["destination_city"] == "Kochi"
        assert data["departure_datetime"] == "2026-12-20T22:30:00"
        assert data["prediction_cutoff"] == "2026-12-19T22:30:00"
        assert data["bus_capacity"] == 45
        assert data["model_name"] == "HistGradientBoostingRegressor"

        # Demand output checks
        assert isinstance(data["predicted_passenger_demand"], float)
        assert data["predicted_passenger_demand"] > 0.0
        assert data["capacity_status"] in ["within_capacity", "capacity_exceeded"]

    def test_exact_24_hour_cutoff(self, client):
        payload = {
            "route_id": "R_MAA_BLR",
            "departure_date": "2026-12-20",
            "departure_time": "06:00",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        dep_dt = datetime.fromisoformat(data["departure_datetime"])
        cutoff_dt = datetime.fromisoformat(data["prediction_cutoff"])
        delta = dep_dt - cutoff_dt
        assert delta.total_seconds() == 24 * 3600

    def test_capacity_status_and_unserved_demand_logic(self, client):
        # Predict on peak weekend night sleeper to test capacity exceedance
        payload = {
            "route_id": "R_BLR_COK",
            "departure_date": "2026-12-20",
            "departure_time": "22:30",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        demand = data["predicted_passenger_demand"]
        capacity = data["bus_capacity"]
        unserved = data["estimated_unserved_demand"]

        if demand > capacity:
            assert data["capacity_status"] == "capacity_exceeded"
            assert unserved == pytest.approx(round(demand - capacity, 1))
        else:
            assert data["capacity_status"] == "within_capacity"
            assert unserved == 0.0


class TestPredictionValidationAndErrors:
    """Verify input validation, boundary checking, and informative error messages."""

    def test_invalid_route_returns_404(self, client):
        payload = {
            "route_id": "R_UNKNOWN_CORRIDOR",
            "departure_date": "2026-12-20",
            "departure_time": "22:30",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_date_out_of_range_returns_400(self, client):
        # Date before simulation start
        payload = {
            "route_id": "R_BLR_COK",
            "departure_date": "2023-12-31",
            "departure_time": "22:30",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
        assert "outside the supported calendar range" in response.json()["detail"]

        # Date after simulation end / operational horizon
        payload["departure_date"] = "2028-01-01"
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
        assert "outside the supported calendar range" in response.json()["detail"]

    def test_unsupported_departure_time_returns_400(self, client):
        payload = {
            "route_id": "R_BLR_COK",
            "departure_date": "2026-12-20",
            "departure_time": "09:15",  # Not one of 06:00, 11:30, 17:00, 22:30
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
        assert "not a scheduled slot" in response.json()["detail"]

    def test_malformed_date_returns_422(self, client):
        payload = {
            "route_id": "R_BLR_COK",
            "departure_date": "not-a-date",
            "departure_time": "22:30",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestFeatureBuilderAndModelLifecycle:
    """Verify underlying components, schemas, and failure modes."""

    def test_feature_builder_output_schema_and_types(self):
        builder = FeatureBuilder()
        df, meta = builder.build_feature_dataframe(
            route_id="R_BLR_MAA",
            departure_date="2026-12-20",
            departure_time="17:00",
        )
        assert list(df.columns) == MODEL_FEATURE_COLUMNS
        assert len(df) == 1
        assert not df.isnull().any().any()

        # Confirm non-leakage
        for col in ["passenger_demand", "capacity_exceeded", "estimated_unserved_demand", "trip_id"]:
            assert col not in df.columns

    def test_missing_model_artifact_raises_error(self):
        bogus_path = Path("artifacts/non_existent_model.joblib")
        predictor = ModelPredictor(artifact_path=bogus_path, auto_load=False)
        assert not predictor.is_loaded()
        with pytest.raises(ModelArtifactNotFoundError):
            predictor.load()

    def test_frontend_page_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "BusDemand AI" in response.text
        # Verify fractional explanation and updated label presence in HTML
        assert "Expected passengers (model estimate)" in response.text
        assert "Fractional values represent estimated average demand" in response.text
        assert "updateDatePickerBounds" in response.text


class TestOperationalDateValidationAndPredictionDisplay:
    """Detailed verification of operational date validation, timezone handling, and fractional display."""

    def test_1_past_departure_date_rejected(self, client):
        """Reject departures in the past with HTTP 400 and OperationalCutoffError."""
        payload = {
            "route_id": "R_BLR_COK",
            "departure_date": "2024-10-31",  # Past date
            "departure_time": "22:30",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert data["error_type"] == "OperationalCutoffError"
        assert "in the past" in data["detail"].lower()

    def test_2_departure_less_than_24_hours_ahead_rejected(self, client):
        """Reject departures scheduled less than 24 hours from server reference time."""
        ref_time = datetime(2026, 12, 1, 10, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        with patch("src.api.feature_builder.get_current_server_time", return_value=ref_time):
            # Departure at 22:30 on same day is 12.5 hours ahead (< 24h)
            payload = {
                "route_id": "R_BLR_COK",
                "departure_date": "2026-12-01",
                "departure_time": "22:30",
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 400
            data = response.json()
            assert data["error_type"] == "OperationalCutoffError"
            assert "less than 24 hours ahead" in data["detail"].lower()

    def test_3_departure_exactly_24_hours_ahead_accepted(self, client):
        """Accept departures scheduled exactly 24 hours ahead of server reference time."""
        ref_time = datetime(2026, 12, 1, 6, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        with patch("src.api.feature_builder.get_current_server_time", return_value=ref_time):
            # Departure at 06:00 next day is exactly 24 hours ahead
            payload = {
                "route_id": "R_BLR_COK",
                "departure_date": "2026-12-02",
                "departure_time": "06:00",
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["route_id"] == "R_BLR_COK"
            assert data["departure_datetime"] == "2026-12-02T06:00:00"

    def test_4_departure_more_than_24_hours_ahead_accepted(self, client):
        """Accept departures scheduled more than 24 hours ahead."""
        ref_time = datetime(2026, 12, 1, 6, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        with patch("src.api.feature_builder.get_current_server_time", return_value=ref_time):
            # Departure 48 hours ahead
            payload = {
                "route_id": "R_BLR_COK",
                "departure_date": "2026-12-03",
                "departure_time": "06:00",
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            assert response.json()["predicted_passenger_demand"] > 0.0

    def test_5_departure_beyond_calendar_upper_bound_rejected(self, client):
        """Reject departures beyond 2027-12-31 with informative calendar range message."""
        payload = {
            "route_id": "R_BLR_COK",
            "departure_date": "2028-01-01",
            "departure_time": "06:00",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert data["error_type"] == "DateOutOfRangeError"
        assert "exceeds the operational horizon" in data["detail"]
        assert OPERATIONAL_HORIZON_END_DATE in data["detail"]

    def test_6_date_picker_minimum_date_calculation(self, client):
        """Verify dynamic earliest departure date calculation logic."""
        builder = FeatureBuilder()

        # Morning reference: 02:00 -> earliest valid dt is 02:00 next day (all slots valid)
        ref_morning = datetime(2026, 12, 1, 2, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        info_morning = builder.get_earliest_departure_info(reference_time=ref_morning)
        assert info_morning["min_departure_date"] == "2026-12-02"
        assert info_morning["max_departure_date"] == "2027-12-31"

        # Late night reference: 23:00 -> earliest valid dt is 23:00 next day (past 22:30 latest slot)
        # So the minimum date advances to 2026-12-03
        ref_late = datetime(2026, 12, 1, 23, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        info_late = builder.get_earliest_departure_info(reference_time=ref_late)
        assert info_late["min_departure_date"] == "2026-12-03"

        # Verify departure-slots endpoint exposes this metadata
        response = client.get("/departure-slots")
        assert response.status_code == 200
        body = response.json()
        assert "min_departure_date" in body
        assert "max_departure_date" in body
        assert "earliest_departure_datetime" in body
        assert "timezone" in body
        assert body["max_departure_date"] == "2027-12-31"

    def test_7_fractional_prediction_capacity_calculations(self):
        """Verify decimal demand is kept at 1 decimal place and not rounded before capacity checks."""
        builder = FeatureBuilder()
        df, _ = builder.build_feature_dataframe(
            route_id="R_BLR_COK",
            departure_date="2026-12-20",
            departure_time="22:30",
            allow_historical=True,
        )

        predictor = ModelPredictor(auto_load=False)
        predictor.model_pipeline = MagicMock()

        # Case 1: 45.4 with capacity 45 -> Exceeds capacity, unserved 0.4
        predictor.model_pipeline.predict.return_value = np.array([45.4])
        demand, status, unserved = predictor.predict(df, bus_capacity=45)
        assert demand == 45.4
        assert status == "capacity_exceeded"
        assert unserved == 0.4

        # Case 2: 41.6 with capacity 45 -> Within capacity, unserved 0.0
        predictor.model_pipeline.predict.return_value = np.array([41.6])
        demand, status, unserved = predictor.predict(df, bus_capacity=45)
        assert demand == 41.6
        assert status == "within_capacity"
        assert unserved == 0.0

        # Case 3: 45.0 with capacity 45 -> Within capacity, unserved 0.0
        predictor.model_pipeline.predict.return_value = np.array([45.0])
        demand, status, unserved = predictor.predict(df, bus_capacity=45)
        assert demand == 45.0
        assert status == "within_capacity"
        assert unserved == 0.0

        # Case 4: 45.1 must not round down to 45 (which would erroneously mark it within capacity)
        predictor.model_pipeline.predict.return_value = np.array([45.1])
        demand, status, unserved = predictor.predict(df, bus_capacity=45)
        assert demand == 45.1
        assert status == "capacity_exceeded"
        assert unserved == 0.1

    def test_8_historical_training_and_evaluation_splits_unchanged(self):
        """Confirm that historical train, validation, and test split boundaries remain intact."""
        trip_path = PROCESSED_DATA_DIR / "trip_demand.csv"
        if not trip_path.exists():
            pytest.skip("trip_demand.csv not found")

        df = pd.read_csv(trip_path)
        train_df, val_df, test_df = split_dataframes(df)

        # Exact row counts from Stage 7
        assert len(train_df) == 58480
        assert len(val_df) == 14480
        assert len(test_df) == 14720
        assert len(train_df) + len(val_df) + len(test_df) == 87680

        # Exact boundary dates
        assert train_df["departure_date"].min() == "2024-01-01"
        assert train_df["departure_date"].max() == "2025-12-31"
        assert val_df["departure_date"].min() == "2026-01-01"
        assert val_df["departure_date"].max() == "2026-06-30"
        assert test_df["departure_date"].min() == "2026-07-01"
        assert test_df["departure_date"].max() == "2026-12-31"

    def test_historical_prediction_bypass_and_endpoint(self, client):
        """Verify dedicated mechanisms for evaluating historical departures."""
        # 1. Historical endpoint permits historical dates
        payload = {
            "route_id": "R_BLR_COK",
            "departure_date": "2024-10-31",
            "departure_time": "22:30",
        }
        res_hist = client.post("/predict/historical", json=payload)
        assert res_hist.status_code == 200
        assert res_hist.json()["predicted_passenger_demand"] > 0.0

        # 2. allow_historical=True flag permits historical dates on /predict
        payload["allow_historical"] = True
        res_flag = client.post("/predict", json=payload)
        assert res_flag.status_code == 200
        assert res_flag.json()["predicted_passenger_demand"] > 0.0

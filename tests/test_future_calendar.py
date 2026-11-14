"""Tests for 2027 future calendar features, data separation, and inference."""

import datetime
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.feature_builder import FeatureBuilder
from src.api.main import app
from src.config import (
    FUTURE_CALENDAR_PATH,
    OPERATIONAL_HORIZON_END_DATE,
    RAW_DATA_DIR,
    SIMULATION_END_DATE,
    SIMULATION_START_DATE,
    get_operating_timezone,
)
from src.data.generate_calendar_features import CALENDAR_FEATURE_COLUMNS
from src.models.preprocessing import MODEL_FEATURE_COLUMNS

CALENDAR_FEATURES_PATH = RAW_DATA_DIR / "calendar_features.csv"


@pytest.fixture
def feature_builder():
    return FeatureBuilder()


@pytest.fixture(scope="module")
def client():
    """TestClient context manager that triggers app lifespan (loading model once)."""
    with TestClient(app) as test_client:
        yield test_client


def test_1_historical_calendar_remains_unchanged():
    """Verify that data/raw/calendar_features.csv is strictly 2024-01-01 to 2026-12-31 (1096 days)."""
    assert CALENDAR_FEATURES_PATH.exists(), f"Missing {CALENDAR_FEATURES_PATH}"
    df_hist = pd.read_csv(CALENDAR_FEATURES_PATH)

    assert len(df_hist) == 1096, f"Expected 1,096 historical days, got {len(df_hist)}"
    assert df_hist["date"].min() == "2024-01-01"
    assert df_hist["date"].max() == "2026-12-31"
    assert (
        df_hist["date"].nunique() == 1096
    ), "Historical calendar contains duplicates"
    assert (
        list(df_hist.columns) == CALENDAR_FEATURE_COLUMNS
    ), "Historical columns altered"


def test_2_future_calendar_has_exact_365_days_for_2027():
    """Verify data/future/2027_calendar_features.csv has exactly 365 daily rows for 2027."""
    assert FUTURE_CALENDAR_PATH.exists(), f"Missing {FUTURE_CALENDAR_PATH}"
    df_fut = pd.read_csv(FUTURE_CALENDAR_PATH)

    assert len(df_fut) == 365, f"Expected 365 days for 2027, got {len(df_fut)}"
    assert df_fut["date"].min() == "2027-01-01"
    assert df_fut["date"].max() == "2027-12-31"
    assert (
        list(df_fut.columns) == CALENDAR_FEATURE_COLUMNS
    ), "Future calendar columns mismatch"


def test_3_future_calendar_continuity_and_no_missing():
    """Verify future calendar has no missing dates, no duplicates, and strict 1-day step continuity."""
    df_fut = pd.read_csv(FUTURE_CALENDAR_PATH)
    dt_series = pd.to_datetime(df_fut["date"])

    # Duplicates check
    assert (
        not dt_series.duplicated().any()
    ), "Future calendar contains duplicate dates"

    # Strict monotonicity & 1-day step
    diffs = dt_series.diff().dropna()
    assert (
        diffs == pd.Timedelta(days=1)
    ).all(), "Dates are not continuous 1-day steps"

    # Missing value check across all columns
    assert (
        df_fut.isna().sum().sum() == 0
    ), f"Missing values detected: {df_fut.isna().sum()}"


def test_4_future_calendar_features_validity():
    """Verify day of week, quarters, weekends, and holidays are valid for 2027."""
    df_fut = pd.read_csv(FUTURE_CALENDAR_PATH)

    # 2027-01-01 is a Friday (day_of_week == 4)
    jan1 = df_fut[df_fut["date"] == "2027-01-01"].iloc[0]
    assert jan1["day_of_week"] == 4
    assert jan1["is_weekend"] == 0
    assert jan1["month"] == 1
    assert jan1["quarter"] == 1
    assert jan1["season"] == "Winter"

    # Check statutory holiday (Republic Day: 2027-01-26, Tuesday)
    jan26 = df_fut[df_fut["date"] == "2027-01-26"].iloc[0]
    assert jan26["is_national_holiday"] == 1

    # Check Independence Day: 2027-08-15
    aug15 = df_fut[df_fut["date"] == "2027-08-15"].iloc[0]
    assert aug15["is_national_holiday"] == 1

    # Check Gandhi Jayanti: 2027-10-02
    oct02 = df_fut[df_fut["date"] == "2027-10-02"].iloc[0]
    assert oct02["is_national_holiday"] == 1


def test_5_calendar_source_routing(feature_builder):
    """Verify dates route correctly to 'historical' vs 'future'."""
    assert (
        feature_builder.get_calendar_source("2024-05-10") == "historical"
    )
    assert (
        feature_builder.get_calendar_source("2026-12-31") == "historical"
    )
    assert (
        feature_builder.get_calendar_source("2027-01-01") == "future"
    )
    assert (
        feature_builder.get_calendar_source("2027-12-31") == "future"
    )
    assert (
        feature_builder.get_calendar_source("2023-12-31") == "missing"
    )
    assert (
        feature_builder.get_calendar_source("2028-01-01") == "missing"
    )


def test_6_feature_builder_exact_25_features_for_2027(feature_builder):
    """Verify that feature builder produces the exact 25 model features in identical order for 2027."""
    features_df, meta = feature_builder.build_feature_dataframe(
        route_id="R_BLR_COK",
        departure_date="2027-03-15",
        departure_time="22:30",
        allow_historical=True,
    )

    assert meta["calendar_source"] == "future"
    assert list(features_df.columns) == MODEL_FEATURE_COLUMNS
    assert len(features_df) == 1
    assert features_df.isna().sum().sum() == 0

    # Ensure numeric columns are finite
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    assert np.isfinite(features_df[numeric_cols].values).all()


def test_7_real_model_inference_for_2027(client):
    """Verify end-to-end model inference via the API for a 2027 departure."""
    payload = {
        "route_id": "R_BLR_COK",
        "departure_date": "2027-02-20",
        "departure_time": "22:30",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["route_id"] == "R_BLR_COK"
    assert data["departure_datetime"].startswith("2027-02-20")
    assert data["calendar_source"] == "future"
    assert data["predicted_passenger_demand"] >= 0.0
    assert data["bus_capacity"] == 45
    assert data["capacity_status"] in ["within_capacity", "capacity_exceeded"]
    assert "model_name" in data


def test_8_advance_24h_cutoff_enforced_for_future(client):
    """Verify that the 24-hour cutoff rule is strictly evaluated against current server time."""
    # When simulating server time as 2027-06-15 10:00:
    tz = get_operating_timezone()
    mock_now = datetime.datetime(2027, 6, 15, 10, 0, 0, tzinfo=tz)

    with patch("src.api.feature_builder.get_current_server_time", return_value=mock_now):
        # 1. Past departure relative to mock_now -> Rejected
        res_past = client.post(
            "/predict",
            json={
                "route_id": "R_BLR_COK",
                "departure_date": "2027-06-14",
                "departure_time": "22:30",
            },
        )
        assert res_past.status_code == 400
        assert "past" in res_past.json()["detail"].lower()

        # 2. Less than 24 hours ahead -> Rejected (same day 22:30 is 12.5 hrs ahead)
        res_short = client.post(
            "/predict",
            json={
                "route_id": "R_BLR_COK",
                "departure_date": "2027-06-15",
                "departure_time": "22:30",
            },
        )
        assert res_short.status_code == 400
        assert "less than 24 hours" in res_short.json()["detail"].lower()

        # 3. >= 24 hours ahead -> Accepted (2027-06-16 22:30 is > 36 hrs ahead)
        res_valid = client.post(
            "/predict",
            json={
                "route_id": "R_BLR_COK",
                "departure_date": "2027-06-16",
                "departure_time": "22:30",
            },
        )
        assert res_valid.status_code == 200
        assert res_valid.json()["calendar_source"] == "future"

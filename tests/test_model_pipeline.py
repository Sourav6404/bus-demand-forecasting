"""Unit tests for Stage 7 baseline and machine learning modeling pipeline.

Tests verify:
1. Chronological dataset splitting boundaries and zero date leakage.
2. Prediction feature integrity (exclusion of targets and leakage fields).
3. HistoricalRouteSlotBaseline behavior and multi-tier fallback mechanism.
4. Preprocessing pipelines (RF OneHot and HGB Ordinal encodings).
5. Evaluation metrics (MAE, RMSE, zero-safe WAPE).
6. Serialized model artifact validity and inference capability.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR
from src.models.preprocessing import (
    split_chronological,
    split_dataframes,
    build_rf_preprocessor,
    build_hgb_preprocessor,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    PREDICTION_FEATURES,
    TARGET_COLUMN,
    METADATA_COLUMNS,
    LEAKAGE_COLUMNS,
)
from src.models.baseline import HistoricalRouteSlotBaseline
from src.models.evaluate_models import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    compute_metrics,
)


@pytest.fixture(scope="module")
def trip_df():
    """Load the synthetic trip demand dataset."""
    path = PROCESSED_DATA_DIR / "trip_demand.csv"
    if not path.exists():
        pytest.skip(f"Processed trip demand dataset not found at {path}")
    df = pd.read_csv(path, parse_dates=["departure_date"])
    return df


class TestChronologicalSplitting:
    """Verify chronological split sizes and partition integrity."""

    def test_split_sizes(self, trip_df):
        train_df, val_df, test_df = split_dataframes(trip_df)
        assert len(train_df) == 58480, f"Expected 58,480 train rows, got {len(train_df)}"
        assert len(val_df) == 14480, f"Expected 14,480 val rows, got {len(val_df)}"
        assert len(test_df) == 14720, f"Expected 14,720 test rows, got {len(test_df)}"
        assert len(train_df) + len(val_df) + len(test_df) == len(trip_df)

    def test_strict_chronological_ordering_and_zero_overlap(self, trip_df):
        train_df, val_df, test_df = split_dataframes(trip_df)

        train_max = train_df["departure_date"].max()
        val_min = val_df["departure_date"].min()
        val_max = val_df["departure_date"].max()
        test_min = test_df["departure_date"].min()
        test_max = test_df["departure_date"].max()

        assert train_max < val_min, f"Train date overlaps with val: {train_max} vs {val_min}"
        assert val_max < test_min, f"Val date overlaps with test: {val_max} vs {test_min}"

        # Verify exact target boundaries
        assert train_max == pd.Timestamp("2025-12-31")
        assert val_min == pd.Timestamp("2026-01-01")
        assert val_max == pd.Timestamp("2026-06-30")
        assert test_min == pd.Timestamp("2026-07-01")
        assert test_max == pd.Timestamp("2026-12-31")

    def test_split_chronological_returns_matrices(self, trip_df):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_chronological(trip_df)
        assert X_train.shape == (58480, 25)
        assert X_val.shape == (14480, 25)
        assert X_test.shape == (14720, 25)
        assert len(y_train) == 58480
        assert len(y_val) == 14480
        assert len(y_test) == 14720


class TestFeatureExclusionAndSchemas:
    """Ensure no data leakage fields or identifiers enter prediction features."""

    def test_target_and_leakage_excluded_from_features(self):
        for col in [TARGET_COLUMN] + METADATA_COLUMNS + LEAKAGE_COLUMNS:
            assert col not in PREDICTION_FEATURES, f"Column '{col}' must not be in PREDICTION_FEATURES"

    def test_prediction_features_disjoint(self):
        overlap = set(CATEGORICAL_FEATURES).intersection(set(NUMERICAL_FEATURES))
        assert len(overlap) == 0, f"Categorical and Numerical features must be disjoint, found: {overlap}"
        assert set(PREDICTION_FEATURES) == set(CATEGORICAL_FEATURES).union(set(NUMERICAL_FEATURES))
        assert len(PREDICTION_FEATURES) == 25


class TestHistoricalBaseline:
    """Verify historical baseline model behavior and fallback logic."""

    def test_baseline_fit_and_predict(self, trip_df):
        (X_train, y_train), (X_val, y_val), _ = split_chronological(trip_df)

        model = HistoricalRouteSlotBaseline()
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        assert (preds >= 0).all()

    def test_baseline_fallback_hierarchy(self):
        # Create a toy dataframe for testing fallback hierarchy
        toy_train = pd.DataFrame({
            "route_id": ["R1", "R1", "R2"],
            "departure_slot": ["DEP_01", "DEP_02", "DEP_01"],
            "passenger_demand": [10.0, 20.0, 30.0],
        })
        model = HistoricalRouteSlotBaseline()
        model.fit(toy_train[["route_id", "departure_slot"]], toy_train["passenger_demand"])

        # Global mean is (10 + 20 + 30) / 3 = 20.0
        # R1 mean is 15.0, R2 mean is 30.0
        # (R1, DEP_01) = 10.0, (R1, DEP_02) = 20.0, (R2, DEP_01) = 30.0
        assert model.global_mean_ == pytest.approx(20.0)

        toy_test = pd.DataFrame({
            "route_id": [
                "R1",       # Case 1: Exact match (R1, DEP_01) -> 10.0
                "R1",       # Case 2: Unseen slot for R1 (R1, DEP_03) -> route fallback = 15.0
                "R3",       # Case 3: Completely unseen route (R3, DEP_01) -> global fallback = 20.0
            ],
            "departure_slot": ["DEP_01", "DEP_03", "DEP_01"],
        })
        preds = model.predict(toy_test)
        assert preds[0] == pytest.approx(10.0)
        assert preds[1] == pytest.approx(15.0)
        assert preds[2] == pytest.approx(20.0)


class TestMetrics:
    """Verify evaluation metric calculations and edge case safety."""

    def test_metric_calculations_toy(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])

        mae = calculate_mae(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)
        wape = calculate_wape(y_true, y_pred)

        assert mae == pytest.approx(7.0 / 3.0)
        assert rmse == pytest.approx(np.sqrt(17.0 / 3.0))
        assert wape == pytest.approx((7.0 / 60.0) * 100.0)

    def test_wape_zero_division_safety(self):
        y_true = np.zeros(5)
        y_pred = np.zeros(5)
        assert calculate_wape(y_true, y_pred) == 0.0

        y_pred_nonzero = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calculate_wape(y_true, y_pred_nonzero) == 0.0

    def test_compute_metrics_dict(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([10.0, 20.0])
        res = compute_metrics(y_true, y_pred, model_name="PerfectModel", split_name="Test")
        assert res["mae"] == 0.0
        assert res["rmse"] == 0.0
        assert res["wape_pct"] == 0.0
        assert res["model"] == "PerfectModel"
        assert res["split"] == "Test"


class TestSerializedArtifacts:
    """Verify saved models, metadata, and reproducibility."""

    def test_artifacts_exist(self):
        assert (ARTIFACTS_DIR / "historical_baseline.joblib").exists()
        assert (ARTIFACTS_DIR / "random_forest_pipeline.joblib").exists()
        assert (ARTIFACTS_DIR / "hist_gradient_boosting_pipeline.joblib").exists()
        assert (ARTIFACTS_DIR / "best_model_pipeline.joblib").exists()
        assert (ARTIFACTS_DIR / "evaluation_metrics.json").exists()

    def test_best_model_can_load_and_predict(self, trip_df):
        best_model = joblib.load(ARTIFACTS_DIR / "best_model_pipeline.joblib")
        sample_X = trip_df[PREDICTION_FEATURES].head(20)
        preds = best_model.predict(sample_X)
        assert len(preds) == 20
        assert not np.isnan(preds).any()
        assert (preds >= 0).all()

    def test_baseline_can_load_and_predict(self, trip_df):
        baseline = joblib.load(ARTIFACTS_DIR / "historical_baseline.joblib")
        sample_X = trip_df[PREDICTION_FEATURES].head(20)
        preds = baseline.predict(sample_X)
        assert len(preds) == 20
        assert not np.isnan(preds).any()

    def test_metrics_json_content(self):
        with open(ARTIFACTS_DIR / "evaluation_metrics.json", "r") as f:
            metrics_data = json.load(f)

        assert metrics_data["best_model"] == "HistGradientBoosting"
        assert metrics_data["train_rows"] == 58480
        assert metrics_data["val_rows"] == 14480
        assert metrics_data["test_rows"] == 14720
        assert len(metrics_data["validation_metrics"]) == 3
        assert len(metrics_data["test_metrics"]) == 2
        assert metrics_data["mae_improvement_pct"] > 15.0
        assert metrics_data["wape_improvement_pct"] > 15.0

"""Feature preprocessing and chronological splitting pipelines for BusDemand AI.

Provides reusable ColumnTransformers and strict temporal split boundaries
guaranteeing zero target leakage and appropriate handling of categorical features.
"""

from typing import Dict, List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature Column Specifications
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES: List[str] = [
    "route_id",
    "origin_city_id",
    "destination_city_id",
    "departure_slot",
    "season",
]

NUMERICAL_FEATURES: List[str] = [
    "distance_km",
    "journey_duration_minutes",
    "route_demand_multiplier",
    "route_mobility_score",
    "departure_hour",
    "day_of_week",
    "is_weekend",
    "month",
    "quarter",
    "is_holiday_origin",
    "is_holiday_destination",
    "is_national_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_long_weekend",
    "is_school_vacation_origin",
    "is_school_vacation_destination",
    "is_college_vacation_origin",
    "is_college_vacation_destination",
    "bus_capacity",
]

MODEL_FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
PREDICTION_FEATURES: List[str] = MODEL_FEATURE_COLUMNS

TARGET_COLUMN: str = "passenger_demand"
METADATA_COLUMNS: List[str] = ["trip_id", "departure_date", "departure_time"]
LEAKAGE_COLUMNS: List[str] = ["capacity_exceeded", "estimated_unserved_demand"]

# Strict exclusion list
EXCLUDED_COLUMNS: List[str] = [
    "trip_id",
    "departure_date",
    "departure_time",
    "passenger_demand",
    "capacity_exceeded",
    "estimated_unserved_demand",
]


# ---------------------------------------------------------------------------
# Chronological Splitting
# ---------------------------------------------------------------------------

def split_dataframes(
    df: pd.DataFrame,
    train_end_date: str = "2025-12-31",
    val_end_date: str = "2026-06-30",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition DataFrame into train, val, and test slices without dropping columns."""
    if "departure_date" not in df.columns:
        raise ValueError("DataFrame must contain departure_date.")
    train_mask = df["departure_date"] <= train_end_date
    val_mask = (df["departure_date"] > train_end_date) & (df["departure_date"] <= val_end_date)
    test_mask = df["departure_date"] > val_end_date
    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

def split_chronological(
    df: pd.DataFrame,
    train_end_date: str = "2025-12-31",
    val_end_date: str = "2026-06-30",
) -> Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    """Partition trip dataset chronologically without random shuffling.

    Split Boundaries:
    - Training: 2024-01-01 to train_end_date (2025-12-31) -> 58,480 trips (731 days)
    - Validation: 2026-01-01 to val_end_date (2026-06-30) -> 14,480 trips (181 days)
    - Test: 2026-07-01 to 2026-12-31                      -> 14,720 trips (184 days)

    Args:
        df: Processed trip demand DataFrame.
        train_end_date: Cutoff date string for training partition.
        val_end_date: Cutoff date string for validation partition.

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    """
    if "departure_date" not in df.columns or "passenger_demand" not in df.columns:
        raise ValueError("DataFrame must contain departure_date and passenger_demand.")

    train_mask = df["departure_date"] <= train_end_date
    val_mask = (df["departure_date"] > train_end_date) & (df["departure_date"] <= val_end_date)
    test_mask = df["departure_date"] > val_end_date

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train = train_df[MODEL_FEATURE_COLUMNS].copy()
    y_train = train_df["passenger_demand"].copy()

    X_val = val_df[MODEL_FEATURE_COLUMNS].copy()
    y_val = val_df["passenger_demand"].copy()

    X_test = test_df[MODEL_FEATURE_COLUMNS].copy()
    y_test = test_df["passenger_demand"].copy()

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------------------------------------------------------------------------
# Preprocessor Factories
# ---------------------------------------------------------------------------

def build_rf_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for Random Forest using One-Hot Encoding for categoricals."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_hgb_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for HistGradientBoosting using OrdinalEncoder."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

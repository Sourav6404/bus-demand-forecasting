"""Unit tests for the synthetic scenario-based city profile generator and validator."""

import pytest
import pandas as pd
import numpy as np

from src.data.generate_city_profiles import (
    build_city_profiles,
    add_mobility_indices,
    validate_city_profiles,
    CITY_SCENARIOS,
    COMPONENT_COLUMNS,
)
from src.config import TARGET_CITIES, RAW_DATA_DIR


def test_build_city_profiles_count():
    """Verify exactly 8 cities are generated."""
    df = build_city_profiles()
    assert len(df) == 8
    assert len(df) == len(TARGET_CITIES)


def test_unique_city_ids():
    """Verify city IDs are unique and match the 8 target corridor cities."""
    df = build_city_profiles()
    expected_ids = {"BLR", "MYS", "COK", "CCJ", "TCR", "TVM", "MAA", "CBE"}
    assert set(df["city_id"]) == expected_ids
    assert df["city_id"].is_unique


def test_no_missing_values():
    """Ensure no null or NaN values exist in the city profile table."""
    df = build_city_profiles()
    assert not df.isnull().any().any()


def test_score_ranges_zero_to_one():
    """Verify all raw scores and calculated indices fall within [0.0, 1.0]."""
    df = build_city_profiles()
    score_cols = COMPONENT_COLUMNS + [
        "demand_generation_index",
        "demand_attraction_index",
        "city_mobility_index",
    ]
    for col in score_cols:
        assert (df[col] >= 0.0).all(), f"{col} has values below 0.0"
        assert (df[col] <= 1.0).all(), f"{col} has values above 1.0"


def test_mobility_indices_calculation():
    """Verify manual calculation matches add_mobility_indices output."""
    df = pd.DataFrame(CITY_SCENARIOS)
    scored = add_mobility_indices(df)
    blr = scored[scored["city_id"] == "BLR"].iloc[0]

    expected_gen = (
        0.10 * 0.98 + 0.25 * 1.00 + 0.20 * 0.95 + 0.05 * 0.55 + 0.15 * 1.00 + 0.25 * 0.95
    )
    expected_att = (
        0.10 * 0.98 + 0.20 * 1.00 + 0.10 * 0.95 + 0.25 * 0.55 + 0.25 * 1.00 + 0.10 * 0.95
    )
    expected_cmi = (expected_gen + expected_att) / 2

    assert pytest.approx(blr["demand_generation_index"], rel=1e-5) == expected_gen
    assert pytest.approx(blr["demand_attraction_index"], rel=1e-5) == expected_att
    assert pytest.approx(blr["city_mobility_index"], rel=1e-5) == expected_cmi


def test_validator_detects_duplicates():
    """Validator should raise ValueError when duplicate city IDs are present."""
    df = build_city_profiles()
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate city IDs found"):
        validate_city_profiles(df_dup)


def test_validator_detects_missing_values():
    """Validator should raise ValueError when null values are present."""
    df = build_city_profiles()
    df_null = df.copy()
    df_null.loc[0, "employment_score"] = np.nan
    with pytest.raises(ValueError, match="must not contain missing values"):
        validate_city_profiles(df_null)


def test_validator_detects_out_of_range_scores():
    """Validator should raise ValueError when scores are outside [0, 1]."""
    df = build_city_profiles()
    df_invalid = df.copy()
    df_invalid.loc[0, "tourism_score"] = 1.25
    with pytest.raises(ValueError, match="Scores outside the 0–1 range"):
        validate_city_profiles(df_invalid)


def test_validator_detects_missing_column():
    """Validator should raise ValueError when a required column is missing."""
    df = build_city_profiles()
    df_missing = df.drop(columns=["urban_scale_score"])
    with pytest.raises(ValueError, match="Missing required city profile columns"):
        validate_city_profiles(df_missing)


def test_persisted_csv():
    """Verify that data/raw/city_profiles.csv exists, is non-empty, and valid."""
    csv_path = RAW_DATA_DIR / "city_profiles.csv"
    assert csv_path.exists()
    loaded_df = pd.read_csv(csv_path)
    assert len(loaded_df) == 8
    validate_city_profiles(loaded_df)

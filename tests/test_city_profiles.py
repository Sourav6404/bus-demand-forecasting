"""Unit tests for the synthetic city profile generator and validator."""

import pytest
import pandas as pd
import numpy as np

from src.data.generate_city_profiles import (
    generate_city_profiles,
    validate_city_profiles,
    calculate_demand_generation_index,
    calculate_demand_attraction_index,
    calculate_city_mobility_index,
    RAW_CITY_BENCHMARKS,
)
from src.config import TARGET_CITIES, RAW_DATA_DIR


def test_generate_city_profiles_count():
    """Verify exactly 8 cities are generated."""
    df = generate_city_profiles(output_path=None)
    assert len(df) == 8
    assert len(df) == len(TARGET_CITIES)


def test_unique_city_ids():
    """Verify city IDs are unique and match target cities."""
    df = generate_city_profiles(output_path=None)
    expected_ids = {"BLR", "MYS", "MAA", "CJB", "COK", "CCJ", "TCR", "TRV"}
    assert set(df["city_id"]) == expected_ids
    assert df["city_id"].is_unique


def test_no_missing_values():
    """Ensure no null or NaN values exist in the output."""
    df = generate_city_profiles(output_path=None)
    assert not df.isnull().any().any()


def test_score_ranges():
    """Verify all raw scores and calculated indices fall within [0.0, 10.0]."""
    df = generate_city_profiles(output_path=None)
    score_cols = [
        "urban_scale_score",
        "employment_score",
        "education_score",
        "tourism_score",
        "economic_activity_score",
        "migration_score",
        "demand_generation_index",
        "demand_attraction_index",
        "city_mobility_index",
    ]
    for col in score_cols:
        assert (df[col] >= 0.0).all(), f"{col} has values below 0.0"
        assert (df[col] <= 10.0).all(), f"{col} has values above 10.0"


def test_tier_hierarchies():
    """Tier 1 hubs (Bengaluru, Chennai) should exhibit higher urban scale than Tier 2."""
    df = generate_city_profiles(output_path=None)
    tier1_scale = df[df["tier"] == 1]["urban_scale_score"]
    tier2_scale = df[df["tier"] == 2]["urban_scale_score"]
    assert tier1_scale.min() > tier2_scale.max()


def test_validator_detects_duplicates():
    """Validator should raise ValueError when duplicate city IDs are present."""
    df = generate_city_profiles(output_path=None)
    # Force duplicate
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate city IDs"):
        validate_city_profiles(df_dup)


def test_validator_detects_missing_values():
    """Validator should raise ValueError when null values are present."""
    df = generate_city_profiles(output_path=None)
    df_null = df.copy()
    df_null.loc[0, "employment_score"] = np.nan
    with pytest.raises(ValueError, match="Missing values found"):
        validate_city_profiles(df_null)


def test_validator_detects_out_of_range_scores():
    """Validator should raise ValueError when scores are outside [0, 10]."""
    df = generate_city_profiles(output_path=None)
    df_invalid = df.copy()
    df_invalid.loc[0, "tourism_score"] = 12.5
    with pytest.raises(ValueError, match="outside valid range"):
        validate_city_profiles(df_invalid)


def test_persisted_csv():
    """Verify that city_profiles.csv exists and can be loaded accurately."""
    csv_path = RAW_DATA_DIR / "city_profiles.csv"
    generate_city_profiles(output_path=csv_path)
    assert csv_path.exists()
    loaded_df = pd.read_csv(csv_path)
    assert len(loaded_df) == 8
    validate_city_profiles(loaded_df)

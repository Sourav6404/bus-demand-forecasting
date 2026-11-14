"""Unit tests for the directed route profile generator and validator."""

import pytest
import pandas as pd
import numpy as np

from src.data.generate_route_profiles import (
    generate_route_profiles,
    calculate_route_features,
    validate_route_profiles,
    load_city_profiles,
    BASE_ROUTE_DEFINITIONS,
    REQUIRED_ROUTE_COLUMNS,
)
from src.config import RAW_DATA_DIR, DEFAULT_BUS_CAPACITY


@pytest.fixture
def city_profiles():
    """Load valid city profiles fixture."""
    return load_city_profiles()


@pytest.fixture
def route_profiles(city_profiles):
    """Generate route profiles fixture without writing to disk."""
    return generate_route_profiles()


def test_route_count(route_profiles):
    """Verify exactly 20 directed routes are generated."""
    assert len(route_profiles) == 20


def test_unique_route_ids(route_profiles):
    """Verify route IDs are unique across the entire network."""
    assert route_profiles["route_id"].is_unique
    assert len(route_profiles["route_id"].unique()) == 20


def test_unique_directed_pairs(route_profiles):
    """Verify all 20 directed origin-destination pairs are unique."""
    od_pairs = list(zip(route_profiles["origin_city_id"], route_profiles["destination_city_id"]))
    assert len(od_pairs) == len(set(od_pairs))


def test_no_self_routes(route_profiles):
    """Ensure no routes have the same origin and destination."""
    self_routes = route_profiles[route_profiles["origin_city_id"] == route_profiles["destination_city_id"]]
    assert self_routes.empty, f"Found self-routes: {self_routes['route_id'].tolist()}"


def test_valid_city_ids(route_profiles, city_profiles):
    """Ensure all origin and destination IDs exist in city profiles."""
    valid_ids = set(city_profiles["city_id"].unique())
    assert set(route_profiles["origin_city_id"]).issubset(valid_ids)
    assert set(route_profiles["destination_city_id"]).issubset(valid_ids)


def test_positive_metrics(route_profiles):
    """Verify distances, durations, speeds, capacity, and multipliers are strictly positive."""
    assert (route_profiles["distance_km"] > 0).all()
    assert (route_profiles["estimated_duration_hours"] > 0).all()
    assert (route_profiles["average_speed_kmph"] > 0).all()
    assert (route_profiles["bus_capacity"] > 0).all()
    assert (route_profiles["route_demand_multiplier"] > 0).all()


def test_bus_capacity_standard(route_profiles):
    """Verify standard bus capacity matches configuration."""
    assert (route_profiles["bus_capacity"] == DEFAULT_BUS_CAPACITY).all()


def test_no_missing_values(route_profiles):
    """Verify no NaN/null values exist across all required columns."""
    for col in REQUIRED_ROUTE_COLUMNS:
        assert not route_profiles[col].isna().any(), f"Column {col} contains missing values"


def test_feature_calculations(route_profiles, city_profiles):
    """Verify mathematical correctness of speed and mobility score."""
    dgi_map = dict(zip(city_profiles["city_id"], city_profiles["demand_generation_index"]))
    dai_map = dict(zip(city_profiles["city_id"], city_profiles["demand_attraction_index"]))

    for _, row in route_profiles.iterrows():
        # Check speed = distance / duration
        expected_speed = round(row["distance_km"] / row["estimated_duration_hours"], 2)
        assert pytest.approx(row["average_speed_kmph"], rel=1e-3) == expected_speed

        # Check origin DGI and destination DAI mapping from city profiles
        expected_orig_dgi = dgi_map[row["origin_city_id"]]
        expected_dest_dai = dai_map[row["destination_city_id"]]
        assert pytest.approx(row["origin_generation_index"], rel=1e-5) == expected_orig_dgi
        assert pytest.approx(row["destination_attraction_index"], rel=1e-5) == expected_dest_dai

        # Check route_mobility_score = (origin DGI + dest DAI) / 2
        expected_mobility = (expected_orig_dgi + expected_dest_dai) / 2.0
        assert pytest.approx(row["route_mobility_score"], abs=1e-3) == expected_mobility


def test_bidirectional_pairing(route_profiles):
    """Ensure every directed route has a corresponding reverse route in the network."""
    pairs = set(zip(route_profiles["origin_city_id"], route_profiles["destination_city_id"]))
    for orig, dest in pairs:
        assert (dest, orig) in pairs, f"Missing reverse direction for route: {orig}->{dest}"


def test_validator_detects_duplicate_route_ids(route_profiles):
    """Validator should reject duplicate route IDs."""
    invalid = route_profiles.copy()
    invalid.loc[1, "route_id"] = invalid.loc[0, "route_id"]
    with pytest.raises(ValueError, match="Duplicate route IDs"):
        validate_route_profiles(invalid)


def test_validator_detects_duplicate_od_pairs(route_profiles):
    """Validator should reject duplicate directed origin-destination pairs."""
    invalid = route_profiles.copy()
    invalid.loc[1, "origin_city_id"] = invalid.loc[0, "origin_city_id"]
    invalid.loc[1, "destination_city_id"] = invalid.loc[0, "destination_city_id"]
    with pytest.raises(ValueError, match="Duplicate directed origin-destination"):
        validate_route_profiles(invalid)


def test_validator_detects_self_routes(route_profiles):
    """Validator should reject self-routes (origin == destination)."""
    invalid = route_profiles.copy()
    invalid.loc[0, "destination_city_id"] = invalid.loc[0, "origin_city_id"]
    with pytest.raises(ValueError, match="Self-routes detected"):
        validate_route_profiles(invalid)


def test_validator_detects_invalid_count(route_profiles):
    """Validator should reject route sets not having exactly 20 routes."""
    invalid = route_profiles.iloc[:-1].copy()
    with pytest.raises(ValueError, match="Expected exactly 20 routes"):
        validate_route_profiles(invalid)


def test_validator_detects_non_positive_metrics(route_profiles):
    """Validator should reject non-positive distances and durations."""
    invalid_dist = route_profiles.copy()
    invalid_dist.loc[0, "distance_km"] = -10.0
    with pytest.raises(ValueError, match="must be strictly positive"):
        validate_route_profiles(invalid_dist)

    invalid_dur = route_profiles.copy()
    invalid_dur.loc[0, "estimated_duration_hours"] = 0.0
    with pytest.raises(ValueError, match="must be strictly positive"):
        validate_route_profiles(invalid_dur)


def test_validator_detects_unknown_city_id(route_profiles):
    """Validator should reject city IDs not in valid set."""
    invalid = route_profiles.copy()
    invalid.loc[0, "origin_city_id"] = "XYZ"
    valid_ids = {"BLR", "MYS", "MAA", "CBE", "COK", "CCJ", "TCR", "TVM"}
    with pytest.raises(ValueError, match="Unknown city IDs"):
        validate_route_profiles(invalid, valid_city_ids=valid_ids)


def test_persisted_csv():
    """Verify data/raw/route_profiles.csv exists and matches schema and rules."""
    csv_path = RAW_DATA_DIR / "route_profiles.csv"
    assert csv_path.exists(), f"File {csv_path} does not exist"
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == 20
    validate_route_profiles(loaded)

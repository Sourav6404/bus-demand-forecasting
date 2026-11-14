"""Unit tests for the trip-level synthetic passenger demand simulator."""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEFAULT_BUS_CAPACITY,
    DEFAULT_RANDOM_SEED,
    DEPARTURES_PER_ROUTE_DAY,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SIMULATION_END_DATE,
    SIMULATION_START_DATE,
)
from src.data.generate_trip_demand import (
    OUTPUT_SCHEMA,
    PREDICTION_FEATURES,
    TARGET_AND_OPERATIONAL_COLUMNS,
    build_trip_grid,
    calculate_operational_metrics,
    compute_expected_demand,
    generate_trip_demand,
    load_input_datasets,
    sample_passenger_demand,
    validate_trip_demand,
)


@pytest.fixture(scope="module")
def input_data():
    """Load raw city, route, and calendar datasets once for test module."""
    return load_input_datasets(RAW_DATA_DIR)


@pytest.fixture(scope="module")
def full_trip_demand():
    """Generate or load persisted trip demand dataset for module testing."""
    csv_path = PROCESSED_DATA_DIR / "trip_demand.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) == 87680:
            return df
    return generate_trip_demand(RAW_DATA_DIR, seed=DEFAULT_RANDOM_SEED)


# 1. Total row count
def test_trip_demand_total_row_count(full_trip_demand):
    """Verify exactly 87,680 trips are generated (20 routes * 1,096 dates * 4 slots)."""
    assert len(full_trip_demand) == 87680


# 2. Every route has 4 departures per day
def test_four_departures_per_route_per_day(full_trip_demand):
    """Verify every route on every calendar date has exactly 4 departure slots."""
    counts = full_trip_demand.groupby(["route_id", "departure_date"]).size()
    assert (counts == DEPARTURES_PER_ROUTE_DAY).all()
    assert len(counts) == 20 * 1096  # 21,920 route-date pairs


# 3. Every date in configured range is represented
def test_all_dates_represented(full_trip_demand):
    """Verify simulation boundaries and continuity across all 1,096 calendar dates."""
    dates = pd.to_datetime(full_trip_demand["departure_date"].unique())
    dates = dates.sort_values()
    assert len(dates) == 1096
    assert dates.min() == pd.to_datetime(SIMULATION_START_DATE)
    assert dates.max() == pd.to_datetime(SIMULATION_END_DATE)
    diffs = dates.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all()


# 4. All 20 directed routes are represented
def test_all_twenty_directed_routes_represented(full_trip_demand):
    """Verify all 20 directed intercity routes are present with equal representation."""
    route_counts = full_trip_demand["route_id"].value_counts()
    assert len(route_counts) == 20
    assert (route_counts == 1096 * 4).all()


# 5. Reverse routes remain distinct
def test_reverse_routes_distinct(full_trip_demand):
    """Verify that bidirectional route pairs maintain distinct characteristics and demand."""
    # Check BLR-MAA vs MAA-BLR
    blr_maa = full_trip_demand[full_trip_demand["route_id"] == "R_BLR_MAA"]
    maa_blr = full_trip_demand[full_trip_demand["route_id"] == "R_MAA_BLR"]

    assert blr_maa["origin_city_id"].iloc[0] == "BLR"
    assert blr_maa["destination_city_id"].iloc[0] == "MAA"
    assert maa_blr["origin_city_id"].iloc[0] == "MAA"
    assert maa_blr["destination_city_id"].iloc[0] == "BLR"

    # Demand multipliers and mobility scores must match their distinct profiles
    assert blr_maa["route_demand_multiplier"].iloc[0] == 1.25
    assert maa_blr["route_demand_multiplier"].iloc[0] == 1.22
    assert blr_maa["route_mobility_score"].iloc[0] != maa_blr["route_mobility_score"].iloc[0]

    # Mean demand across 3 years must be distinct
    assert abs(blr_maa["passenger_demand"].mean() - maa_blr["passenger_demand"].mean()) > 0.5


# 6. No duplicate trip IDs exist
def test_unique_trip_ids(full_trip_demand):
    """Verify trip_id uniqueness across the entire 87,680-row dataset."""
    assert full_trip_demand["trip_id"].is_unique
    assert full_trip_demand["trip_id"].str.startswith("TRIP_").all()


# 7. No negative passenger demand exists
def test_non_negative_passenger_demand(full_trip_demand):
    """Verify passenger demand is strictly non-negative."""
    assert (full_trip_demand["passenger_demand"] >= 0).all()


# 8. Passenger demand is an integer
def test_passenger_demand_is_integer(full_trip_demand):
    """Verify passenger demand contains integer counts."""
    assert pd.api.types.is_integer_dtype(full_trip_demand["passenger_demand"])


# 9. Capacity calculations are correct
def test_capacity_and_operational_calculations(full_trip_demand):
    """Verify operational indicators: capacity_exceeded and estimated_unserved_demand."""
    demand = full_trip_demand["passenger_demand"]
    cap = full_trip_demand["bus_capacity"]

    expected_exceeded = demand > cap
    assert (full_trip_demand["capacity_exceeded"] == expected_exceeded).all()

    expected_unserved = np.maximum(0, demand - cap)
    assert (full_trip_demand["estimated_unserved_demand"] == expected_unserved).all()

    # When not exceeded, unserved demand must be 0
    not_exceeded = ~full_trip_demand["capacity_exceeded"]
    assert (full_trip_demand.loc[not_exceeded, "estimated_unserved_demand"] == 0).all()

    # When exceeded, unserved demand must equal demand - capacity
    exceeded = full_trip_demand["capacity_exceeded"]
    assert (
        full_trip_demand.loc[exceeded, "estimated_unserved_demand"]
        == demand[exceeded] - cap[exceeded]
    ).all()


# 10. No missing critical fields exist
def test_no_missing_values(full_trip_demand):
    """Verify zero missing values across all output schema columns."""
    for col in OUTPUT_SCHEMA:
        assert not full_trip_demand[col].isna().any(), f"Column {col} has null values"


# 11. Calendar and route joins are correct
def test_route_and_calendar_joins(full_trip_demand):
    """Verify route attributes and calendar features are correctly joined."""
    sample = full_trip_demand[
        (full_trip_demand["route_id"] == "R_BLR_MYS")
        & (full_trip_demand["departure_date"] == "2024-01-01")
        & (full_trip_demand["departure_slot"] == "DEP_01")
    ].iloc[0]

    assert sample["origin_city_id"] == "BLR"
    assert sample["destination_city_id"] == "MYS"
    assert sample["distance_km"] == 145.0
    assert sample["journey_duration_minutes"] == 180
    assert sample["departure_time"] == "06:00"
    assert sample["departure_hour"] == 6
    assert sample["day_of_week"] == 0  # 2024-01-01 is Monday
    assert sample["season"] == "Winter"
    assert sample["bus_capacity"] == 45


# 12. Random-seed reproducibility works
def test_random_seed_reproducibility(input_data):
    """Verify that identical random seeds produce identical demand arrays."""
    cities_df, routes_df, calendar_df = input_data
    grid = build_trip_grid(routes_df, calendar_df)
    mu = compute_expected_demand(grid)

    sample1 = sample_passenger_demand(mu[:500], seed=42)
    sample2 = sample_passenger_demand(mu[:500], seed=42)
    assert np.array_equal(sample1, sample2)

    sample_diff_seed = sample_passenger_demand(mu[:500], seed=999)
    assert not np.array_equal(sample1, sample_diff_seed)


# 13. The dataset has meaningful demand variation
def test_meaningful_demand_variation(full_trip_demand):
    """Verify expected behavioral variance patterns in the simulated dataset."""
    demand = full_trip_demand["passenger_demand"]

    # Overall spread
    assert demand.std() > 10.0
    assert demand.min() >= 0
    assert demand.max() > 100  # Festive surges exist

    # Friday/Sunday peaks vs midweek for outbound hub corridors
    outbound_hub = full_trip_demand[
        full_trip_demand["origin_city_id"].isin(["BLR", "MAA"])
        & ~full_trip_demand["destination_city_id"].isin(["BLR", "MAA"])
    ]
    fri_evening = outbound_hub[
        (outbound_hub["day_of_week"] == 4)
        & (outbound_hub["departure_slot"].isin(["DEP_03", "DEP_04"]))
    ]["passenger_demand"].mean()

    tue_midday = outbound_hub[
        (outbound_hub["day_of_week"] == 1)
        & (outbound_hub["departure_slot"] == "DEP_02")
    ]["passenger_demand"].mean()

    assert fri_evening > tue_midday * 1.30, (
        f"Expected Friday evening peak ({fri_evening:.1f}) to exceed "
        f"Tuesday midday ({tue_midday:.1f}) by >= 30%"
    )

    # Long weekend demand should exceed non-long weekend demand
    long_we_mean = full_trip_demand[full_trip_demand["is_long_weekend"]]["passenger_demand"].mean()
    regular_mean = full_trip_demand[~full_trip_demand["is_long_weekend"]]["passenger_demand"].mean()
    assert long_we_mean > regular_mean


# 14. No target-derived fields appear among prediction features
def test_target_leakage_segregation():
    """Verify prediction features do not include target or operational post-demand fields."""
    for post_field in TARGET_AND_OPERATIONAL_COLUMNS:
        assert post_field not in PREDICTION_FEATURES, (
            f"Target leakage: {post_field} found in PREDICTION_FEATURES"
        )


# 15. All generated values meet documented validation bounds
def test_documented_validation_bounds(full_trip_demand):
    """Verify all columns conform to expected domain ranges and validation boundaries."""
    assert full_trip_demand["departure_hour"].isin([6, 11, 17, 22]).all()
    assert full_trip_demand["departure_slot"].isin(["DEP_01", "DEP_02", "DEP_03", "DEP_04"]).all()
    assert full_trip_demand["month"].between(1, 12).all()
    assert full_trip_demand["quarter"].between(1, 4).all()
    assert full_trip_demand["day_of_week"].between(0, 6).all()
    assert set(full_trip_demand["season"].unique()) == {
        "Winter",
        "Summer",
        "Monsoon",
        "Post-Monsoon",
    }
    assert (full_trip_demand["distance_km"] > 0).all()
    assert (full_trip_demand["journey_duration_minutes"] > 0).all()
    assert (full_trip_demand["bus_capacity"] == DEFAULT_BUS_CAPACITY).all()

    # Capacity exceeded percentage should be in realistic operational range [15%, 30%]
    exceeded_pct = full_trip_demand["capacity_exceeded"].mean() * 100
    assert 15.0 <= exceeded_pct <= 30.0, f"Exceeded percentage {exceeded_pct:.2f}% out of range"

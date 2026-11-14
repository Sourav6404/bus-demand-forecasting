"""Synthetic Passenger Demand Simulator for BusDemand AI.

This module simulates realistic trip-level passenger demand for the 20 directed
intercity bus routes across the 3-year horizon (2024-01-01 to 2026-12-31, 1,096 days),
with 4 daily departures per route, yielding exactly 87,680 scheduled trips.

IMPORTANT:
- Demand is simulated using a behavioral extended Gravity Model with directional,
  temporal, seasonal, holiday, and vacation multipliers.
- Passenger count arrivals follow a Negative Binomial distribution to capture
  natural party-size overdispersion without arbitrary clipping.
- Target: `passenger_demand` (unconstrained requested volume, not physical bus occupancy).
- Forecast Horizon: 24 hours prior to scheduled departure. Calendar/holiday features
  are assumed available at cutoff (planned gazette dates serve as reference scenarios).
- Prediction features are strictly segregated from target/operational fields
  to prevent target leakage.
- `trip_id` is an audit/join identifier and should not be passed directly as a feature into ML models.
  Route, city, and slot IDs should be treated as categorical variables where appropriate.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_BUS_CAPACITY,
    DEFAULT_RANDOM_SEED,
    DEPARTURE_SLOTS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_CITIES,
)

# ---------------------------------------------------------------------------
# Schema Definitions & Column Groups
# ---------------------------------------------------------------------------

PREDICTION_FEATURES: List[str] = [
    # Trip identifiers
    "trip_id",
    "route_id",
    "origin_city_id",
    "destination_city_id",
    # Schedule
    "departure_date",
    "departure_time",
    "departure_hour",
    "departure_slot",
    # Route characteristics
    "distance_km",
    "journey_duration_minutes",
    "route_demand_multiplier",
    "route_mobility_score",
    # Calendar features
    "day_of_week",
    "is_weekend",
    "month",
    "quarter",
    "season",
    "is_holiday_origin",
    "is_holiday_destination",
    "is_national_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_long_weekend",
    # Vacation indicators
    "is_school_vacation_origin",
    "is_school_vacation_destination",
    "is_college_vacation_origin",
    "is_college_vacation_destination",
    # Fleet spec (known at scheduling time)
    "bus_capacity",
]

TARGET_AND_OPERATIONAL_COLUMNS: List[str] = [
    "passenger_demand",
    "capacity_exceeded",
    "estimated_unserved_demand",
]

OUTPUT_SCHEMA: List[str] = PREDICTION_FEATURES + TARGET_AND_OPERATIONAL_COLUMNS


# ---------------------------------------------------------------------------
# State Mapping Utilities
# ---------------------------------------------------------------------------

STATE_NAME_TO_CODE: Dict[str, str] = {
    "Karnataka": "KA",
    "Kerala": "KL",
    "Tamil Nadu": "TN",
}


def get_city_to_state_code_map() -> Dict[str, str]:
    """Map city ID to 2-letter state code based on TARGET_CITIES configuration."""
    return {c.city_id: STATE_NAME_TO_CODE[c.state] for c in TARGET_CITIES}


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_input_datasets(
    raw_dir: Path = RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate city profiles, route profiles, and calendar features.

    Args:
        raw_dir: Path to directory containing raw CSV artifacts.

    Returns:
        Tuple of (city_profiles_df, route_profiles_df, calendar_features_df).

    Raises:
        FileNotFoundError: If any required CSV is missing.
        ValueError: If input datasets fail basic integrity checks.
    """
    city_path = raw_dir / "city_profiles.csv"
    route_path = raw_dir / "route_profiles.csv"
    calendar_path = raw_dir / "calendar_features.csv"

    for p in [city_path, route_path, calendar_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input dataset not found: {p.resolve()}")

    cities = pd.read_csv(city_path)
    routes = pd.read_csv(route_path)
    calendar = pd.read_csv(calendar_path)

    if len(cities) != 8:
        raise ValueError(f"Expected 8 city profiles, found {len(cities)}")
    if len(routes) != 20:
        raise ValueError(f"Expected 20 route profiles, found {len(routes)}")
    if len(calendar) != 1096:
        raise ValueError(f"Expected 1,096 calendar rows, found {len(calendar)}")

    return cities, routes, calendar


# ---------------------------------------------------------------------------
# Trip Schedule Grid Builder
# ---------------------------------------------------------------------------

def build_trip_grid(
    routes_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    slots: List[Dict[str, str]] = DEPARTURE_SLOTS,
) -> pd.DataFrame:
    """Construct full Cartesian product grid of routes, dates, and departure slots.

    Total rows generated: 20 routes * 1,096 dates * 4 slots = 87,680 trips.

    Args:
        routes_df: 20 directed route profiles.
        calendar_df: 1,096 calendar feature records.
        slots: List of 4 daily departure slot configurations.

    Returns:
        DataFrame containing joined route, calendar, and schedule features.
    """
    city_to_state = get_city_to_state_code_map()

    # Pre-compute route attributes
    route_meta = []
    for _, r in routes_df.iterrows():
        orig = r["origin_city_id"]
        dest = r["destination_city_id"]
        route_meta.append({
            "route_id": r["route_id"],
            "origin_city_id": orig,
            "destination_city_id": dest,
            "origin_state": city_to_state[orig],
            "destination_state": city_to_state[dest],
            "distance_km": float(r["distance_km"]),
            "journey_duration_minutes": int(round(float(r["estimated_duration_hours"]) * 60)),
            "route_demand_multiplier": float(r["route_demand_multiplier"]),
            "route_mobility_score": float(r["route_mobility_score"]),
            "bus_capacity": int(r.get("bus_capacity", DEFAULT_BUS_CAPACITY)),
        })
    routes_pre = pd.DataFrame(route_meta)

    # Pre-parse slot information
    slot_records = []
    for s in slots:
        slot_records.append({
            "departure_slot": s["slot_id"],
            "departure_time": s["time"],
            "departure_hour": int(s["time"].split(":")[0]),
        })
    slots_pre = pd.DataFrame(slot_records)

    # 1. Expand calendar x slots (1,096 * 4 = 4,384)
    cal_slots = calendar_df.merge(slots_pre, how="cross")

    # 2. Expand with routes (4,384 * 20 = 87,680)
    grid = routes_pre.merge(cal_slots, how="cross")

    # Rename date to departure_date
    grid["departure_date"] = grid["date"]

    # Construct stable, unique trip_id
    date_compact = grid["departure_date"].str.replace("-", "", regex=False)
    grid["trip_id"] = (
        "TRIP_"
        + grid["route_id"]
        + "_"
        + date_compact
        + "_"
        + grid["departure_slot"]
    )

    # Map state-specific holiday flags to origin and destination
    orig_ka = grid["origin_state"] == "KA"
    orig_kl = grid["origin_state"] == "KL"
    orig_tn = grid["origin_state"] == "TN"

    dest_ka = grid["destination_state"] == "KA"
    dest_kl = grid["destination_state"] == "KL"
    dest_tn = grid["destination_state"] == "TN"

    # Holiday indicators
    grid["is_holiday_origin"] = (
        (orig_ka & grid["is_holiday_ka"])
        | (orig_kl & grid["is_holiday_kl"])
        | (orig_tn & grid["is_holiday_tn"])
    )
    grid["is_holiday_destination"] = (
        (dest_ka & grid["is_holiday_ka"])
        | (dest_kl & grid["is_holiday_kl"])
        | (dest_tn & grid["is_holiday_tn"])
    )

    # Vacation indicators
    grid["is_school_vacation_origin"] = (
        (orig_ka & grid["is_school_vacation_ka"])
        | (orig_kl & grid["is_school_vacation_kl"])
        | (orig_tn & grid["is_school_vacation_tn"])
    )
    grid["is_school_vacation_destination"] = (
        (dest_ka & grid["is_school_vacation_ka"])
        | (dest_kl & grid["is_school_vacation_kl"])
        | (dest_tn & grid["is_school_vacation_tn"])
    )
    grid["is_college_vacation_origin"] = (
        (orig_ka & grid["is_college_vacation_ka"])
        | (orig_kl & grid["is_college_vacation_kl"])
        | (orig_tn & grid["is_college_vacation_tn"])
    )
    grid["is_college_vacation_destination"] = (
        (dest_ka & grid["is_college_vacation_ka"])
        | (dest_kl & grid["is_college_vacation_kl"])
        | (dest_tn & grid["is_college_vacation_tn"])
    )

    # Sort deterministically
    grid = grid.sort_values(
        by=["departure_date", "route_id", "departure_slot"]
    ).reset_index(drop=True)

    return grid


# ---------------------------------------------------------------------------
# Behavioral Demand Calculation Engine
# ---------------------------------------------------------------------------

def compute_expected_demand(
    df: pd.DataFrame,
    base_scale: float = 37.5,
) -> np.ndarray:
    """Calculate deterministic expected passenger demand rate (mu) for each trip.

    Formulation:
        mu = BaseRate * M_slot * M_dow * M_season * M_holiday * M_vacation

    Components:
    1. BaseRate = base_scale * route_mobility_score * route_demand_multiplier
       (Avoids double-counting: uses route_mobility_score and RDM exclusively).
    2. M_slot: Distance-slot interaction (morning/evening peaks for short routes,
       night sleeper preference for long distances > 250 km).
    3. M_dow: Day-of-week with directional commuter hub asymmetry (Friday evening
       outbound surge from BLR/MAA to hometowns; Sunday evening return rush to BLR/MAA).
    4. M_season: IMD meteorological season + monsoon suppression in Kerala.
    5. M_holiday: Directional long-weekend and festival travel surges.
    6. M_vacation: School and college vacation scenario increments.

    Args:
        df: DataFrame containing the trip grid with route and calendar features.
        base_scale: Baseline scale parameter in seats (default 37.5).

    Returns:
        1D numpy array of expected demand rates (mu) > 0.
    """
    n = len(df)
    orig = df["origin_city_id"].to_numpy()
    dest = df["destination_city_id"].to_numpy()
    orig_state = df["origin_state"].to_numpy()
    dest_state = df["destination_state"].to_numpy()
    dist = df["distance_km"].to_numpy()
    rms = df["route_mobility_score"].to_numpy()
    rdm = df["route_demand_multiplier"].to_numpy()
    slot = df["departure_slot"].to_numpy()
    dow = df["day_of_week"].to_numpy()
    season = df["season"].to_numpy()

    # Directional hub masks
    is_outbound_hub = np.isin(orig, ["BLR", "MAA"]) & ~np.isin(dest, ["BLR", "MAA"])
    is_inbound_hub = np.isin(dest, ["BLR", "MAA"]) & ~np.isin(orig, ["BLR", "MAA"])
    is_inter_hub = np.isin(orig, ["BLR", "MAA"]) & np.isin(dest, ["BLR", "MAA"])

    # 1. Base Rate
    base_rate = base_scale * rms * rdm

    # 2. Slot Multiplier by Distance Category
    # Short routes (< 250 km) favor Early Morning (DEP_01) & Evening (DEP_03)
    # Long routes (>= 250 km) favor Night Sleeper (DEP_04)
    m_slot = np.ones(n, dtype=float)
    is_short = dist < 250.0

    # Short routes
    m_slot[is_short & (slot == "DEP_01")] = 1.18
    m_slot[is_short & (slot == "DEP_02")] = 0.88
    m_slot[is_short & (slot == "DEP_03")] = 1.22
    m_slot[is_short & (slot == "DEP_04")] = 0.72

    # Long routes
    m_slot[~is_short & (slot == "DEP_01")] = 1.00
    m_slot[~is_short & (slot == "DEP_02")] = 0.78
    m_slot[~is_short & (slot == "DEP_03")] = 1.02
    m_slot[~is_short & (slot == "DEP_04")] = 1.28

    # 3. Day of Week & Directional Asymmetry
    m_dow = np.ones(n, dtype=float)
    is_evening_or_night = np.isin(slot, ["DEP_03", "DEP_04"])
    is_morning = slot == "DEP_01"

    # Friday (dow == 4)
    fri = dow == 4
    m_dow[fri & is_outbound_hub & is_evening_or_night] = 1.35
    m_dow[fri & is_outbound_hub & ~is_evening_or_night] = 1.15
    m_dow[fri & is_inbound_hub] = 0.95
    m_dow[fri & is_inter_hub & is_evening_or_night] = 1.28
    m_dow[fri & is_inter_hub & ~is_evening_or_night] = 1.12
    m_dow[fri & ~is_outbound_hub & ~is_inbound_hub & ~is_inter_hub] = 1.10

    # Saturday (dow == 5)
    sat = dow == 5
    m_dow[sat & is_outbound_hub & is_morning] = 1.16
    m_dow[sat & is_outbound_hub & ~is_morning] = 1.02
    m_dow[sat & is_inbound_hub] = 0.95
    m_dow[sat & is_inter_hub] = 1.05
    m_dow[sat & ~is_outbound_hub & ~is_inbound_hub & ~is_inter_hub] = 1.05

    # Sunday (dow == 6)
    sun = dow == 6
    m_dow[sun & is_inbound_hub & is_evening_or_night] = 1.38
    m_dow[sun & is_inbound_hub & ~is_evening_or_night] = 1.12
    m_dow[sun & is_outbound_hub] = 0.82
    m_dow[sun & is_inter_hub & is_evening_or_night] = 1.28
    m_dow[sun & is_inter_hub & ~is_evening_or_night] = 1.08
    m_dow[sun & ~is_outbound_hub & ~is_inbound_hub & ~is_inter_hub] = 1.12

    # Monday (dow == 0)
    mon = dow == 0
    m_dow[mon & is_inbound_hub & is_morning] = 1.15
    m_dow[mon & is_inbound_hub & ~is_morning] = 0.95
    m_dow[mon & is_outbound_hub] = 0.90
    m_dow[mon & is_inter_hub] = 1.05
    m_dow[mon & ~is_outbound_hub & ~is_inbound_hub & ~is_inter_hub] = 0.95

    # Midweek (Tuesday=1, Wednesday=2)
    midweek = np.isin(dow, [1, 2])
    m_dow[midweek] = 0.90

    # Thursday (dow == 3)
    thu = dow == 3
    m_dow[thu] = 0.96

    # 4. Seasonal & Monthly Dynamics
    m_season = np.ones(n, dtype=float)
    is_kl_involved = (orig_state == "KL") | (dest_state == "KL")

    m_season[season == "Winter"] = 1.02
    m_season[season == "Summer"] = 1.08
    m_season[(season == "Monsoon") & is_kl_involved] = 0.88
    m_season[(season == "Monsoon") & ~is_kl_involved] = 0.94
    m_season[season == "Post-Monsoon"] = 1.08

    # Month start/end commercial bump
    is_month_boundary = df["is_month_start"].to_numpy() | df["is_month_end"].to_numpy()
    m_season[is_month_boundary] *= 1.03

    # 5. Holiday & Long-Weekend Effects
    is_nat = df["is_national_holiday"].to_numpy()
    is_long_we = df["is_long_weekend"].to_numpy()
    is_pre = df["is_pre_holiday"].to_numpy()
    is_post = df["is_post_holiday"].to_numpy()
    is_h_dest = df["is_holiday_destination"].to_numpy()
    is_h_orig = df["is_holiday_origin"].to_numpy()

    m_hol = np.ones(n, dtype=float)
    m_hol[is_nat & is_long_we] *= 1.20
    m_hol[is_nat & ~is_long_we] *= 1.08

    # Long weekend directional flow
    m_hol[is_long_we & is_outbound_hub & (dow == 4 | is_pre)] *= 1.25
    m_hol[is_long_we & is_inbound_hub & (np.isin(dow, [0, 6]) | is_post)] *= 1.28
    m_hol[is_long_we & ~is_outbound_hub & ~is_inbound_hub] *= 1.12

    m_hol[is_pre] *= 1.15
    m_hol[is_post] *= 1.12
    m_hol[is_h_dest] *= 1.15
    m_hol[is_h_orig] *= 1.06

    # Cap combined holiday multiplier to prevent runaway compounding
    m_hol = np.minimum(m_hol, 1.65)

    # 6. Vacation Effects
    is_sch_vac = (
        df["is_school_vacation_origin"].to_numpy()
        | df["is_school_vacation_destination"].to_numpy()
    )
    is_col_vac = (
        df["is_college_vacation_origin"].to_numpy()
        | df["is_college_vacation_destination"].to_numpy()
    )

    m_vac = np.ones(n, dtype=float)
    is_weekend_day = np.isin(dow, [4, 5, 6])
    m_vac[is_sch_vac & is_weekend_day] *= 1.08
    m_vac[is_sch_vac & ~is_weekend_day] *= 1.03
    m_vac[is_col_vac] *= 1.04

    # Combined expected demand
    expected_demand = base_rate * m_slot * m_dow * m_season * m_hol * m_vac
    return expected_demand


# ---------------------------------------------------------------------------
# Stochastic Demand Sampling
# ---------------------------------------------------------------------------

def sample_passenger_demand(
    expected_demand: np.ndarray,
    seed: int = DEFAULT_RANDOM_SEED,
    dispersion_alpha: float = 0.04,
) -> np.ndarray:
    """Sample discrete integer passenger counts using a Negative Binomial distribution.

    Variance relationship: Var(Y) = mu + alpha * mu^2.
    Number of successes parameter: n = 1 / alpha.
    Probability of success parameter: p = n / (n + mu).

    This captures natural intercity transit party-size clustering (solo, couples,
    small groups) and ensures non-negative integer passenger demand values.

    Args:
        expected_demand: 1D array of expected demand rates (mu).
        seed: Random seed for reproducibility.
        dispersion_alpha: Overdispersion parameter alpha (default 0.04 -> n=25).

    Returns:
        1D integer array of sampled passenger counts >= 0.
    """
    rng = np.random.default_rng(seed)
    n_param = 1.0 / dispersion_alpha
    p_param = n_param / (n_param + expected_demand)
    samples = rng.negative_binomial(n=n_param, p=p_param)
    return samples.astype(int)


# ---------------------------------------------------------------------------
# Operational Capacity Indicators
# ---------------------------------------------------------------------------

def calculate_operational_metrics(
    demand: np.ndarray,
    capacity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute post-demand operational evaluation indicators.

    Args:
        demand: Integer array of simulated passenger demand.
        capacity: Integer array of bus seat capacities.

    Returns:
        Tuple of (capacity_exceeded: bool array, estimated_unserved_demand: int array).
    """
    capacity_exceeded = demand > capacity
    estimated_unserved = np.maximum(0, demand - capacity)
    return capacity_exceeded, estimated_unserved


# ---------------------------------------------------------------------------
# End-to-End Generator & Validator
# ---------------------------------------------------------------------------

def generate_trip_demand(
    raw_dir: Path = RAW_DATA_DIR,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Execute the end-to-end synthetic trip demand generation pipeline.

    Args:
        raw_dir: Directory containing city, route, and calendar CSVs.
        seed: Seed for stochastic sampling.

    Returns:
        DataFrame with 87,680 rows and all 31 output columns.
    """
    cities_df, routes_df, calendar_df = load_input_datasets(raw_dir)

    # 1. Build scheduled trip grid
    grid = build_trip_grid(routes_df, calendar_df, DEPARTURE_SLOTS)

    # 2. Compute expected demand rates
    mu = compute_expected_demand(grid)

    # 3. Sample discrete passenger counts
    demand = sample_passenger_demand(mu, seed=seed)

    # 4. Compute operational capacity metrics
    cap = grid["bus_capacity"].to_numpy()
    cap_exceeded, unserved = calculate_operational_metrics(demand, cap)

    # 5. Attach target and operational columns
    grid["passenger_demand"] = demand
    grid["capacity_exceeded"] = cap_exceeded
    grid["estimated_unserved_demand"] = unserved

    # 6. Select and order canonical columns
    result_df = grid[OUTPUT_SCHEMA].copy()

    # 7. Validate dataset integrity
    validate_trip_demand(result_df)

    return result_df


def validate_trip_demand(df: pd.DataFrame) -> None:
    """Validate schema, row count, uniqueness, ranges, and target segregation.

    Args:
        df: Generated trip demand DataFrame.

    Raises:
        ValueError: If any validation rule fails.
    """
    # 1. Row count
    expected_rows = 87680
    if len(df) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} rows, found {len(df)}")

    # 2. Required columns & zero missing values
    missing_cols = set(OUTPUT_SCHEMA) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    null_counts = df[OUTPUT_SCHEMA].isna().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values detected in columns: {bad_cols}")

    # 3. Unique trip_id
    if not df["trip_id"].is_unique:
        num_dups = df["trip_id"].duplicated().sum()
        raise ValueError(f"Duplicate trip_ids detected: {num_dups}")

    # 4. Every route has 4 departures per day
    trips_per_route_date = df.groupby(["route_id", "departure_date"]).size()
    if not (trips_per_route_date == 4).all():
        raise ValueError("Not every route has exactly 4 departures per day.")

    # 5. Exactly 20 routes and 1,096 dates
    if df["route_id"].nunique() != 20:
        raise ValueError(f"Expected 20 unique routes, got {df['route_id'].nunique()}")
    if df["departure_date"].nunique() != 1096:
        raise ValueError(f"Expected 1,096 dates, got {df['departure_date'].nunique()}")

    # 6. Target bounds and types
    if (df["passenger_demand"] < 0).any():
        raise ValueError("Negative passenger demand values detected.")
    if not pd.api.types.is_integer_dtype(df["passenger_demand"]):
        raise ValueError("passenger_demand must be integer dtype.")

    # 7. Operational calculations
    expected_exceeded = df["passenger_demand"] > df["bus_capacity"]
    if not (df["capacity_exceeded"] == expected_exceeded).all():
        raise ValueError("capacity_exceeded does not match (passenger_demand > bus_capacity).")

    expected_unserved = np.maximum(0, df["passenger_demand"] - df["bus_capacity"])
    if not (df["estimated_unserved_demand"] == expected_unserved).all():
        raise ValueError("estimated_unserved_demand calculation mismatch.")

    # 8. Reasonable demand variation
    if df["passenger_demand"].std() < 5.0:
        raise ValueError("Simulated demand lacks realistic variance.")


def save_trip_demand(
    df: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_DIR / "trip_demand.csv",
) -> None:
    """Save validated trip demand dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# CLI Execution & Summary
# ---------------------------------------------------------------------------

def main() -> None:
    """Run generation pipeline, validate, persist, and print summary report."""
    output_path = PROCESSED_DATA_DIR / "trip_demand.csv"
    print("=" * 90)
    print("BUS DEMAND AI - STAGE 6: SYNTHETIC PASSENGER DEMAND SIMULATOR")
    print("=" * 90)
    print(f"Generating trip demand dataset (seed={DEFAULT_RANDOM_SEED})...")

    df = generate_trip_demand()
    save_trip_demand(df, output_path)

    print(f"\nSaved {len(df):,} trip records to: {output_path.resolve()}")
    print("\n--- Summary Statistics: passenger_demand ---")
    desc = df["passenger_demand"].describe()
    print(desc.to_string())

    cap_exceeded_pct = df["capacity_exceeded"].mean() * 100
    total_unserved = df["estimated_unserved_demand"].sum()
    print(f"\nBus Capacity: {DEFAULT_BUS_CAPACITY} seats")
    print(f"Capacity Exceeded Trips: {df['capacity_exceeded'].sum():,} ({cap_exceeded_pct:.2f}%)")
    print(f"Total Estimated Unserved Demand: {total_unserved:,} passengers")

    print("\n--- Demand by Departure Slot ---")
    slot_stats = df.groupby("departure_slot")["passenger_demand"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    print(slot_stats.to_string())

    print("\n--- Demand by Route (Top 5 & Bottom 5) ---")
    route_means = df.groupby("route_id")["passenger_demand"].mean().sort_values(ascending=False)
    print("Top 5 Corridors (Mean Demand):")
    print(route_means.head(5).to_string())
    print("\nBottom 5 Corridors (Mean Demand):")
    print(route_means.tail(5).to_string())

    print("\n--- Sample Records ---")
    sample_cols = [
        "trip_id",
        "departure_date",
        "departure_slot",
        "distance_km",
        "route_mobility_score",
        "passenger_demand",
        "capacity_exceeded",
        "estimated_unserved_demand",
    ]
    print(df[sample_cols].head(6).to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

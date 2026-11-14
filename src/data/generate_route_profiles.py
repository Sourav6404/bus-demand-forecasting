"""Directed Route Network Generator for BusDemand AI.

IMPORTANT: All route distances, travel durations, speeds, and demand
multipliers in this module are synthetic scenario modelling assumptions,
not official measurements from road authorities or transport operators.
They establish a realistic, controlled topology connecting the 8 primary
South Indian urban hubs for prototype demand simulation.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.config import DEFAULT_BUS_CAPACITY, RAW_DATA_DIR

# ---------------------------------------------------------------------------
# Route Network Specifications (20 Directed Intercity Routes)
# ---------------------------------------------------------------------------
# 10 bidirectional corridors = 20 directed routes.
# Directional differences in route_demand_multiplier capture static baseline
# scenario adjustments (e.g., structural workforce and commercial exchange),
# distinct from dynamic day-of-week, seasonal, or festival surges.
BASE_ROUTE_DEFINITIONS: List[Dict[str, Any]] = [
    # 1 & 2: Bengaluru <-> Mysuru (Major high-frequency expressway corridor)
    {
        "route_id": "R_BLR_MYS",
        "origin_city_id": "BLR",
        "destination_city_id": "MYS",
        "distance_km": 145.0,
        "estimated_duration_hours": 3.0,
        "route_demand_multiplier": 1.10,
        "corridor_note": "Expressway weekend tourism and commuter corridor",
    },
    {
        "route_id": "R_MYS_BLR",
        "origin_city_id": "MYS",
        "destination_city_id": "BLR",
        "distance_km": 145.0,
        "estimated_duration_hours": 3.0,
        "route_demand_multiplier": 1.08,
        "corridor_note": "Commuter and student return flow to state capital",
    },

    # 3 & 4: Bengaluru <-> Chennai (Interstate metropolitan economic spine)
    {
        "route_id": "R_BLR_MAA",
        "origin_city_id": "BLR",
        "destination_city_id": "MAA",
        "distance_km": 345.0,
        "estimated_duration_hours": 6.5,
        "route_demand_multiplier": 1.25,
        "corridor_note": "Metropolitan IT and manufacturing business corridor",
    },
    {
        "route_id": "R_MAA_BLR",
        "origin_city_id": "MAA",
        "destination_city_id": "BLR",
        "distance_km": 345.0,
        "estimated_duration_hours": 6.5,
        "route_demand_multiplier": 1.22,
        "corridor_note": "Metropolitan commercial and workforce reverse flow",
    },

    # 5 & 6: Bengaluru <-> Kozhikode (Karnataka to Malabar interstate gateway)
    {
        "route_id": "R_BLR_CCJ",
        "origin_city_id": "BLR",
        "destination_city_id": "CCJ",
        "distance_km": 360.0,
        "estimated_duration_hours": 8.0,
        "route_demand_multiplier": 1.15,
        "corridor_note": "Ghat corridor connecting Malabar diaspora and students",
    },
    {
        "route_id": "R_CCJ_BLR",
        "origin_city_id": "CCJ",
        "destination_city_id": "BLR",
        "distance_km": 360.0,
        "estimated_duration_hours": 8.0,
        "route_demand_multiplier": 1.12,
        "corridor_note": "Malabar return travel to Bengaluru employment market",
    },

    # 7 & 8: Bengaluru <-> Thrissur (Central Kerala transit connection)
    {
        "route_id": "R_BLR_TCR",
        "origin_city_id": "BLR",
        "destination_city_id": "TCR",
        "distance_km": 470.0,
        "estimated_duration_hours": 9.5,
        "route_demand_multiplier": 1.12,
        "corridor_note": "Long-distance overnight cultural and commercial connection",
    },
    {
        "route_id": "R_TCR_BLR",
        "origin_city_id": "TCR",
        "destination_city_id": "BLR",
        "distance_km": 470.0,
        "estimated_duration_hours": 9.5,
        "route_demand_multiplier": 1.10,
        "corridor_note": "Central Kerala workforce return flow",
    },

    # 9 & 10: Bengaluru <-> Kochi (Flagship long-distance overnight corridor)
    {
        "route_id": "R_BLR_COK",
        "origin_city_id": "BLR",
        "destination_city_id": "COK",
        "distance_km": 540.0,
        "estimated_duration_hours": 11.0,
        "route_demand_multiplier": 1.28,
        "corridor_note": "High-volume tech workforce and tourism sleeper route",
    },
    {
        "route_id": "R_COK_BLR",
        "origin_city_id": "COK",
        "destination_city_id": "BLR",
        "distance_km": 540.0,
        "estimated_duration_hours": 11.0,
        "route_demand_multiplier": 1.25,
        "corridor_note": "Kochi port and IT diaspora return flow to Bengaluru",
    },

    # 11 & 12: Chennai <-> Coimbatore (Tamil Nadu east-west industrial trunk line)
    {
        "route_id": "R_MAA_CBE",
        "origin_city_id": "MAA",
        "destination_city_id": "CBE",
        "distance_km": 510.0,
        "estimated_duration_hours": 9.5,
        "route_demand_multiplier": 1.18,
        "corridor_note": "Industrial and educational connector between TN hubs",
    },
    {
        "route_id": "R_CBE_MAA",
        "origin_city_id": "CBE",
        "destination_city_id": "MAA",
        "distance_km": 510.0,
        "estimated_duration_hours": 9.5,
        "route_demand_multiplier": 1.15,
        "corridor_note": "Textile and engineering commercial travel to state capital",
    },

    # 13 & 14: Coimbatore <-> Kochi (Palakkad Gap interstate highway)
    {
        "route_id": "R_CBE_COK",
        "origin_city_id": "CBE",
        "destination_city_id": "COK",
        "distance_km": 190.0,
        "estimated_duration_hours": 4.5,
        "route_demand_multiplier": 1.05,
        "corridor_note": "Palakkad Gap regional trade and airport feeder route",
    },
    {
        "route_id": "R_COK_CBE",
        "origin_city_id": "COK",
        "destination_city_id": "CBE",
        "distance_km": 190.0,
        "estimated_duration_hours": 4.5,
        "route_demand_multiplier": 1.02,
        "corridor_note": "Regional commercial and industrial feeder route",
    },

    # 15 & 16: Kochi <-> Kozhikode (Kerala North-South coastal corridor)
    {
        "route_id": "R_COK_CCJ",
        "origin_city_id": "COK",
        "destination_city_id": "CCJ",
        "distance_km": 180.0,
        "estimated_duration_hours": 4.8,
        "route_demand_multiplier": 1.06,
        "corridor_note": "Coastal business, trade, and medical connection",
    },
    {
        "route_id": "R_CCJ_COK",
        "origin_city_id": "CCJ",
        "destination_city_id": "COK",
        "distance_km": 180.0,
        "estimated_duration_hours": 4.8,
        "route_demand_multiplier": 1.04,
        "corridor_note": "Malabar regional travel to commercial capital",
    },

    # 17 & 18: Kochi <-> Thiruvananthapuram (Kerala commercial to administrative capital)
    {
        "route_id": "R_COK_TVM",
        "origin_city_id": "COK",
        "destination_city_id": "TVM",
        "distance_km": 210.0,
        "estimated_duration_hours": 5.5,
        "route_demand_multiplier": 1.12,
        "corridor_note": "Commercial to administrative and Technopark connector",
    },
    {
        "route_id": "R_TVM_COK",
        "origin_city_id": "TVM",
        "destination_city_id": "COK",
        "distance_km": 210.0,
        "estimated_duration_hours": 5.5,
        "route_demand_multiplier": 1.10,
        "corridor_note": "Administrative capital to port city commercial return flow",
    },

    # 19 & 20: Kozhikode <-> Thrissur (Malabar to Central Kerala feeder route)
    {
        "route_id": "R_CCJ_TCR",
        "origin_city_id": "CCJ",
        "destination_city_id": "TCR",
        "distance_km": 115.0,
        "estimated_duration_hours": 3.0,
        "route_demand_multiplier": 0.95,
        "corridor_note": "Regional transit connecting Malabar and cultural capital",
    },
    {
        "route_id": "R_TCR_CCJ",
        "origin_city_id": "TCR",
        "destination_city_id": "CCJ",
        "distance_km": 115.0,
        "estimated_duration_hours": 3.0,
        "route_demand_multiplier": 0.92,
        "corridor_note": "Regional trade and education feeder route",
    },
]

REQUIRED_ROUTE_COLUMNS = [
    "route_id",
    "origin_city_id",
    "destination_city_id",
    "origin_city",
    "destination_city",
    "distance_km",
    "estimated_duration_hours",
    "average_speed_kmph",
    "bus_capacity",
    "origin_generation_index",
    "destination_attraction_index",
    "route_mobility_score",
    "route_demand_multiplier",
    "profile_type",
]


# ---------------------------------------------------------------------------
# Core Pipeline Functions
# ---------------------------------------------------------------------------
def load_city_profiles(
    city_profiles_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and validate existing city profiles from CSV.

    Args:
        city_profiles_path: Path to city_profiles.csv. If None, default RAW_DATA_DIR
            location is used.

    Returns:
        pd.DataFrame containing validated city profiles.

    Raises:
        FileNotFoundError: If the city profiles file does not exist.
        ValueError: If required columns or cities are missing.
    """
    if city_profiles_path is None:
        city_profiles_path = RAW_DATA_DIR / "city_profiles.csv"

    path = Path(city_profiles_path)
    if not path.exists():
        raise FileNotFoundError(
            f"City profiles file not found at: {path.resolve()}. "
            "Please run generate_city_profiles.py first."
        )

    df = pd.read_csv(path)
    required_cols = {
        "city_id",
        "city",
        "demand_generation_index",
        "demand_attraction_index",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"City profiles file is missing required columns: {sorted(missing)}"
        )

    return df


def calculate_route_features(
    routes_df: pd.DataFrame,
    city_profiles: pd.DataFrame,
    bus_capacity: int = DEFAULT_BUS_CAPACITY,
) -> pd.DataFrame:
    """Calculate and assign route-level mobility and operational features.

    Calculates:
        - average_speed_kmph = distance_km / estimated_duration_hours
        - origin_city: Name of the origin city
        - destination_city: Name of the destination city
        - origin_generation_index: DGI of the origin city
        - destination_attraction_index: DAI of the destination city
        - route_mobility_score: (origin_generation_index + destination_attraction_index) / 2
        - bus_capacity: Number of available seats per coach
        - profile_type: "synthetic_scenario_assumption"

    Args:
        routes_df: DataFrame with base route definitions.
        city_profiles: DataFrame containing city indices and metadata.
        bus_capacity: Standard bus passenger capacity (default: 45).

    Returns:
        pd.DataFrame with complete calculated route features.
    """
    result = routes_df.copy()

    # Create fast lookup maps from city profiles
    id_to_city = dict(zip(city_profiles["city_id"], city_profiles["city"]))
    id_to_dgi = dict(
        zip(city_profiles["city_id"], city_profiles["demand_generation_index"])
    )
    id_to_dai = dict(
        zip(city_profiles["city_id"], city_profiles["demand_attraction_index"])
    )

    # Map city names
    result["origin_city"] = result["origin_city_id"].map(id_to_city)
    result["destination_city"] = result["destination_city_id"].map(id_to_city)

    # Calculate average speed (km/h)
    result["average_speed_kmph"] = (
        result["distance_km"] / result["estimated_duration_hours"]
    ).round(2)

    # Standard coach capacity
    result["bus_capacity"] = int(bus_capacity)

    # Attach directional city mobility indices (exact from city profiles)
    result["origin_generation_index"] = result["origin_city_id"].map(id_to_dgi)
    result["destination_attraction_index"] = result["destination_city_id"].map(id_to_dai)

    # Combined route mobility score = Mean(Origin DGI, Destination DAI)
    result["route_mobility_score"] = (
        (result["origin_generation_index"] + result["destination_attraction_index"]) / 2.0
    ).round(4)

    # Metadata tag
    result["profile_type"] = "synthetic_scenario_assumption"

    # Reorder to target schema
    return result[REQUIRED_ROUTE_COLUMNS]


def validate_route_profiles(
    routes: pd.DataFrame,
    valid_city_ids: Optional[set] = None,
) -> None:
    """Validate route profile integrity, network topology, and numerical ranges.

    Args:
        routes: Route profile DataFrame to validate.
        valid_city_ids: Set of allowable city IDs. If None, derived from routes.

    Raises:
        ValueError: If duplicate route IDs, duplicate directed OD pairs,
            self-routes, missing values, or invalid numerical ranges are detected.
    """
    # 1. Required columns presence
    missing_cols = set(REQUIRED_ROUTE_COLUMNS).difference(routes.columns)
    if missing_cols:
        raise ValueError(f"Missing required route columns: {sorted(missing_cols)}")

    # 2. Exactly 20 route records
    if len(routes) != 20:
        raise ValueError(f"Expected exactly 20 routes, but found {len(routes)}.")

    # 3. Unique Route IDs
    if routes["route_id"].duplicated().any():
        dups = routes.loc[routes["route_id"].duplicated(), "route_id"].tolist()
        raise ValueError(f"Duplicate route IDs detected: {dups}")

    # 4. Unique directed (Origin, Destination) pairs
    od_pairs = list(zip(routes["origin_city_id"], routes["destination_city_id"]))
    if len(od_pairs) != len(set(od_pairs)):
        seen = set()
        duplicate_od = []
        for pair in od_pairs:
            if pair in seen:
                duplicate_od.append(f"{pair[0]}->{pair[1]}")
            seen.add(pair)
        raise ValueError(
            f"Duplicate directed origin-destination pairs detected: {duplicate_od}"
        )

    # 5. No self-routes (origin == destination)
    self_routes = routes[routes["origin_city_id"] == routes["destination_city_id"]]
    if not self_routes.empty:
        invalid_ids = self_routes["route_id"].tolist()
        raise ValueError(f"Self-routes detected (origin == destination): {invalid_ids}")

    # 6. Valid city ID references
    if valid_city_ids is not None:
        invalid_origins = set(routes["origin_city_id"]) - valid_city_ids
        invalid_destinations = set(routes["destination_city_id"]) - valid_city_ids
        if invalid_origins or invalid_destinations:
            raise ValueError(
                f"Unknown city IDs in routes. Origin: {invalid_origins}, "
                f"Destination: {invalid_destinations}"
            )

    # 7. No missing values in required fields
    if routes[REQUIRED_ROUTE_COLUMNS].isna().any().any():
        null_counts = routes[REQUIRED_ROUTE_COLUMNS].isna().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Route profiles contain missing values: {null_cols}")

    # 8. Positive numerical constraints
    if (routes["distance_km"] <= 0).any():
        raise ValueError("All route distances (distance_km) must be strictly positive.")

    if (routes["estimated_duration_hours"] <= 0).any():
        raise ValueError(
            "All route journey durations (estimated_duration_hours) must be strictly positive."
        )

    if (routes["average_speed_kmph"] <= 0).any():
        raise ValueError("Calculated average speed must be strictly positive.")

    if (routes["bus_capacity"] <= 0).any():
        raise ValueError("Bus capacity must be a positive integer.")

    if (routes["route_demand_multiplier"] <= 0).any():
        raise ValueError("Route demand multiplier must be strictly positive.")

    if not routes["route_mobility_score"].between(0.0, 1.0).all():
        raise ValueError("Route mobility score must be bounded between 0.0 and 1.0.")


def generate_route_profiles(
    city_profiles_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate, calculate, validate, and optionally persist the 20-route network.

    Args:
        city_profiles_path: Path to city profiles CSV. If None, default RAW_DATA_DIR.
        output_path: Destination path for output CSV. If provided, saves to disk.

    Returns:
        pd.DataFrame containing the 20 validated directed routes.
    """
    city_profiles = load_city_profiles(city_profiles_path)
    valid_cities = set(city_profiles["city_id"].unique())

    raw_routes = pd.DataFrame(BASE_ROUTE_DEFINITIONS)
    routes = calculate_route_features(raw_routes, city_profiles)

    validate_route_profiles(routes, valid_city_ids=valid_cities)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        routes.to_csv(out_p, index=False)

    return routes


def main() -> None:
    """Execute route network generation CLI and write to data/raw/route_profiles.csv."""
    output_path = RAW_DATA_DIR / "route_profiles.csv"
    routes = generate_route_profiles(output_path=output_path)

    print(f"\nCreated {len(routes)} directed route profiles: {output_path.resolve()}")
    print("=" * 105)
    print("BUS DEMAND AI - 20 DIRECTED INTERCITY ROUTE NETWORK")
    print("=" * 105)
    display_cols = [
        "route_id",
        "origin_city",
        "destination_city",
        "distance_km",
        "estimated_duration_hours",
        "average_speed_kmph",
        "route_mobility_score",
        "route_demand_multiplier",
    ]
    print(routes[display_cols].to_string(index=False))
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()

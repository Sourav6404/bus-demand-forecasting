"""Synthetic City Profile Generator for BusDemand AI.

This module generates standardized socioeconomic, mobility, and travel propensity
profiles for the 8 primary South Indian urban hubs in the simulation corridor:
Bengaluru, Mysuru, Chennai, Coimbatore, Kochi, Kozhikode, Thrissur, and Thiruvananthapuram.

Each profile quantifies urban scale, economic drivers, education, tourism, and
migration dynamics, which subsequently calibrate the Gravity Model for passenger
demand generation and attraction.
"""

from dataclasses import asdict
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.config import (
    CITY_ID_TO_NAME,
    DEFAULT_RANDOM_SEED,
    RAW_DATA_DIR,
    TARGET_CITIES,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generate_city_profiles")

# Output path for city profiles
DEFAULT_OUTPUT_PATH = RAW_DATA_DIR / "city_profiles.csv"

# ---------------------------------------------------------------------------
# Empirical Socioeconomic Baseline Data
# ---------------------------------------------------------------------------
# Scores are normalized on a 0.0 - 10.0 scale based on census benchmarks,
# metropolitan economic outputs, university enrollments, and tourism footfall.
RAW_CITY_BENCHMARKS: List[Dict] = [
    {
        "city_id": "BLR",
        "city_name": "Bengaluru",
        "state": "Karnataka",
        "tier": 1,
        "population": 13600000,
        "latitude": 12.9716,
        "longitude": 77.5946,
        "urban_scale_score": 9.6,
        "employment_score": 9.8,        # Major global tech & aerospace hub
        "education_score": 9.2,         # IISc, IIMB, NLSIU, numerous engineering institutions
        "tourism_score": 6.8,           # Transit hub, gardens, business/MICE tourism
        "economic_activity_score": 9.7, # High disposable income, fintech, venture capital
        "migration_score": 9.5,         # Heavy pan-India & regional white-collar inflow
    },
    {
        "city_id": "MYS",
        "city_name": "Mysuru",
        "state": "Karnataka",
        "tier": 2,
        "population": 1320000,
        "latitude": 12.2958,
        "longitude": 76.6394,
        "urban_scale_score": 6.2,
        "employment_score": 6.5,        # Manufacturing, IT satellite campus hub
        "education_score": 7.4,         # University of Mysore, premier research institutes
        "tourism_score": 9.3,           # Mysore Palace, Dasara, heritage gateway
        "economic_activity_score": 6.4, # Growing regional services, tourism economy
        "migration_score": 5.8,         # Moderate student & commuter inflow from surrounding districts
    },
    {
        "city_id": "MAA",
        "city_name": "Chennai",
        "state": "Tamil Nadu",
        "tier": 1,
        "population": 11500000,
        "latitude": 13.0827,
        "longitude": 80.2707,
        "urban_scale_score": 9.4,
        "employment_score": 9.3,        # Automotive capital, SaaS, manufacturing, port
        "education_score": 9.1,         # IIT Madras, Anna University, Madras Medical College
        "tourism_score": 7.5,           # Cultural heritage, medical tourism, Marina beach
        "economic_activity_score": 9.3, # Industrial GDP powerhouse, major sea/air transit hub
        "migration_score": 8.7,         # Significant industrial and healthcare migration inflow
    },
    {
        "city_id": "CJB",
        "city_name": "Coimbatore",
        "state": "Tamil Nadu",
        "tier": 2,
        "population": 2900000,
        "latitude": 11.0168,
        "longitude": 76.9558,
        "urban_scale_score": 7.3,
        "employment_score": 8.2,        # Textile capital, precision engineering, pump manufacturing
        "education_score": 8.5,         # Major higher education & engineering college cluster
        "tourism_score": 7.0,           # Gateway to Ooty/Nilgiris, Isha Yoga center
        "economic_activity_score": 7.9, # High entrepreneurial index, robust SME ecosystem
        "migration_score": 6.9,         # Industrial workforce inflow from South TN & Kerala
    },
    {
        "city_id": "COK",
        "city_name": "Kochi",
        "state": "Kerala",
        "tier": 2,
        "population": 2150000,
        "latitude": 9.9312,
        "longitude": 76.2673,
        "urban_scale_score": 7.1,
        "employment_score": 7.6,        # Commercial port, shipbuilding, Infopark IT hub
        "education_score": 7.8,         # CUSAT, prominent medical and naval training institutes
        "tourism_score": 9.1,           # Fort Kochi, backwaters, international gateway
        "economic_activity_score": 8.0, # Highest commercial GDP in Kerala, remittance influx
        "migration_score": 7.4,         # High NRI diaspora return flux, commercial transit
    },
    {
        "city_id": "CCJ",
        "city_name": "Kozhikode",
        "state": "Kerala",
        "tier": 2,
        "population": 2050000,
        "latitude": 11.2588,
        "longitude": 75.7804,
        "urban_scale_score": 6.8,
        "employment_score": 6.7,        # Malabar trade center, timber, retail & healthcare
        "education_score": 8.1,         # IIM Kozhikode, NIT Calicut, Calicut Medical College
        "tourism_score": 7.6,           # Culinary tourism, historic coast, Wayanad gateway
        "economic_activity_score": 7.1, # Robust merchant community, Middle East trade connections
        "migration_score": 6.6,         # Student and Gulf diaspora travel dynamics
    },
    {
        "city_id": "TCR",
        "city_name": "Thrissur",
        "state": "Kerala",
        "tier": 2,
        "population": 1050000,
        "latitude": 10.5276,
        "longitude": 76.2144,
        "urban_scale_score": 5.9,
        "employment_score": 6.2,        # Banking headquarters (South Indian Bank, CSB), gold retail
        "education_score": 7.5,         # Kerala Agricultural Univ, Kerala University of Health Sciences
        "tourism_score": 8.8,           # Cultural capital of Kerala, Thrissur Pooram, temple circuits
        "economic_activity_score": 6.7, # Private banking hub, jewelry trade, NRI investments
        "migration_score": 5.9,         # Festival-driven and cultural seasonal mobility
    },
    {
        "city_id": "TRV",
        "city_name": "Thiruvananthapuram",
        "state": "Kerala",
        "tier": 2,
        "population": 1720000,
        "latitude": 8.5241,
        "longitude": 76.9366,
        "urban_scale_score": 6.9,
        "employment_score": 7.4,        # State administrative capital, Technopark, VSSC/ISRO
        "education_score": 8.3,         # University of Kerala, IISER, College of Engineering Trivandrum
        "tourism_score": 8.4,           # Padmanabhaswamy Temple, Kovalam beach, Varkala gateway
        "economic_activity_score": 7.3, # Public sector employment, IT services, aerospace research
        "migration_score": 6.7,         # Civil service, IT workforce, and educational migration
    },
]


# ---------------------------------------------------------------------------
# Derived Index Calculation Formulations
# ---------------------------------------------------------------------------
def calculate_demand_generation_index(row: pd.Series) -> float:
    """Calculate the Demand Generation Index (DGI) for an origin city.

    Quantifies the propensity of the city's populace to originate outbound
    intercity bus trips. High scores correlate with large urban scale,
    active workforce, economic means to travel, inward migrant population
    making frequent return visits home, and student mobility.

    Weights:
        - Urban Scale: 28%
        - Employment: 24%
        - Economic Activity: 20%
        - Migration Inflow/Flux: 18%
        - Education: 10%
    """
    dgi = (
        0.28 * row["urban_scale_score"]
        + 0.24 * row["employment_score"]
        + 0.20 * row["economic_activity_score"]
        + 0.18 * row["migration_score"]
        + 0.10 * row["education_score"]
    )
    return round(float(np.clip(dgi, 0.0, 10.0)), 2)


def calculate_demand_attraction_index(row: pd.Series) -> float:
    """Calculate the Demand Attraction Index (DAI) for a destination city.

    Quantifies the draw of the city for inbound travelers.
    High scores correlate with concentrated employment opportunities,
    commercial markets, major tourist/pilgrimage destinations, premier
    educational and healthcare campuses, and metropolitan amenities.

    Weights:
        - Employment Opportunities: 25%
        - Economic & Commercial Hub: 25%
        - Tourism, Heritage & Leisure: 22%
        - Higher Education & Healthcare: 18%
        - Urban Scale & Infrastructure: 10%
    """
    dai = (
        0.25 * row["employment_score"]
        + 0.25 * row["economic_activity_score"]
        + 0.22 * row["tourism_score"]
        + 0.18 * row["education_score"]
        + 0.10 * row["urban_scale_score"]
    )
    return round(float(np.clip(dai, 0.0, 10.0)), 2)


def calculate_city_mobility_index(row: pd.Series) -> float:
    """Calculate the aggregate City Mobility Index (CMI).

    Measures overall intercity mobility footprint, balancing generation
    and attraction potential, with a modest Tier-1 agglomeration adjustment.

    Formulation:
        CMI = Mean(DGI, DAI) * (1 + 0.05 * (Tier == 1))
    """
    dgi = row["demand_generation_index"]
    dai = row["demand_attraction_index"]
    tier_bonus = 1.05 if row["tier"] == 1 else 1.00
    cmi = ((dgi + dai) / 2.0) * tier_bonus
    return round(float(np.clip(cmi, 0.0, 10.0)), 2)


# ---------------------------------------------------------------------------
# Validation Engine
# ---------------------------------------------------------------------------
def validate_city_profiles(df: pd.DataFrame) -> None:
    """Validate city profile DataFrame integrity, schema, and statistical ranges.

    Raises:
        ValueError: If duplicate city IDs exist, expected cities are missing,
            null values are detected, or scores fall outside valid boundaries [0, 10].
    """
    expected_ids = set(CITY_ID_TO_NAME.keys())
    present_ids = set(df["city_id"].unique())

    # 1. Unique City IDs check
    if len(df["city_id"]) != len(present_ids):
        duplicates = df[df["city_id"].duplicated()]["city_id"].tolist()
        raise ValueError(f"Duplicate city IDs detected: {duplicates}")

    if present_ids != expected_ids:
        missing = expected_ids - present_ids
        unexpected = present_ids - expected_ids
        raise ValueError(
            f"City ID mismatch. Missing: {missing}, Unexpected: {unexpected}"
        )

    # 2. Missing Values check
    null_counts = df.isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Missing values found in city profiles: {null_cols}")

    # 3. Score Range Checks [0.0, 10.0]
    score_columns = [
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

    for col in score_columns:
        if col not in df.columns:
            raise ValueError(f"Required score column '{col}' missing from DataFrame.")
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val < 0.0 or max_val > 10.0:
            raise ValueError(
                f"Column '{col}' has scores outside valid range [0.0, 10.0]: "
                f"min={min_val}, max={max_val}"
            )

    # 4. Logical Hierarchical Consistency
    # Tier 1 cities must have urban_scale_score > Tier 2 cities minimum
    tier1_scale_min = df[df["tier"] == 1]["urban_scale_score"].min()
    tier2_scale_max = df[df["tier"] == 2]["urban_scale_score"].max()
    if tier1_scale_min <= tier2_scale_max:
        logger.warning(
            f"Tier 1 minimum urban scale ({tier1_scale_min}) is close to or below "
            f"Tier 2 maximum ({tier2_scale_max}). Verify tier classifications."
        )

    logger.info("City profiles validation passed successfully: 8/8 cities verified.")


# ---------------------------------------------------------------------------
# Generation Pipeline
# ---------------------------------------------------------------------------
def generate_city_profiles(
    output_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate, validate, and persist standardized city profiles.

    Args:
        output_path: Destination path for CSV output. If None, file is not saved.
        random_seed: Fixed random seed for reproducibility.

    Returns:
        pd.DataFrame containing the validated city profiles.
    """
    set_seed(random_seed)
    logger.info("Initializing city profile generation for 8 South Indian hubs...")

    # Load baseline benchmarks into DataFrame
    df = pd.DataFrame(RAW_CITY_BENCHMARKS)

    # Calculate derived mobility indices
    df["demand_generation_index"] = df.apply(calculate_demand_generation_index, axis=1)
    df["demand_attraction_index"] = df.apply(calculate_demand_attraction_index, axis=1)
    df["city_mobility_index"] = df.apply(calculate_city_mobility_index, axis=1)

    # Sort deterministically by tier ascending, then city_id ascending
    df = df.sort_values(by=["tier", "city_id"]).reset_index(drop=True)

    # Validate generated profiles
    validate_city_profiles(df)

    # Persist to disk if output path is provided
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info(f"City profiles successfully saved to: {out_p.resolve()}")

    return df


if __name__ == "__main__":
    profiles_df = generate_city_profiles()
    print("\n" + "=" * 95)
    print("BUS DEMAND AI - GENERATED CITY PROFILES (SOUTH INDIAN CORRIDOR)")
    print("=" * 95)
    display_cols = [
        "city_id",
        "city_name",
        "state",
        "tier",
        "population",
        "demand_generation_index",
        "demand_attraction_index",
        "city_mobility_index",
    ]
    print(profiles_df[display_cols].to_string(index=False))
    print("=" * 95 + "\n")

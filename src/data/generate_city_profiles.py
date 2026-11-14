"""Create scenario-based city profiles for the BusDemand AI project.

IMPORTANT: Scores in this file are synthetic modelling assumptions, not
official measurements or claims about real-world city populations/economies.
They provide controlled variation for a prototype demand simulator.
"""

from pathlib import Path
from typing import Any

import pandas as pd


# All component scores use a 0.0–1.0 scale.
# The values are deliberately illustrative and must be documented as such.
CITY_SCENARIOS: list[dict[str, Any]] = [
    {
        "city_id": "BLR", "city": "Bengaluru", "state": "Karnataka",
        "urban_scale_score": 0.98, "employment_score": 1.00,
        "education_score": 0.95, "tourism_score": 0.55,
        "economic_activity_score": 1.00, "migration_score": 0.95,
        "profile_note": "Large employment and education hub; scenario assumption",
    },
    {
        "city_id": "MYS", "city": "Mysuru", "state": "Karnataka",
        "urban_scale_score": 0.55, "employment_score": 0.50,
        "education_score": 0.65, "tourism_score": 0.90,
        "economic_activity_score": 0.55, "migration_score": 0.40,
        "profile_note": "Tourism and education influence; scenario assumption",
    },
    {
        "city_id": "COK", "city": "Kochi", "state": "Kerala",
        "urban_scale_score": 0.78, "employment_score": 0.78,
        "education_score": 0.72, "tourism_score": 0.92,
        "economic_activity_score": 0.85, "migration_score": 0.72,
        "profile_note": "Commercial and tourism hub; scenario assumption",
    },
    {
        "city_id": "CCJ", "city": "Kozhikode", "state": "Kerala",
        "urban_scale_score": 0.62, "employment_score": 0.62,
        "education_score": 0.72, "tourism_score": 0.65,
        "economic_activity_score": 0.62, "migration_score": 0.78,
        "profile_note": "Education and migration influence; scenario assumption",
    },
    {
        "city_id": "TCR", "city": "Thrissur", "state": "Kerala",
        "urban_scale_score": 0.58, "employment_score": 0.58,
        "education_score": 0.68, "tourism_score": 0.62,
        "economic_activity_score": 0.68, "migration_score": 0.60,
        "profile_note": "Regional commercial and education influence; scenario assumption",
    },
    {
        "city_id": "TVM", "city": "Thiruvananthapuram", "state": "Kerala",
        "urban_scale_score": 0.70, "employment_score": 0.82,
        "education_score": 0.82, "tourism_score": 0.68,
        "economic_activity_score": 0.78, "migration_score": 0.58,
        "profile_note": "Administrative, employment and education hub; scenario assumption",
    },
    {
        "city_id": "MAA", "city": "Chennai", "state": "Tamil Nadu",
        "urban_scale_score": 0.98, "employment_score": 0.98,
        "education_score": 0.92, "tourism_score": 0.58,
        "economic_activity_score": 0.98, "migration_score": 0.90,
        "profile_note": "Large employment, industry and education hub; scenario assumption",
    },
    {
        "city_id": "CBE", "city": "Coimbatore", "state": "Tamil Nadu",
        "urban_scale_score": 0.72, "employment_score": 0.82,
        "education_score": 0.78, "tourism_score": 0.48,
        "economic_activity_score": 0.85, "migration_score": 0.68,
        "profile_note": "Industry, employment and education influence; scenario assumption",
    },
]

COMPONENT_COLUMNS = [
    "urban_scale_score",
    "employment_score",
    "education_score",
    "tourism_score",
    "economic_activity_score",
    "migration_score",
]


def add_mobility_indices(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add directional city indices using documented illustrative weights.

    Generation places more weight on employment, education and migration.
    Attraction places more weight on employment, tourism and economic activity.
    These weights are design assumptions and should be sensitivity-tested later.
    """
    result = profiles.copy()

    generation_weights = {
        "urban_scale_score": 0.10,
        "employment_score": 0.25,
        "education_score": 0.20,
        "tourism_score": 0.05,
        "economic_activity_score": 0.15,
        "migration_score": 0.25,
    }
    attraction_weights = {
        "urban_scale_score": 0.10,
        "employment_score": 0.20,
        "education_score": 0.10,
        "tourism_score": 0.25,
        "economic_activity_score": 0.25,
        "migration_score": 0.10,
    }

    result["demand_generation_index"] = sum(
        result[column] * weight for column, weight in generation_weights.items()
    )
    result["demand_attraction_index"] = sum(
        result[column] * weight for column, weight in attraction_weights.items()
    )

    # A combined summary score; directional indices remain the more useful fields.
    result["city_mobility_index"] = (
        result["demand_generation_index"] + result["demand_attraction_index"]
    ) / 2
    return result


def validate_city_profiles(profiles: pd.DataFrame) -> None:
    """Fail early if IDs, fields or score ranges are invalid."""
    required = {
        "city_id", "city", "state", *COMPONENT_COLUMNS,
        "demand_generation_index", "demand_attraction_index",
        "city_mobility_index",
    }
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"Missing required city profile columns: {sorted(missing)}")

    if profiles["city_id"].duplicated().any():
        duplicates = profiles.loc[profiles["city_id"].duplicated(), "city_id"].tolist()
        raise ValueError(f"Duplicate city IDs found: {duplicates}")

    score_columns = COMPONENT_COLUMNS + [
        "demand_generation_index", "demand_attraction_index", "city_mobility_index"
    ]
    if profiles[score_columns].isna().any().any():
        raise ValueError("City profile scores must not contain missing values")

    invalid = ~profiles[score_columns].apply(lambda col: col.between(0, 1)).all()
    if invalid.any():
        raise ValueError(f"Scores outside the 0–1 range: {invalid[invalid].index.tolist()}")


def build_city_profiles() -> pd.DataFrame:
    """Build, score and validate the city profile table."""
    profiles = pd.DataFrame(CITY_SCENARIOS)
    profiles["profile_type"] = "synthetic_scenario_assumption"
    profiles["profile_period"] = "scenario_2026"
    profiles = add_mobility_indices(profiles)
    validate_city_profiles(profiles)
    return profiles


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "raw" / "city_profiles.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = build_city_profiles()
    profiles.to_csv(output_path, index=False)

    print(f"Created {len(profiles)} city profiles: {output_path}")
    print(
        profiles[[
            "city_id", "city", "demand_generation_index",
            "demand_attraction_index", "city_mobility_index",
        ]].round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()

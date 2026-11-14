"""Project configuration, directory paths, and reproducibility settings for BusDemand AI.

This module centralizes all simulation parameters, geographical scopes,
file paths, and deterministic seed definitions across the project.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import random
from typing import Dict, List
import numpy as np

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GENERATED_DATA_DIR = DATA_DIR / "generated"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Reproducibility Settings
# ---------------------------------------------------------------------------
DEFAULT_RANDOM_SEED: int = 42


def set_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Set global seeds across Python standard library and NumPy for reproducibility.

    Args:
        seed: The integer seed to set. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Simulation Scope & Calendar
# ---------------------------------------------------------------------------
SIMULATION_YEARS: int = 3
SIMULATION_START_DATE: str = "2024-01-01"
SIMULATION_END_DATE: str = "2026-12-31"  # 3 full years = 1,096 days

# 4 departures per route per day
DEPARTURES_PER_ROUTE_DAY: int = 4
DEPARTURE_SLOTS: List[Dict[str, str]] = [
    {"slot_id": "DEP_01", "name": "Early Morning", "time": "06:00", "window": "05:30-08:00"},
    {"slot_id": "DEP_02", "name": "Midday",        "time": "11:30", "window": "11:00-13:30"},
    {"slot_id": "DEP_03", "name": "Evening",       "time": "17:00", "window": "16:30-18:30"},
    {"slot_id": "DEP_04", "name": "Night Sleeper", "time": "22:30", "window": "21:30-23:59"},
]

# Standard bus fleet capacity baseline
DEFAULT_BUS_CAPACITY: int = 45  # Standard 2+2 AC Seater/Sleeper intercity coach

# ---------------------------------------------------------------------------
# Initial Geography: 8 South Indian Urban Hubs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CityConfig:
    city_id: str
    city_name: str
    state: str
    tier: int
    latitude: float
    longitude: float

    @property
    def city(self) -> str:
        return self.city_name


TARGET_CITIES: List[CityConfig] = [
    CityConfig("BLR", "Bengaluru", "Karnataka", 1, 12.9716, 77.5946),
    CityConfig("MYS", "Mysuru", "Karnataka", 2, 12.2958, 76.6394),
    CityConfig("MAA", "Chennai", "Tamil Nadu", 1, 13.0827, 80.2707),
    CityConfig("CBE", "Coimbatore", "Tamil Nadu", 2, 11.0168, 76.9558),
    CityConfig("COK", "Kochi", "Kerala", 2, 9.9312, 76.2673),
    CityConfig("CCJ", "Kozhikode", "Kerala", 2, 11.2588, 75.7804),
    CityConfig("TCR", "Thrissur", "Kerala", 2, 10.5276, 76.2144),
    CityConfig("TVM", "Thiruvananthapuram", "Kerala", 2, 8.5241, 76.9366),
]

CITY_ID_TO_NAME: Dict[str, str] = {c.city_id: c.city_name for c in TARGET_CITIES}
CITY_NAME_TO_ID: Dict[str, str] = {c.city_name: c.city_id for c in TARGET_CITIES}


# ---------------------------------------------------------------------------
# Project Configuration Dataclass
# ---------------------------------------------------------------------------
@dataclass
class ProjectConfig:
    random_seed: int = DEFAULT_RANDOM_SEED
    simulation_years: int = SIMULATION_YEARS
    simulation_start_date: str = SIMULATION_START_DATE
    simulation_end_date: str = SIMULATION_END_DATE
    departures_per_route_day: int = DEPARTURES_PER_ROUTE_DAY
    bus_capacity: int = DEFAULT_BUS_CAPACITY
    cities: List[CityConfig] = field(default_factory=lambda: list(TARGET_CITIES))
    raw_data_dir: Path = RAW_DATA_DIR
    generated_data_dir: Path = GENERATED_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    artifacts_dir: Path = ARTIFACTS_DIR
    reports_dir: Path = REPORTS_DIR


# Instantiate default active configuration
active_config = ProjectConfig()

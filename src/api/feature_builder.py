"""Feature Builder for BusDemand AI Inference Pipeline.

Loads immutable raw reference datasets and constructs the exact 25-feature input row
required by the trained machine learning models, ensuring zero data leakage and
consistent encoding of spatio-temporal and calendar dynamics.
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.config import (
    DEFAULT_BUS_CAPACITY,
    DEPARTURE_SLOTS,
    FUTURE_CALENDAR_PATH,
    FUTURE_DATA_DIR,
    OPERATIONAL_HORIZON_END_DATE,
    RAW_DATA_DIR,
    SIMULATION_END_DATE,
    SIMULATION_START_DATE,
    TARGET_CITIES,
    get_current_server_time,
    get_operating_timezone,
)
from src.models.preprocessing import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
)


class RouteNotFoundError(ValueError):
    """Raised when the requested corridor does not exist in the route catalog."""
    pass


class DateOutOfRangeError(ValueError):
    """Raised when the requested departure date is outside the supported calendar range."""
    pass


class UnsupportedDepartureTimeError(ValueError):
    """Raised when the requested departure time does not match a scheduled slot."""
    pass


class OperationalCutoffError(ValueError):
    """Raised when an operational departure datetime is in the past or less than 24 hours ahead."""
    pass


class FeatureBuilder:
    """Constructs model-compatible feature matrices from raw and future reference datasets."""

    def __init__(
        self,
        raw_data_dir: Path = RAW_DATA_DIR,
        future_data_dir: Path = FUTURE_DATA_DIR,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.future_data_dir = Path(future_data_dir)
        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Load and index route profiles, city profiles, and separate calendar features."""
        route_path = self.raw_data_dir / "route_profiles.csv"
        city_path = self.raw_data_dir / "city_profiles.csv"
        hist_cal_path = self.raw_data_dir / "calendar_features.csv"
        future_cal_path = self.future_data_dir / "2027_calendar_features.csv"

        for p in [route_path, city_path, hist_cal_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Reference dataset not found at {p.resolve()}. "
                    "Ensure Stage 3-5 data generation scripts have been run."
                )

        self.routes_df = pd.read_csv(route_path)
        self.cities_df = pd.read_csv(city_path)
        self.hist_calendar_df = pd.read_csv(hist_cal_path)

        # Index routes by route_id
        self.routes_by_id: Dict[str, Dict[str, Any]] = {
            r["route_id"]: r.to_dict() for _, r in self.routes_df.iterrows()
        }

        # Index historical calendar rows by ISO date string (YYYY-MM-DD)
        self.historical_calendar: Dict[str, Dict[str, Any]] = {
            str(r["date"]): r.to_dict() for _, r in self.hist_calendar_df.iterrows()
        }

        # Index future calendar rows (2027) if present
        self.future_calendar: Dict[str, Dict[str, Any]] = {}
        if future_cal_path.exists():
            self.future_calendar_df = pd.read_csv(future_cal_path)
            self.future_calendar = {
                str(r["date"]): r.to_dict() for _, r in self.future_calendar_df.iterrows()
            }
        else:
            self.future_calendar_df = pd.DataFrame()

        # Merged lookup for fast date membership validation
        self.calendar_by_date: Dict[str, Dict[str, Any]] = {
            **self.historical_calendar,
            **self.future_calendar,
        }

        # Map city ID to 2-letter state code
        state_map = {"Karnataka": "KA", "Kerala": "KL", "Tamil Nadu": "TN"}
        self.city_to_state: Dict[str, str] = {
            c.city_id: state_map[c.state] for c in TARGET_CITIES
        }

        # Index departure slots
        self.time_to_slot: Dict[str, Dict[str, str]] = {}
        self.slot_id_to_slot: Dict[str, Dict[str, str]] = {}
        for s in DEPARTURE_SLOTS:
            self.time_to_slot[s["time"]] = s
            self.slot_id_to_slot[s["slot_id"]] = s

    def get_calendar_source(self, departure_date: str) -> str:
        """Identify which calendar partition powers the requested departure date.

        Returns:
            'historical' (2024-2026), 'future' (2027), or 'missing'.
        """
        if departure_date in self.historical_calendar:
            return "historical"
        if departure_date in self.future_calendar:
            return "future"
        return "missing"

    def get_supported_routes(self) -> List[Dict[str, Any]]:
        """Return list of all supported intercity corridors."""
        items = []
        for _, r in self.routes_df.iterrows():
            items.append({
                "route_id": r["route_id"],
                "origin_city_id": r["origin_city_id"],
                "destination_city_id": r["destination_city_id"],
                "origin_city": r["origin_city"],
                "destination_city": r["destination_city"],
                "distance_km": float(r["distance_km"]),
                "estimated_duration_hours": float(r["estimated_duration_hours"]),
                "bus_capacity": int(r.get("bus_capacity", DEFAULT_BUS_CAPACITY)),
                "route_mobility_score": float(r["route_mobility_score"]),
                "route_demand_multiplier": float(r["route_demand_multiplier"]),
            })
        return items

    def get_departure_slots(self) -> List[Dict[str, str]]:
        """Return list of supported scheduled departure slots."""
        return list(DEPARTURE_SLOTS)

    def resolve_slot(self, departure_time: str) -> Dict[str, str]:
        """Resolve a time string or slot identifier to its slot definition.

        Args:
            departure_time: Time string (e.g. '22:30') or slot ID (e.g. 'DEP_04').

        Returns:
            Dictionary with slot_id, time, name, window.

        Raises:
            UnsupportedDepartureTimeError: If the time does not match a scheduled slot.
        """
        cleaned = departure_time.strip()
        # Direct match on time
        if cleaned in self.time_to_slot:
            return self.time_to_slot[cleaned]
        # Direct match on slot_id
        if cleaned.upper() in self.slot_id_to_slot:
            return self.slot_id_to_slot[cleaned.upper()]

        supported_examples = [f"{s['time']} ({s['slot_id']} - {s['name']})" for s in DEPARTURE_SLOTS]
        raise UnsupportedDepartureTimeError(
            f"Departure time '{departure_time}' is not a scheduled slot. "
            f"Supported scheduled slots are: {', '.join(supported_examples)}."
        )

    def get_earliest_departure_info(
        self,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Calculate earliest operational departure datetime and minimum permitted calendar date.

        Args:
            reference_time: Optional datetime reference (defaults to current time in operating timezone).

        Returns:
            Dict containing server_time, earliest_departure_datetime, min_departure_date,
            max_departure_date, and timezone.
        """
        tz = get_operating_timezone()
        if reference_time is None:
            ref_dt = get_current_server_time()
        elif reference_time.tzinfo is None:
            ref_dt = reference_time.replace(tzinfo=tz)
        else:
            ref_dt = reference_time.astimezone(tz)

        earliest_dt = ref_dt + timedelta(hours=24)

        # Standard slots: 06:00, 11:30, 17:00, 22:30
        # If earliest_dt's time is strictly past 22:30, then no slot on earliest_dt's date
        # meets the 24-hour advance notice requirement, so the minimum date advances by 1 day.
        latest_slot_time = time(22, 30)
        if earliest_dt.time() > latest_slot_time:
            min_date = earliest_dt.date() + timedelta(days=1)
        else:
            min_date = earliest_dt.date()

        return {
            "server_time": ref_dt.isoformat(),
            "earliest_departure_datetime": earliest_dt.isoformat(),
            "min_departure_date": min_date.isoformat(),
            "max_departure_date": OPERATIONAL_HORIZON_END_DATE,
            "timezone": str(tz),
        }

    def validate_request_inputs(
        self,
        route_id: str,
        departure_date: str,
        departure_time: str,
        allow_historical: bool = False,
        reference_time: Optional[datetime] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        """Validate all prediction inputs against reference catalogs and operational constraints.

        Args:
            route_id: Directed corridor identifier.
            departure_date: Scheduled departure date in YYYY-MM-DD format.
            departure_time: Scheduled departure slot time or ID.
            allow_historical: If True, bypasses the operational 24-hour future advance constraint
                (used exclusively for historical model testing/evaluation).
            reference_time: Optional reference datetime for reproducible cutoff testing.

        Returns:
            Tuple of (route_meta, calendar_row, slot_meta).

        Raises:
            RouteNotFoundError: If the route ID does not exist in the catalog.
            DateOutOfRangeError: If the departure date is outside supported bounds (2024-01-01 to 2027-12-31).
            UnsupportedDepartureTimeError: If the time is not one of the 4 scheduled slots.
            OperationalCutoffError: If the departure is in the past or < 24h ahead (operational mode).
        """
        # 1. Validate route
        if route_id not in self.routes_by_id:
            valid_ids = sorted(list(self.routes_by_id.keys()))
            raise RouteNotFoundError(
                f"Route '{route_id}' was not found. "
                f"Supported route IDs: {', '.join(valid_ids)}."
            )
        route_meta = self.routes_by_id[route_id]

        # 2. Validate departure time slot first so we have the exact slot time
        slot_meta = self.resolve_slot(departure_time)
        slot_time = slot_meta["time"]

        # 3. Validate calendar date format and simulation bounds
        try:
            dep_date_obj = date.fromisoformat(departure_date.strip())
        except ValueError:
            raise ValueError(f"Invalid date format: '{departure_date}'. Expected YYYY-MM-DD.")

        min_sim_date = date.fromisoformat(SIMULATION_START_DATE)
        max_sim_date = date.fromisoformat(OPERATIONAL_HORIZON_END_DATE)

        if dep_date_obj > max_sim_date:
            raise DateOutOfRangeError(
                f"Departure date '{departure_date}' is outside the supported calendar range: "
                f"exceeds the operational horizon ({OPERATIONAL_HORIZON_END_DATE}). "
                "Forecasting beyond this range requires extending the future calendar reference dataset."
            )
        if dep_date_obj < min_sim_date:
            raise DateOutOfRangeError(
                f"Departure date '{departure_date}' is outside the supported calendar range: "
                f"before the earliest supported simulation date ({SIMULATION_START_DATE})."
            )

        if departure_date not in self.calendar_by_date:
            raise DateOutOfRangeError(
                f"Departure date '{departure_date}' is outside the supported calendar range "
                f"({SIMULATION_START_DATE} to {OPERATIONAL_HORIZON_END_DATE}). "
                "Date record is missing from both historical and future calendar reference datasets."
            )
        calendar_row = self.calendar_by_date[departure_date]

        # 4. Enforce operational 24-hour advance booking notice (unless allow_historical=True)
        if not allow_historical:
            tz = get_operating_timezone()
            slot_h, slot_m = [int(p) for p in slot_time.split(":")[:2]]
            departure_dt = datetime.combine(dep_date_obj, time(slot_h, slot_m), tzinfo=tz)

            if reference_time is None:
                ref_dt = get_current_server_time()
            elif reference_time.tzinfo is None:
                ref_dt = reference_time.replace(tzinfo=tz)
            else:
                ref_dt = reference_time.astimezone(tz)

            earliest_valid_dt = ref_dt + timedelta(hours=24)

            # Check if departure is in the past
            if departure_dt < ref_dt:
                raise OperationalCutoffError(
                    f"Departure datetime ({departure_dt.strftime('%Y-%m-%d %H:%M %Z')}) is in the past. "
                    f"Current server time is {ref_dt.strftime('%Y-%m-%d %H:%M %Z')}. "
                    "Operational forecasts require a departure datetime at least 24 hours in the future."
                )

            # Check if departure is less than 24 hours ahead
            if departure_dt < earliest_valid_dt:
                raise OperationalCutoffError(
                    f"Departure datetime ({departure_dt.strftime('%Y-%m-%d %H:%M %Z')}) is less than 24 hours ahead "
                    f"of current server time ({ref_dt.strftime('%Y-%m-%d %H:%M %Z')}). "
                    f"Operational forecasting requires a minimum 24-hour advance notice. "
                    f"Earliest permitted departure is {earliest_valid_dt.strftime('%Y-%m-%d %H:%M %Z')}."
                )

        return route_meta, calendar_row, slot_meta

    def calculate_cutoff(
        self,
        departure_date: str,
        departure_time: str,
    ) -> Tuple[str, str]:
        """Calculate the 24-hour prediction cutoff ISO timestamp.

        Args:
            departure_date: YYYY-MM-DD
            departure_time: HH:MM

        Returns:
            Tuple of (departure_datetime_iso, prediction_cutoff_iso).
        """
        dep_dt = datetime.fromisoformat(f"{departure_date}T{departure_time}:00")
        cutoff_dt = dep_dt - timedelta(hours=24)
        return dep_dt.isoformat(), cutoff_dt.isoformat()

    def build_feature_dataframe(
        self,
        route_id: str,
        departure_date: str,
        departure_time: str,
        allow_historical: bool = False,
        reference_time: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Build a single-row DataFrame containing the exact 25 model features.

        Args:
            route_id: Valid route ID (e.g. 'R_BLR_COK').
            departure_date: ISO date (e.g. '2026-12-20').
            departure_time: Scheduled slot time (e.g. '22:30').
            allow_historical: Flag to bypass operational 24h cutoff for historical model testing.
            reference_time: Optional reference datetime for reproducible cutoff testing.

        Returns:
            Tuple of (features_df, metadata_dict).
        """
        route_meta, cal_row, slot_meta = self.validate_request_inputs(
            route_id=route_id,
            departure_date=departure_date,
            departure_time=departure_time,
            allow_historical=allow_historical,
            reference_time=reference_time,
        )

        orig_city_id = route_meta["origin_city_id"]
        dest_city_id = route_meta["destination_city_id"]
        orig_state = self.city_to_state[orig_city_id]
        dest_state = self.city_to_state[dest_city_id]

        # Derive state-specific holiday and vacation indicators
        is_holiday_origin = bool(cal_row[f"is_holiday_{orig_state.lower()}"])
        is_holiday_destination = bool(cal_row[f"is_holiday_{dest_state.lower()}"])
        is_school_vacation_origin = bool(cal_row[f"is_school_vacation_{orig_state.lower()}"])
        is_school_vacation_destination = bool(cal_row[f"is_school_vacation_{dest_state.lower()}"])
        is_college_vacation_origin = bool(cal_row[f"is_college_vacation_{orig_state.lower()}"])
        is_college_vacation_destination = bool(cal_row[f"is_college_vacation_{dest_state.lower()}"])

        # Scheduled time & duration
        slot_time = slot_meta["time"]
        dep_hour = int(slot_time.split(":")[0])
        duration_minutes = int(round(float(route_meta["estimated_duration_hours"]) * 60))

        # Assemble exact 25-feature record in correct order
        feature_dict: Dict[str, Any] = {
            # Categorical features (5)
            "route_id": str(route_id),
            "origin_city_id": str(orig_city_id),
            "destination_city_id": str(dest_city_id),
            "departure_slot": str(slot_meta["slot_id"]),
            "season": str(cal_row["season"]),
            # Numerical features (20)
            "distance_km": float(route_meta["distance_km"]),
            "journey_duration_minutes": int(duration_minutes),
            "route_demand_multiplier": float(route_meta["route_demand_multiplier"]),
            "route_mobility_score": float(route_meta["route_mobility_score"]),
            "departure_hour": int(dep_hour),
            "day_of_week": int(cal_row["day_of_week"]),
            "is_weekend": bool(cal_row["is_weekend"]),
            "month": int(cal_row["month"]),
            "quarter": int(cal_row["quarter"]),
            "is_holiday_origin": bool(is_holiday_origin),
            "is_holiday_destination": bool(is_holiday_destination),
            "is_national_holiday": bool(cal_row["is_national_holiday"]),
            "is_pre_holiday": bool(cal_row["is_pre_holiday"]),
            "is_post_holiday": bool(cal_row["is_post_holiday"]),
            "is_long_weekend": bool(cal_row["is_long_weekend"]),
            "is_school_vacation_origin": bool(is_school_vacation_origin),
            "is_school_vacation_destination": bool(is_school_vacation_destination),
            "is_college_vacation_origin": bool(is_college_vacation_origin),
            "is_college_vacation_destination": bool(is_college_vacation_destination),
            "bus_capacity": int(route_meta.get("bus_capacity", DEFAULT_BUS_CAPACITY)),
        }

        # Build single-row DataFrame with strict column order
        features_df = pd.DataFrame([feature_dict])[MODEL_FEATURE_COLUMNS]

        # Calculate cutoff timestamps
        dep_dt_iso, cutoff_iso = self.calculate_cutoff(departure_date, slot_time)

        meta = {
            "route_id": route_id,
            "origin_city": route_meta["origin_city"],
            "destination_city": route_meta["destination_city"],
            "bus_capacity": int(route_meta.get("bus_capacity", DEFAULT_BUS_CAPACITY)),
            "departure_datetime": dep_dt_iso,
            "prediction_cutoff": cutoff_iso,
            "departure_slot": slot_meta["slot_id"],
            "slot_name": slot_meta["name"],
            "calendar_source": self.get_calendar_source(departure_date),
        }

        return features_df, meta

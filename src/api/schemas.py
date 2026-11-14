"""Pydantic request and response schemas for the BusDemand AI Inference API."""

from datetime import datetime, date, time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Passenger demand forecasting request schema."""

    route_id: str = Field(
        ...,
        description="Supported directed corridor identifier (e.g. 'R_BLR_COK').",
        examples=["R_BLR_COK"],
    )
    departure_date: str = Field(
        ...,
        description="Scheduled departure date in YYYY-MM-DD format (within 2024-01-01 to 2026-12-31).",
        examples=["2026-12-20"],
    )
    departure_time: str = Field(
        ...,
        description="Scheduled departure time matching a supported departure slot ('06:00', '11:30', '17:00', '22:30').",
        examples=["22:30"],
    )
    allow_historical: bool = Field(
        default=False,
        description="Internal testing flag to bypass the 24-hour advance booking notice for historical backtesting. Must remain False for operational forecasts.",
    )

    @field_validator("departure_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            date.fromisoformat(v.strip())
        except ValueError:
            raise ValueError(f"Invalid date format: '{v}'. Expected YYYY-MM-DD.")
        return v.strip()

    @field_validator("departure_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        cleaned = v.strip()
        # Ensure it has hours and minutes
        parts = cleaned.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid time format: '{v}'. Expected HH:MM.")
        try:
            h, m = int(parts[0]), int(parts[1])
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid time values: '{v}'. Expected HH:MM within 00:00-23:59.")
        return f"{h:02d}:{m:02d}"


class PredictionResponse(BaseModel):
    """Structured passenger demand forecast response schema."""

    route_id: str = Field(..., description="Corridor route identifier.")
    origin_city: str = Field(..., description="Name of the departure city.")
    destination_city: str = Field(..., description="Name of the arrival city.")
    departure_datetime: str = Field(..., description="ISO 8601 scheduled departure timestamp.")
    prediction_cutoff: str = Field(..., description="ISO 8601 forecast cutoff timestamp (scheduled departure - 24 hours).")
    predicted_passenger_demand: float = Field(
        ...,
        description="Predicted unconstrained passenger volume requested for the trip.",
    )
    bus_capacity: int = Field(..., description="Standard coach physical capacity (seats).")
    capacity_status: str = Field(
        ...,
        description="Operational status indicator ('within_capacity' or 'capacity_exceeded').",
    )
    estimated_unserved_demand: float = Field(
        ...,
        description="Estimated spillover / unserved demand exceeding physical capacity (max(0, predicted - capacity)).",
    )
    model_name: str = Field(..., description="Name of the machine learning model pipeline used for inference.")
    calendar_source: Optional[str] = Field("historical", description="Calendar partition used for feature lookup ('historical' or 'future').")


class HealthResponse(BaseModel):
    """Service and model health check schema."""

    status: str = Field(..., description="Overall service status ('healthy' or 'degraded').")
    service: str = Field("BusDemand AI Inference API", description="Service identifier.")
    version: str = Field("1.0.0", description="API version.")
    model_loaded: bool = Field(..., description="Whether the model pipeline is loaded and ready.")
    model_name: Optional[str] = Field(None, description="Active model identifier.")
    model_artifact_path: Optional[str] = Field(None, description="Path to serialized model artifact.")
    supported_horizon: Dict[str, str] = Field(
        default_factory=lambda: {
            "start_date": "2024-01-01",
            "end_date": "2026-12-31",
            "total_calendar_days": "1096",
        },
        description="Supported calendar and simulation bounds.",
    )
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Runtime environment metadata.",
    )


class RouteItem(BaseModel):
    """Individual corridor route metadata."""

    route_id: str
    origin_city_id: str
    destination_city_id: str
    origin_city: str
    destination_city: str
    distance_km: float
    estimated_duration_hours: float
    bus_capacity: int
    route_mobility_score: float
    route_demand_multiplier: float


class RoutesResponse(BaseModel):
    """Response containing list of all supported intercity routes."""

    total_routes: int
    routes: List[RouteItem]


class DepartureSlotItem(BaseModel):
    """Individual departure slot definition."""

    slot_id: str
    time: str
    name: str
    window: str


class DepartureSlotsResponse(BaseModel):
    """Response containing all supported daily departure slots and operational cutoff parameters."""

    total_slots: int
    slots: List[DepartureSlotItem]
    server_time: Optional[str] = Field(None, description="Current server time (ISO 8601 with timezone).")
    earliest_departure_datetime: Optional[str] = Field(None, description="Earliest allowable departure datetime (+24h).")
    min_departure_date: Optional[str] = Field(None, description="Earliest calendar date with at least one valid departure slot.")
    max_departure_date: Optional[str] = Field("2027-12-31", description="Maximum supported departure date.")
    timezone: Optional[str] = Field("Asia/Kolkata", description="Configured operating timezone.")

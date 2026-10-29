"""
Pydantic models used throughout the assistant.

Keeping these in one module gives us a single, testable source of truth for
what "valid data" means, both for the station catalogue loaded from JSON and
for the structured arguments the LLM passes into tools.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field, field_validator


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class ConnectorType(str, Enum):
    CCS_TYPE2 = "CCS Type 2"
    TYPE2_AC = "Type 2 AC"
    CHADEMO = "CHAdeMO"
    GBT = "GB/T"
    ANY = "Any"


class ConnectionStatus(str, Enum):
    AVAILABLE = "Available"
    OCCUPIED = "Occupied"
    FAULTED = "Faulted"
    UNKNOWN = "Unknown"


# --------------------------------------------------------------------------- #
# Station catalogue models (mirrors the JSON data source)
# --------------------------------------------------------------------------- #
class Location(BaseModel):
    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    nearest_sector: Optional[str] = None


class Connection(BaseModel):
    connection_type: str
    power_kw: float = Field(gt=0)
    status: ConnectionStatus = ConnectionStatus.UNKNOWN
    current_type: str = "DC"


class ChargingStation(BaseModel):
    station_id: str
    operator: str
    location: Location
    connections: list[Connection]
    pricing_per_kwh_inr: float = Field(gt=0)

    @field_validator("connections")
    @classmethod
    def _must_have_connections(cls, value: list[Connection]) -> list[Connection]:
        if not value:
            raise ValueError("A station must have at least one connection")
        return value

    @computed_field  # type: ignore[misc]
    @property
    def max_power_kw(self) -> float:
        return max(c.power_kw for c in self.connections)

    @computed_field  # type: ignore[misc]
    @property
    def has_available_connector(self) -> bool:
        return any(c.status == ConnectionStatus.AVAILABLE for c in self.connections)

    def distance_km_from(self, latitude: float, longitude: float) -> float:
        """Great-circle distance (haversine) from a point, in kilometres."""
        return haversine_km(latitude, longitude, self.location.latitude, self.location.longitude)


# --------------------------------------------------------------------------- #
# Route / elevation models (mirrors the elevation CSV data source)
# --------------------------------------------------------------------------- #
class RouteSegment(BaseModel):
    route_sector_id: str
    start_point: str
    end_point: str
    distance_km: float = Field(ge=0)
    elevation_gain_m: float
    avg_incline_deg: float
    base_energy_drain_wh_km: float = Field(gt=0)
    road_condition_index: float = Field(ge=0, le=10)


# --------------------------------------------------------------------------- #
# Tool argument / result schemas
# --------------------------------------------------------------------------- #
class StationSearchArgs(BaseModel):
    """Arguments accepted by the `search_charging_stations` tool."""

    near_latitude: Optional[float] = Field(
        default=None, ge=-90, le=90, description="Latitude of the point to search near"
    )
    near_longitude: Optional[float] = Field(
        default=None, ge=-180, le=180, description="Longitude of the point to search near"
    )
    location_name: Optional[str] = Field(
        default=None,
        description="Free-text place/sector name to search near (used if lat/lon are not given)",
    )
    radius_km: float = Field(default=25.0, gt=0, description="Search radius in kilometres")
    connector_type: ConnectorType = Field(default=ConnectorType.ANY)
    min_power_kw: float = Field(default=0.0, ge=0)
    max_price_per_kwh_inr: Optional[float] = Field(default=None, gt=0)
    only_available: bool = Field(
        default=True, description="If true, only return stations with at least one free connector"
    )
    max_results: int = Field(default=5, ge=1, le=20)


class ChargingCostArgs(BaseModel):
    """Arguments accepted by the `calculate_charging_cost` tool."""

    battery_capacity_kwh: float = Field(gt=0, description="Total usable battery capacity in kWh")
    current_charge_percent: float = Field(ge=0, le=100)
    target_charge_percent: float = Field(ge=0, le=100)
    price_per_kwh_inr: float = Field(gt=0)
    charging_power_kw: Optional[float] = Field(
        default=None, gt=0, description="Charger power, used to estimate charging time"
    )
    charging_efficiency_percent: float = Field(
        default=90.0, gt=0, le=100, description="Round-trip charging efficiency (AC/DC losses)"
    )

    @field_validator("target_charge_percent")
    @classmethod
    def _target_after_current(cls, value: float, info) -> float:
        current = info.data.get("current_charge_percent")
        if current is not None and value <= current:
            raise ValueError("target_charge_percent must be greater than current_charge_percent")
        return value


class RouteEnergyArgs(BaseModel):
    """Arguments accepted by the `estimate_route_energy` tool."""

    start_point: str = Field(description="Starting town/sector name, e.g. 'Chandigarh'")
    end_point: str = Field(description="Destination town/sector name, e.g. 'Solan'")
    vehicle_efficiency_wh_km: Optional[float] = Field(
        default=None,
        gt=0,
        description="Vehicle's flat-road energy consumption in Wh/km; falls back to route data if omitted",
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

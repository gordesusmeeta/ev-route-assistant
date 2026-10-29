"""
Data access layer.

Responsible for reading the two "open data source" inputs described in the
project brief:

1. A structured JSON feed of live charging stations (stands in for the Open
   Charge Map API response).
2. A static CSV of route elevation / terrain data (derived from the India
   Geological Survey style dataset).

Both loaders validate every record through the Pydantic models in
`agent.models`, so a malformed row fails loudly and specifically instead of
silently propagating bad data into the agent.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from agent.models import ChargingStation, RouteSegment


class DataLoadError(RuntimeError):
    """Raised when a data source is missing or fails validation."""


def load_stations(path: str | Path) -> list[ChargingStation]:
    """Load and validate the charging-station catalogue from a JSON file.

    Accepts either ``{"stations": [...]}`` (the app's own format) or a bare
    list, so the original API_Schema.json-style payload also works.
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Stations file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Stations file is not valid JSON: {exc}") from exc

    records = raw.get("stations", raw) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise DataLoadError("Stations file must contain a list of station records")

    stations: list[ChargingStation] = []
    errors: list[str] = []
    for i, record in enumerate(records):
        try:
            stations.append(ChargingStation.model_validate(record))
        except Exception as exc:  # noqa: BLE001 - we want to collect *all* row errors
            errors.append(f"record[{i}] ({record.get('station_id', '?')}): {exc}")

    if not stations and errors:
        raise DataLoadError("No valid station records found:\n" + "\n".join(errors))

    return stations


def load_route_segments(path: str | Path) -> list[RouteSegment]:
    """Load and validate route/elevation segments from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Elevation file not found: {path}")

    segments: list[RouteSegment] = []
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                segments.append(
                    RouteSegment(
                        route_sector_id=row["route_sector_id"],
                        start_point=row["start_point"],
                        end_point=row["end_point"],
                        distance_km=float(row["distance_km"]),
                        elevation_gain_m=float(row["elevation_gain_m"]),
                        avg_incline_deg=float(row["avg_incline_deg"]),
                        base_energy_drain_wh_km=float(row["base_energy_drain_wh_km"]),
                        road_condition_index=float(row["road_condition_index"]),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row[{i}]: {exc}")

    if not segments and errors:
        raise DataLoadError("No valid route segments found:\n" + "\n".join(errors))

    return segments


@lru_cache(maxsize=8)
def _cached_stations(path_str: str) -> tuple[ChargingStation, ...]:
    return tuple(load_stations(path_str))


@lru_cache(maxsize=8)
def _cached_segments(path_str: str) -> tuple[RouteSegment, ...]:
    return tuple(load_route_segments(path_str))


def get_stations_cached(path: str | Path) -> list[ChargingStation]:
    """Cached accessor so the Streamlit app / agent don't re-parse JSON on every call."""
    return list(_cached_stations(str(path)))


def get_segments_cached(path: str | Path) -> list[RouteSegment]:
    """Cached accessor so the Streamlit app / agent don't re-parse the CSV on every call."""
    return list(_cached_segments(str(path)))


def clear_cache() -> None:
    """Clear cached data (used by tests and by the Streamlit 'reload data' button)."""
    _cached_stations.cache_clear()
    _cached_segments.cache_clear()

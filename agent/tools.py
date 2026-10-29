"""
LangChain tools exposed to the agent.

Three tools are provided:

* ``search_charging_stations``  - filters/ranks the station catalogue by
  distance, connector type, availability, power, and price.
* ``calculate_charging_cost``   - pure arithmetic: cost + time to charge.
* ``estimate_route_energy``     - walks the elevation dataset between two
  named points and estimates extra energy draw from terrain.

Each tool validates its own arguments through a Pydantic model (see
``agent.models``) before doing any work, and returns a plain string (LLM tool
outputs should be text) plus - always - a `warnings` field when something is
ambiguous.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from agent.config import Settings
from agent.data_loader import DataLoadError, get_segments_cached, get_stations_cached
from agent.models import (
    ChargingCostArgs,
    ChargingStation,
    ConnectorType,
    RouteEnergyArgs,
    StationSearchArgs,
    haversine_km,
)


class ToolError(RuntimeError):
    """Raised for user-facing tool failures (bad input, no data, etc.)."""


# --------------------------------------------------------------------------- #
# Tool 1: station search
# --------------------------------------------------------------------------- #
def _resolve_search_origin(args: StationSearchArgs, settings: Settings) -> tuple[float, float]:
    """Work out the (lat, lon) to search around."""
    if args.near_latitude is not None and args.near_longitude is not None:
        return args.near_latitude, args.near_longitude

    if args.location_name:
        segments = get_segments_cached(settings.elevation_file)
        stations = get_stations_cached(settings.stations_file)
        name = args.location_name.strip().lower()

        # 1) Try to match a named place in the route/elevation data via a station
        #    that reports that sector, since the elevation file has no lat/lon.
        for seg in segments:
            if name in (seg.start_point.lower(), seg.end_point.lower()):
                for st in stations:
                    if st.location.nearest_sector == seg.route_sector_id:
                        return st.location.latitude, st.location.longitude

        # 2) Fall back to a direct station-name / location-name match.
        for st in stations:
            if name in st.location.name.lower() or name in st.operator.lower():
                return st.location.latitude, st.location.longitude

        raise ToolError(
            f"Could not resolve location '{args.location_name}' to coordinates. "
            "Try providing near_latitude/near_longitude, or a place name that "
            "appears in the route data (e.g. 'Solan', 'Kalka', 'Dharampur')."
        )

    raise ToolError("Provide either near_latitude/near_longitude or a location_name to search near.")


def _connector_matches(station: ChargingStation, wanted: ConnectorType) -> bool:
    if wanted == ConnectorType.ANY:
        return True
    return any(c.connection_type == wanted.value for c in station.connections)


def search_charging_stations(**kwargs) -> str:
    """Search the charging-station catalogue and return the best matches as JSON text."""
    settings = kwargs.pop("_settings", None) or Settings()
    try:
        args = StationSearchArgs.model_validate(kwargs)
    except ValidationError as exc:
        return _error_payload(f"Invalid search arguments: {exc}")

    try:
        origin_lat, origin_lon = _resolve_search_origin(args, settings)
        stations = get_stations_cached(settings.stations_file)
    except (ToolError, DataLoadError) as exc:
        return _error_payload(str(exc))

    candidates = []
    for st in stations:
        distance = st.distance_km_from(origin_lat, origin_lon)
        if distance > args.radius_km:
            continue
        if not _connector_matches(st, args.connector_type):
            continue
        if st.max_power_kw < args.min_power_kw:
            continue
        if args.max_price_per_kwh_inr is not None and st.pricing_per_kwh_inr > args.max_price_per_kwh_inr:
            continue
        if args.only_available and not st.has_available_connector:
            continue
        candidates.append((distance, st))

    candidates.sort(key=lambda pair: (pair[0], pair[1].pricing_per_kwh_inr))
    top = candidates[: args.max_results]

    if not top:
        return _error_payload(
            "No stations matched the given filters within "
            f"{args.radius_km} km. Try widening radius_km, lowering min_power_kw, "
            "raising max_price_per_kwh_inr, or setting only_available to false."
        )

    results = [
        {
            "station_id": st.station_id,
            "operator": st.operator,
            "location_name": st.location.name,
            "distance_km": round(distance, 2),
            "connections": [
                {
                    "connection_type": c.connection_type,
                    "power_kw": c.power_kw,
                    "status": c.status.value,
                }
                for c in st.connections
            ],
            "max_power_kw": st.max_power_kw,
            "pricing_per_kwh_inr": st.pricing_per_kwh_inr,
            "has_available_connector": st.has_available_connector,
        }
        for distance, st in top
    ]
    return json.dumps({"origin": {"latitude": origin_lat, "longitude": origin_lon}, "results": results}, indent=2)


# --------------------------------------------------------------------------- #
# Tool 2: charging cost / time calculator
# --------------------------------------------------------------------------- #
def calculate_charging_cost(**kwargs) -> str:
    """Compute the energy, cost, and (optionally) time required for a charging session."""
    try:
        args = ChargingCostArgs.model_validate(kwargs)
    except ValidationError as exc:
        return _error_payload(f"Invalid charging-cost arguments: {exc}")

    percent_needed = args.target_charge_percent - args.current_charge_percent
    energy_needed_kwh = args.battery_capacity_kwh * (percent_needed / 100.0)

    # Charging losses mean more energy is drawn from the charger/grid than
    # actually lands in the battery.
    efficiency = args.charging_efficiency_percent / 100.0
    energy_drawn_kwh = energy_needed_kwh / efficiency

    cost_inr = energy_drawn_kwh * args.price_per_kwh_inr

    time_minutes = None
    if args.charging_power_kw:
        time_minutes = (energy_drawn_kwh / args.charging_power_kw) * 60.0

    result = {
        "percent_charged": round(percent_needed, 2),
        "energy_added_to_battery_kwh": round(energy_needed_kwh, 3),
        "energy_drawn_from_charger_kwh": round(energy_drawn_kwh, 3),
        "estimated_cost_inr": round(cost_inr, 2),
        "estimated_time_minutes": round(time_minutes, 1) if time_minutes is not None else None,
        "assumptions": {
            "charging_efficiency_percent": args.charging_efficiency_percent,
            "note": (
                "Time estimate assumes constant charging power; real DC fast "
                "charging tapers above ~80% state of charge."
            ),
        },
    }
    return json.dumps(result, indent=2)


# --------------------------------------------------------------------------- #
# Tool 3: route energy estimator (uses the static elevation dataset)
# --------------------------------------------------------------------------- #
def estimate_route_energy(**kwargs) -> str:
    """Estimate extra energy consumption for a route using terrain/elevation data."""
    settings = kwargs.pop("_settings", None) or Settings()
    try:
        args = RouteEnergyArgs.model_validate(kwargs)
    except ValidationError as exc:
        return _error_payload(f"Invalid route-energy arguments: {exc}")

    try:
        segments = get_segments_cached(settings.elevation_file)
    except DataLoadError as exc:
        return _error_payload(str(exc))

    path = _find_segment_path(segments, args.start_point, args.end_point)
    if not path:
        known_points = sorted({s.start_point for s in segments} | {s.end_point for s in segments})
        return _error_payload(
            f"No contiguous route found between '{args.start_point}' and '{args.end_point}'. "
            f"Known points include: {', '.join(known_points[:15])}{'...' if len(known_points) > 15 else ''}"
        )

    total_distance_km = sum(seg.distance_km for seg in path)
    total_elevation_gain_m = sum(max(seg.elevation_gain_m, 0.0) for seg in path)

    # Base (flat-road) consumption: prefer the vehicle's own rating if given,
    # otherwise fall back to a distance-weighted average from the route data.
    if args.vehicle_efficiency_wh_km:
        base_wh_per_km = args.vehicle_efficiency_wh_km
    else:
        base_wh_per_km = (
            sum(seg.base_energy_drain_wh_km * seg.distance_km for seg in path) / total_distance_km
            if total_distance_km
            else 0.0
        )

    segment_breakdown = []
    total_energy_wh = 0.0
    for seg in path:
        incline_factor = _incline_energy_multiplier(seg.avg_incline_deg)
        segment_wh_per_km = seg.base_energy_drain_wh_km * incline_factor if not args.vehicle_efficiency_wh_km else (
            base_wh_per_km * incline_factor
        )
        segment_energy_wh = segment_wh_per_km * seg.distance_km
        total_energy_wh += segment_energy_wh
        segment_breakdown.append(
            {
                "sector": seg.route_sector_id,
                "from": seg.start_point,
                "to": seg.end_point,
                "distance_km": seg.distance_km,
                "avg_incline_deg": seg.avg_incline_deg,
                "incline_energy_multiplier": round(incline_factor, 2),
                "estimated_energy_wh": round(segment_energy_wh, 1),
            }
        )

    flat_road_energy_wh = base_wh_per_km * total_distance_km
    extra_from_terrain_wh = total_energy_wh - flat_road_energy_wh

    result = {
        "route": f"{args.start_point} -> {args.end_point}",
        "total_distance_km": round(total_distance_km, 2),
        "total_elevation_gain_m": round(total_elevation_gain_m, 1),
        "baseline_flat_road_wh_km": round(base_wh_per_km, 1),
        "estimated_total_energy_kwh": round(total_energy_wh / 1000.0, 3),
        "extra_energy_from_terrain_kwh": round(extra_from_terrain_wh / 1000.0, 3),
        "terrain_drain_multiplier_overall": (
            round(total_energy_wh / flat_road_energy_wh, 2) if flat_road_energy_wh else None
        ),
        "segments": segment_breakdown,
    }
    return json.dumps(result, indent=2)


def _incline_energy_multiplier(avg_incline_deg: float) -> float:
    """Very simple physics-flavoured model: steeper uphill grades drain the
    battery faster; downhill segments recover some energy via regen braking
    (capped, since regen is never 100% efficient).
    """
    if avg_incline_deg >= 0:
        # Roughly +18% consumption per degree of average incline, capped at 3x.
        return min(1.0 + 0.18 * avg_incline_deg, 3.0)
    # Downhill: regen recovers energy, but with losses; floor at 40% of flat draw.
    return max(1.0 + 0.10 * avg_incline_deg, 0.4)


def _find_segment_path(segments, start_point: str, end_point: str):
    """Find a contiguous chain of segments from start_point to end_point.

    The elevation CSV is an ordered chain (each segment's end_point feeds the
    next segment's start_point), so a simple forward walk from the first
    matching start is sufficient for this dataset; a small BFS fallback
    handles any out-of-order rows.
    """
    start_norm, end_norm = start_point.strip().lower(), end_point.strip().lower()

    # Fast path: assume ordering matches the CSV.
    collecting = False
    path = []
    for seg in segments:
        if seg.start_point.lower() == start_norm:
            collecting = True
        if collecting:
            path.append(seg)
        if collecting and seg.end_point.lower() == end_norm:
            return path
    if collecting:
        # started but never reached end_point in-order; fall through to BFS
        pass

    # BFS fallback over the segment graph (handles branches / reversed order).
    from collections import deque

    graph: dict[str, list] = {}
    for seg in segments:
        graph.setdefault(seg.start_point.lower(), []).append(seg)

    queue = deque([(start_norm, [])])
    visited = {start_norm}
    while queue:
        node, so_far = queue.popleft()
        if node == end_norm and so_far:
            return so_far
        for seg in graph.get(node, []):
            nxt = seg.end_point.lower()
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, so_far + [seg]))
    return []


def _error_payload(message: str) -> str:
    return json.dumps({"error": message})


# --------------------------------------------------------------------------- #
# LangChain StructuredTool wrappers
# --------------------------------------------------------------------------- #
def build_tools(settings: Optional[Settings] = None) -> list[StructuredTool]:
    """Build the list of LangChain tools bound to a given Settings instance."""
    settings = settings or Settings()

    def _search(**kwargs) -> str:
        return search_charging_stations(_settings=settings, **kwargs)

    def _cost(**kwargs) -> str:
        return calculate_charging_cost(**kwargs)

    def _energy(**kwargs) -> str:
        return estimate_route_energy(_settings=settings, **kwargs)

    return [
        StructuredTool.from_function(
            func=_search,
            name="search_charging_stations",
            description=(
                "Search for EV charging stations near a point on the route. "
                "Filter by connector_type, min_power_kw, max_price_per_kwh_inr, "
                "and only_available. Provide either near_latitude/near_longitude "
                "or a location_name (e.g. a town along the route)."
            ),
            args_schema=StationSearchArgs,
        ),
        StructuredTool.from_function(
            func=_cost,
            name="calculate_charging_cost",
            description=(
                "Calculate the energy (kWh), cost (INR), and charging time for "
                "going from a current battery percentage to a target percentage "
                "at a given price per kWh and charger power."
            ),
            args_schema=ChargingCostArgs,
        ),
        StructuredTool.from_function(
            func=_energy,
            name="estimate_route_energy",
            description=(
                "Estimate the energy (kWh) required to drive between two named "
                "points on the route, accounting for elevation/incline using the "
                "static terrain dataset. Use this before recommending a charging "
                "stop on a hilly route."
            ),
            args_schema=RouteEnergyArgs,
        ),
    ]

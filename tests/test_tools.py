import json

import pytest

from agent.tools import (
    build_tools,
    calculate_charging_cost,
    estimate_route_energy,
    search_charging_stations,
)


class TestSearchChargingStations:
    def test_search_near_coordinates_returns_results(self, settings):
        raw = search_charging_stations(
            near_latitude=30.9062,
            near_longitude=76.9845,
            radius_km=50,
            _settings=settings,
        )
        payload = json.loads(raw)
        assert "results" in payload
        assert len(payload["results"]) > 0
        assert payload["results"][0]["distance_km"] >= 0

    def test_search_by_location_name(self, settings):
        raw = search_charging_stations(location_name="Solan", radius_km=10, _settings=settings)
        payload = json.loads(raw)
        assert "results" in payload
        assert any("Solan" in r["location_name"] for r in payload["results"])

    def test_search_filters_by_connector_type(self, settings):
        raw = search_charging_stations(
            near_latitude=30.9062,
            near_longitude=76.9845,
            radius_km=100,
            connector_type="CHAdeMO",
            only_available=False,
            _settings=settings,
        )
        payload = json.loads(raw)
        for r in payload["results"]:
            assert any(c["connection_type"] == "CHAdeMO" for c in r["connections"])

    def test_search_filters_by_max_price(self, settings):
        raw = search_charging_stations(
            near_latitude=30.9062,
            near_longitude=76.9845,
            radius_km=200,
            max_price_per_kwh_inr=15.0,
            only_available=False,
            _settings=settings,
        )
        payload = json.loads(raw)
        for r in payload["results"]:
            assert r["pricing_per_kwh_inr"] <= 15.0

    def test_search_no_results_returns_error_payload(self, settings):
        raw = search_charging_stations(
            near_latitude=0.0,
            near_longitude=0.0,
            radius_km=1,
            _settings=settings,
        )
        payload = json.loads(raw)
        assert "error" in payload

    def test_search_unresolvable_location_returns_error(self, settings):
        raw = search_charging_stations(location_name="Atlantis", _settings=settings)
        payload = json.loads(raw)
        assert "error" in payload

    def test_search_missing_origin_returns_error(self, settings):
        raw = search_charging_stations(_settings=settings)
        payload = json.loads(raw)
        assert "error" in payload


class TestCalculateChargingCost:
    def test_basic_calculation(self):
        raw = calculate_charging_cost(
            battery_capacity_kwh=50,
            current_charge_percent=20,
            target_charge_percent=80,
            price_per_kwh_inr=18.5,
            charging_power_kw=50,
            charging_efficiency_percent=100,
        )
        payload = json.loads(raw)
        # 60% of 50kWh = 30kWh, at 100% efficiency, at 18.5/kWh => 555 INR
        assert payload["energy_added_to_battery_kwh"] == pytest.approx(30.0)
        assert payload["estimated_cost_inr"] == pytest.approx(555.0)
        assert payload["estimated_time_minutes"] == pytest.approx(36.0)

    def test_efficiency_increases_cost(self):
        raw = calculate_charging_cost(
            battery_capacity_kwh=50,
            current_charge_percent=20,
            target_charge_percent=80,
            price_per_kwh_inr=18.5,
            charging_efficiency_percent=90,
        )
        payload = json.loads(raw)
        assert payload["energy_drawn_from_charger_kwh"] > payload["energy_added_to_battery_kwh"]

    def test_invalid_target_lower_than_current_returns_error(self):
        raw = calculate_charging_cost(
            battery_capacity_kwh=50,
            current_charge_percent=80,
            target_charge_percent=20,
            price_per_kwh_inr=18.5,
        )
        payload = json.loads(raw)
        assert "error" in payload

    def test_no_charging_power_omits_time(self):
        raw = calculate_charging_cost(
            battery_capacity_kwh=50,
            current_charge_percent=20,
            target_charge_percent=80,
            price_per_kwh_inr=18.5,
        )
        payload = json.loads(raw)
        assert payload["estimated_time_minutes"] is None


class TestEstimateRouteEnergy:
    def test_basic_route_estimate(self, settings):
        raw = estimate_route_energy(start_point="Chandigarh", end_point="Solan", _settings=settings)
        payload = json.loads(raw)
        assert payload["total_distance_km"] > 0
        assert payload["estimated_total_energy_kwh"] > 0
        assert len(payload["segments"]) >= 1

    def test_unknown_route_returns_error(self, settings):
        raw = estimate_route_energy(start_point="Narnia", end_point="Mordor", _settings=settings)
        payload = json.loads(raw)
        assert "error" in payload

    def test_custom_vehicle_efficiency_changes_result(self, settings):
        low = json.loads(
            estimate_route_energy(
                start_point="Chandigarh", end_point="Solan", vehicle_efficiency_wh_km=100, _settings=settings
            )
        )
        high = json.loads(
            estimate_route_energy(
                start_point="Chandigarh", end_point="Solan", vehicle_efficiency_wh_km=300, _settings=settings
            )
        )
        assert high["estimated_total_energy_kwh"] > low["estimated_total_energy_kwh"]


class TestBuildTools:
    def test_build_tools_returns_three_tools_with_expected_names(self, settings):
        tools = build_tools(settings)
        names = {t.name for t in tools}
        assert names == {"search_charging_stations", "calculate_charging_cost", "estimate_route_energy"}

import pytest
from pydantic import ValidationError

from agent.models import (
    ChargingCostArgs,
    ChargingStation,
    Connection,
    ConnectionStatus,
    Location,
    haversine_km,
)


def _sample_station(**overrides) -> dict:
    base = {
        "station_id": "IN-EV-0001",
        "operator": "Test Operator",
        "location": {"name": "Test Plaza", "latitude": 30.9, "longitude": 76.9},
        "connections": [
            {"connection_type": "CCS Type 2", "power_kw": 50, "status": "Available", "current_type": "DC"}
        ],
        "pricing_per_kwh_inr": 18.5,
    }
    base.update(overrides)
    return base


class TestChargingStation:
    def test_valid_station_parses(self):
        station = ChargingStation.model_validate(_sample_station())
        assert station.station_id == "IN-EV-0001"
        assert station.max_power_kw == 50
        assert station.has_available_connector is True

    def test_rejects_empty_connections(self):
        with pytest.raises(ValidationError):
            ChargingStation.model_validate(_sample_station(connections=[]))

    def test_rejects_bad_latitude(self):
        with pytest.raises(ValidationError):
            ChargingStation.model_validate(
                _sample_station(location={"name": "X", "latitude": 999, "longitude": 0})
            )

    def test_rejects_non_positive_price(self):
        with pytest.raises(ValidationError):
            ChargingStation.model_validate(_sample_station(pricing_per_kwh_inr=0))

    def test_max_power_kw_picks_highest_connector(self):
        station = ChargingStation.model_validate(
            _sample_station(
                connections=[
                    {"connection_type": "Type 2 AC", "power_kw": 22, "status": "Available"},
                    {"connection_type": "CCS Type 2", "power_kw": 100, "status": "Occupied"},
                ]
            )
        )
        assert station.max_power_kw == 100

    def test_has_available_connector_false_when_all_occupied(self):
        station = ChargingStation.model_validate(
            _sample_station(connections=[{"connection_type": "CCS Type 2", "power_kw": 50, "status": "Occupied"}])
        )
        assert station.has_available_connector is False

    def test_distance_km_from_zero_at_same_point(self):
        station = ChargingStation.model_validate(_sample_station())
        d = station.distance_km_from(30.9, 76.9)
        assert d == pytest.approx(0.0, abs=1e-6)


class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)

    def test_known_distance_delhi_to_chandigarh(self):
        # Delhi ~ (28.6139, 77.2090), Chandigarh ~ (30.7333, 76.7794)
        d = haversine_km(28.6139, 77.2090, 30.7333, 76.7794)
        assert 220 < d < 260  # sanity-checked real-world distance is ~240km


class TestChargingCostArgs:
    def test_target_must_exceed_current(self):
        with pytest.raises(ValidationError):
            ChargingCostArgs(
                battery_capacity_kwh=50,
                current_charge_percent=80,
                target_charge_percent=50,
                price_per_kwh_inr=18,
            )

    def test_valid_args(self):
        args = ChargingCostArgs(
            battery_capacity_kwh=50,
            current_charge_percent=20,
            target_charge_percent=80,
            price_per_kwh_inr=18,
        )
        assert args.target_charge_percent == 80

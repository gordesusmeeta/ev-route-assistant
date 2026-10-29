import json

import pytest

from agent.data_loader import DataLoadError, load_route_segments, load_stations


class TestLoadStations:
    def test_loads_bundled_sample_data(self, settings):
        stations = load_stations(settings.stations_file)
        assert len(stations) >= 5
        assert all(s.station_id for s in stations)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataLoadError):
            load_stations(tmp_path / "does_not_exist.json")

    def test_invalid_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(DataLoadError):
            load_stations(bad_file)

    def test_accepts_bare_list_format(self, tmp_path):
        payload = [
            {
                "station_id": "X-1",
                "operator": "Op",
                "location": {"name": "Loc", "latitude": 1.0, "longitude": 1.0},
                "connections": [{"connection_type": "CCS Type 2", "power_kw": 50, "status": "Available"}],
                "pricing_per_kwh_inr": 10.0,
            }
        ]
        f = tmp_path / "list.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        stations = load_stations(f)
        assert len(stations) == 1

    def test_skips_invalid_rows_but_keeps_valid_ones(self, tmp_path):
        payload = {
            "stations": [
                {
                    "station_id": "GOOD-1",
                    "operator": "Op",
                    "location": {"name": "Loc", "latitude": 1.0, "longitude": 1.0},
                    "connections": [{"connection_type": "CCS Type 2", "power_kw": 50, "status": "Available"}],
                    "pricing_per_kwh_inr": 10.0,
                },
                {
                    "station_id": "BAD-1",
                    "operator": "Op",
                    "location": {"name": "Loc", "latitude": 999.0, "longitude": 1.0},  # invalid latitude
                    "connections": [{"connection_type": "CCS Type 2", "power_kw": 50, "status": "Available"}],
                    "pricing_per_kwh_inr": 10.0,
                },
            ]
        }
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        stations = load_stations(f)
        assert len(stations) == 1
        assert stations[0].station_id == "GOOD-1"


class TestLoadRouteSegments:
    def test_loads_bundled_sample_data(self, settings):
        segments = load_route_segments(settings.elevation_file)
        assert len(segments) >= 5
        assert all(seg.distance_km >= 0 for seg in segments)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataLoadError):
            load_route_segments(tmp_path / "nope.csv")

    def test_parses_negative_incline_for_downhill(self, tmp_path):
        f = tmp_path / "route.csv"
        f.write_text(
            "route_sector_id,start_point,end_point,distance_km,elevation_gain_m,"
            "avg_incline_deg,base_energy_drain_wh_km,road_condition_index\n"
            "SEC-1,A,B,10,-50,-2.0,120,8\n",
            encoding="utf-8",
        )
        segments = load_route_segments(f)
        assert segments[0].avg_incline_deg == -2.0

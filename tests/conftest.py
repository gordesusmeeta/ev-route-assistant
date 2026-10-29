from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Settings
from agent.data_loader import clear_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clear_data_cache():
    """Every test starts with a clean data cache so file fixtures don't leak."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key="test-key-not-real",
        stations_file=PROJECT_ROOT / "data" / "charging_stations.json",
        elevation_file=PROJECT_ROOT / "data" / "route_elevation.csv",
    )

"""
Application configuration.

All runtime configuration is loaded from environment variables (optionally via a
.env file). Nothing secret is hard-coded. See `.env.example` for the full list
of supported variables.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Central, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM configuration -------------------------------------------------
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Chat model used by the agent")
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # --- Data sources --------------------------------------------------------
    stations_file: Path = Field(default=DATA_DIR / "charging_stations.json")
    elevation_file: Path = Field(default=DATA_DIR / "route_elevation.csv")

    # --- Agent behaviour -----------------------------------------------------
    max_tool_iterations: int = Field(default=6, ge=1, le=20)
    default_search_radius_km: float = Field(default=25.0, gt=0)

    @field_validator("openai_api_key")
    @classmethod
    def _warn_if_missing(cls, value: str) -> str:
        # Left permissive on purpose: importing config should never crash the
        # test-suite or the Streamlit app before the user has had a chance to
        # enter a key in the sidebar. Actual LLM calls fail fast instead,
        # with a clear error (see agent/graph.py).
        return value


def get_settings() -> Settings:
    """Return a fresh Settings instance (re-reads the environment each call).

    Using a function rather than a module-level singleton makes it trivial to
    monkeypatch environment variables in tests and lets the Streamlit sidebar
    override the API key at runtime.
    """
    return Settings()

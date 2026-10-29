"""
Streamlit front-end for the EV Charging Route Assistant.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent.config import Settings, get_settings
from agent.data_loader import DataLoadError, clear_cache, get_segments_cached, get_stations_cached
from agent.graph import AgentUnavailableError, EVRouteAgent

load_dotenv()

st.set_page_config(page_title="EV Charging Route Assistant", page_icon="🔋", layout="wide")


# --------------------------------------------------------------------------- #
# Sidebar: configuration
# --------------------------------------------------------------------------- #
st.sidebar.title("🔋 EV Route Assistant")
st.sidebar.caption("LangGraph + LangChain + OpenAI")

api_key_input = st.sidebar.text_input(
    "OpenAI API key",
    value=os.environ.get("OPENAI_API_KEY", ""),
    type="password",
    help="Loaded from .env if present. You can also paste one here for this session.",
)
model_name = st.sidebar.selectbox(
    "Model",
    options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    index=0,
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

with st.sidebar.expander("Data sources", expanded=False):
    st.write("Charging stations: `data/charging_stations.json`")
    st.write("Route elevation: `data/route_elevation.csv`")
    if st.button("Reload data files"):
        clear_cache()
        st.success("Data cache cleared — will reload on next query.")

if st.sidebar.button("Reset conversation"):
    st.session_state.pop("chat_history", None)
    st.session_state.pop("display_history", None)
    st.rerun()


def _get_settings() -> Settings:
    settings = get_settings()
    if api_key_input:
        settings = settings.model_copy(update={"openai_api_key": api_key_input})
    settings = settings.model_copy(update={"openai_model": model_name, "openai_temperature": temperature})
    return settings


# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #
st.title("🔋 EV Charging Route Assistant")
st.caption(
    "Ask about a route, your current battery level, and I'll estimate energy use "
    "over hilly terrain and recommend real charging stations along the way."
)

tab_chat, tab_data = st.tabs(["💬 Chat", "📊 Data explorer"])

with tab_chat:
    settings = _get_settings()

    if not settings.openai_api_key:
        st.info("Enter an OpenAI API key in the sidebar (or set OPENAI_API_KEY in a .env file) to start chatting.")

    with st.expander("Example questions"):
        st.markdown(
            "- *I'm driving from Chandigarh to Solan with 40% battery in a car with a "
            "50 kWh battery — will I make it, and where should I charge?*\n"
            "- *Find me a fast DC charger near Solan under ₹20/kWh.*\n"
            "- *How much will it cost to charge from 20% to 80% on a 60 kWh battery "
            "at ₹18.5/kWh on a 50 kW charger?*"
        )

    if "display_history" not in st.session_state:
        st.session_state.display_history = []

    for role, content in st.session_state.display_history:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask about your EV route, range, or charging stops...")

    if user_input:
        st.session_state.display_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")
            try:
                agent = EVRouteAgent(settings=settings)
                history = st.session_state.get("chat_history", [])
                reply_text, all_messages = agent.run_and_get_reply(user_input, history=history)

                # Persist LangChain message history (excluding the leading system
                # message, which run_and_get_reply re-adds each turn).
                st.session_state["chat_history"] = [m for m in all_messages if m.type != "system"]

                placeholder.markdown(reply_text)
                st.session_state.display_history.append(("assistant", reply_text))

                with st.expander("🔧 Tool activity for this turn"):
                    for m in all_messages:
                        if m.type == "ai" and getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                st.markdown(f"**Called** `{tc['name']}`")
                                st.json(tc["args"])
                        elif m.type == "tool":
                            st.markdown(f"**Result from** `{m.name}`")
                            st.code(m.content, language="json")

            except AgentUnavailableError as exc:
                placeholder.error(str(exc))
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
                placeholder.error(f"Something went wrong while answering: {exc}")

with tab_data:
    st.subheader("Charging stations")
    try:
        stations = get_stations_cached(settings.stations_file)
        st.dataframe(
            [
                {
                    "station_id": s.station_id,
                    "operator": s.operator,
                    "location": s.location.name,
                    "lat": s.location.latitude,
                    "lon": s.location.longitude,
                    "max_power_kw": s.max_power_kw,
                    "price_inr_per_kwh": s.pricing_per_kwh_inr,
                    "available": s.has_available_connector,
                }
                for s in stations
            ],
            use_container_width=True,
        )
        st.map(
            [{"lat": s.location.latitude, "lon": s.location.longitude} for s in stations],
            size=40,
        )
    except DataLoadError as exc:
        st.error(str(exc))

    st.subheader("Route elevation / terrain")
    try:
        segments = get_segments_cached(settings.elevation_file)
        st.dataframe(
            [
                {
                    "sector": seg.route_sector_id,
                    "from": seg.start_point,
                    "to": seg.end_point,
                    "distance_km": seg.distance_km,
                    "elevation_gain_m": seg.elevation_gain_m,
                    "avg_incline_deg": seg.avg_incline_deg,
                    "base_wh_km": seg.base_energy_drain_wh_km,
                    "road_condition_index": seg.road_condition_index,
                }
                for seg in segments
            ],
            use_container_width=True,
        )
    except DataLoadError as exc:
        st.error(str(exc))

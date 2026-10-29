# ev-route-assistant
# 🔋 EV Charging Route Assistant

An AI co-pilot for electric-vehicle road trips. It answers natural-language
questions about a route, estimates how much energy hills/terrain will cost
you, finds real charging stations that match your needs, and works out what
a charging stop will cost and how long it will take.

Built with **LangGraph** (agent orchestration) + **LangChain** (tool
calling) + the **OpenAI API** (reasoning), served through a **Streamlit**
chat UI, with **Pydantic** validating every data source and every tool call.

> This project was built around the "Smart EV Navigator" brief: use a
> static terrain/elevation dataset to model how much extra battery a hilly
> route costs, and a live charging-station feed to find somewhere to
> recharge before the battery runs out.

---

## How it works

```
User question
     │
     ▼
┌─────────────────────────────┐
│   Streamlit chat UI (app.py)│
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────────────────┐
│  LangGraph agent (agent/graph.py)        │
│                                           │
│   START → agent ⇄ tools → END            │
│           (OpenAI function-calling loop) │
└──────────────┬────────────────────────────┘
               ▼
     ┌─────────────────────────────┐
     │   agent/tools.py             │
     │  • search_charging_stations  │
     │  • calculate_charging_cost   │
     │  • estimate_route_energy     │
     └──────────────┬───────────────┘
                     ▼
     ┌─────────────────────────────┐
     │  agent/data_loader.py        │
     │  validated via agent/models.py│
     └──────────────┬───────────────┘
                     ▼
     data/charging_stations.json   (live station feed, Open-Charge-Map style)
     data/route_elevation.csv      (static terrain / elevation dataset)
```

The LLM never invents numbers: every distance, energy, cost, or time figure
in a response comes from one of the three tools, which do real arithmetic
(or real data lookups) and return structured JSON that the model then
explains in plain language.

### The three tools

| Tool | Purpose | Key inputs |
|---|---|---|
| `estimate_route_energy` | Walks the elevation dataset between two named points and estimates the extra energy an uphill/downhill route costs vs. flat road, segment by segment. | `start_point`, `end_point`, optional `vehicle_efficiency_wh_km` |
| `search_charging_stations` | Filters/ranks the station catalogue by distance, connector type, minimum power, max price, and live availability. | coordinates or a `location_name`, `connector_type`, `min_power_kw`, `max_price_per_kwh_inr`, `only_available` |
| `calculate_charging_cost` | Computes energy needed, cost in INR, and (if charger power is known) charging time, including a configurable charging-efficiency loss. | `battery_capacity_kwh`, `current_charge_percent`, `target_charge_percent`, `price_per_kwh_inr`, optional `charging_power_kw` |

### Data sources

* **`data/charging_stations.json`** — a structured JSON feed styled after
  the Open Charge Map API (station id, operator, GPS location, one or more
  connectors with type/power/status, and price per kWh). Seeded from the
  provided `API_Schema.json` sample and extended with more stations along
  the Chandigarh–Shimla highway (NH-5) for a realistic demo.
* **`data/route_elevation.csv`** — a static terrain dataset (route sector,
  start/end point, distance, elevation gain, average incline, baseline
  energy drain, road-condition index), converted from the provided
  elevation spreadsheet.

Both are loaded and validated through Pydantic models
(`agent/models.py`) before the agent ever sees them — malformed rows are
rejected individually with a clear error rather than crashing the app.

---

## Project structure

```
ev_route_assistant/
├── app.py                     # Streamlit chat UI
├── agent/
│   ├── __init__.py
│   ├── config.py               # Settings (.env-driven, Pydantic Settings)
│   ├── models.py                # Pydantic models for stations, routes, tool I/O
│   ├── data_loader.py          # Loads + validates the JSON/CSV data sources
│   ├── tools.py                 # The 3 LangChain tools
│   └── graph.py                 # LangGraph agent (StateGraph, ReAct-style loop)
├── data/
│   ├── charging_stations.json
│   ├── route_elevation.csv
│   └── raw_api_schema.json     # original sample payload, kept for reference
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_data_loader.py
│   ├── test_tools.py
│   └── test_agent.py            # LLM is mocked - no API key or network needed
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

You can also paste the key directly into the Streamlit sidebar at runtime
instead of using `.env` — useful for quick demos.

### 3. Run the tests

```bash
pytest
```

All tool logic, data validation, and agent wiring are covered by unit
tests. The LLM itself is mocked in `test_agent.py`, so the full suite runs
offline with **no API key and no cost**.

### 4. Launch the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

---

## Example queries to try

* *"I'm driving from Chandigarh to Solan with 40% battery in a car with a
  50 kWh battery — will I make it, and where should I charge?"*
* *"Find me a fast DC charger near Solan under ₹20/kWh."*
* *"How much will it cost to charge from 20% to 80% on a 60 kWh battery at
  ₹18.5/kWh on a 50 kW charger?"*
* *"How much extra energy does the climb from Kalka to Dharampur cost
  compared to flat road?"*

A typical multi-tool exchange: the agent calls `estimate_route_energy` to
see the trip will drain more than 40% battery over the climb, calls
`search_charging_stations` to find a station along the way with an
available fast connector, then calls `calculate_charging_cost` to tell you
exactly how much it will cost and how long to stop for.

---

## Design notes / assumptions

* **Terrain model.** Energy use scales with average incline using a simple,
  clearly-labelled multiplier (steeper uphill = more energy; downhill
  recovers some energy via regenerative braking, floored so it's never
  "free"). This is a reasonable approximation for a route-planning
  assistant, not a certified physics simulator — the assumption is stated
  in the tool's own output (`assumptions` / multiplier fields) so the LLM
  can be transparent about it.
* **Charging time.** Assumes constant charging power for simplicity; the
  tool output explicitly notes that real DC fast-charging tapers above
  ~80% state of charge.
* **Location resolution.** `search_charging_stations` accepts either raw
  coordinates or a place name. Place names are resolved via the route
  dataset first (matching a town to its nearest station sector), then via
  a direct substring match against station/operator names.
* **Error handling.** Every tool validates its own arguments with Pydantic
  and returns a structured `{"error": "..."}` payload (rather than
  raising) when something can't be resolved, so the agent can read the
  reason and adapt (e.g. widen a search radius) instead of crashing the
  conversation. Data loading errors (missing/invalid files) raise a
  dedicated `DataLoadError` that the Streamlit UI catches and displays.
* **No hidden state.** `agent/config.py` re-reads settings on every call
  (rather than caching a global singleton), so the Streamlit sidebar can
  override the API key/model per session without restarting anything.

## Extending this project

* Swap `data/charging_stations.json` for a live call to the real
  [Open Charge Map API](https://openchargemap.org/site/develop/api) — the
  Pydantic `ChargingStation` model already matches its shape closely, so
  only `data_loader.load_stations` needs to point at an HTTP client.
* Add a fourth tool, e.g. `plan_multi_stop_route`, that chains
  `estimate_route_energy` and `search_charging_stations` automatically for
  routes longer than one battery charge.
* Persist chat history to disk/DB instead of `st.session_state` for a
  multi-user deployment.

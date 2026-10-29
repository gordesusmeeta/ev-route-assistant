from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agent import graph as graph_module
from agent.config import Settings
from agent.graph import AgentUnavailableError, EVRouteAgent


class _FakeLLM:
    """Stands in for ChatOpenAI so tests never hit the real OpenAI API."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.invocations = []

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        return self._responses.pop(0)


def test_missing_api_key_raises_agent_unavailable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(openai_api_key="")
    with pytest.raises(AgentUnavailableError):
        EVRouteAgent(settings=settings)


def test_agent_returns_direct_answer_without_tool_calls(monkeypatch, settings):
    final_message = AIMessage(content="You have plenty of range for that trip.")
    fake_llm = _FakeLLM([final_message])
    monkeypatch.setattr(graph_module, "ChatOpenAI", lambda **kwargs: fake_llm)

    agent = EVRouteAgent(settings=settings)
    reply, messages = agent.run_and_get_reply("Can I make it 5km on 90% battery?")

    assert reply == "You have plenty of range for that trip."
    assert fake_llm.invocations, "LLM should have been called at least once"


def test_agent_executes_tool_calls_then_answers(monkeypatch, settings):
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate_charging_cost",
                "args": {
                    "battery_capacity_kwh": 50,
                    "current_charge_percent": 20,
                    "target_charge_percent": 80,
                    "price_per_kwh_inr": 18.5,
                },
                "id": "call_1",
            }
        ],
    )
    final_message = AIMessage(content="It will cost about 555 INR to charge from 20% to 80%.")
    fake_llm = _FakeLLM([tool_call_message, final_message])
    monkeypatch.setattr(graph_module, "ChatOpenAI", lambda **kwargs: fake_llm)

    agent = EVRouteAgent(settings=settings)
    reply, messages = agent.run_and_get_reply("What would it cost to charge my 50kWh car from 20 to 80 percent?")

    assert "555" in reply
    tool_messages = [m for m in messages if m.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "calculate_charging_cost"

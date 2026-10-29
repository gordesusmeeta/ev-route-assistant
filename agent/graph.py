"""
LangGraph agent definition.

A small ReAct-style graph:

    START -> agent -> (tools -> agent)* -> END

The `agent` node calls the OpenAI chat model (with tools bound via function
calling). If the model responds with tool calls, control passes to the
`tools` node, which executes them and feeds results back to the agent. The
loop ends once the model responds without further tool calls, or after
`max_tool_iterations` safety cap is hit.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import Settings, get_settings
from agent.tools import build_tools

SYSTEM_PROMPT = """\
You are the EV Charging Route Assistant, an expert co-pilot for electric \
vehicle road trips in India.

You have three tools:
- estimate_route_energy: use this FIRST whenever the user mentions a route, \
hills/mountains, or a specific stretch of road, to work out how much energy \
the trip will actually take (terrain matters a lot for EV range).
- search_charging_stations: use this to find real charging stations near a \
point or place name, filtered by connector type, power, price, and \
availability.
- calculate_charging_cost: use this to work out the INR cost and time to \
charge from one battery percentage to another at a given station.

Guidelines:
- Always ground numeric claims (range, cost, time) in tool outputs — do not \
guess numbers you could compute.
- If the user gives a current battery percentage and a route, check whether \
they can make it, and if not, recommend a specific station (with distance, \
connector type, power, and price) and how long/what it will cost to charge \
enough to continue safely.
- If a tool call fails or returns no results, explain what went wrong in \
plain language and suggest a next step (e.g. widen the search radius).
- Be concise, concrete, and use the actual numbers returned by the tools. \
Prices are in INR, distances in km, energy in kWh.
"""


class AgentUnavailableError(RuntimeError):
    """Raised when the agent cannot be constructed (e.g. missing API key)."""


class EVRouteAgent:
    """Thin wrapper around a compiled LangGraph app."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise AgentUnavailableError(
                "No OpenAI API key configured. Set OPENAI_API_KEY in your .env "
                "file or enter it in the Streamlit sidebar."
            )

        self.tools = build_tools(self.settings)
        llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=self.settings.openai_temperature,
            api_key=self.settings.openai_api_key,
        )
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    # -- graph construction -------------------------------------------------
    def _build_graph(self):
        tool_node = ToolNode(self.tools)

        def agent_node(state: MessagesState) -> dict:
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: MessagesState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    # -- public API -----------------------------------------------------------
    def run(self, user_message: str, history: Optional[list] = None) -> dict:
        """Run one turn of the agent.

        `history` is an optional list of LangChain BaseMessage objects from
        prior turns (for multi-turn chat in the Streamlit app).
        """
        from langchain_core.messages import HumanMessage

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=user_message))

        config = {"recursion_limit": self.settings.max_tool_iterations * 2 + 2}
        result = self.graph.invoke({"messages": messages}, config=config)
        return result

    def run_and_get_reply(self, user_message: str, history: Optional[list] = None) -> tuple[str, list]:
        """Convenience helper returning (final_text_reply, full_message_list)."""
        result = self.run(user_message, history=history)
        all_messages = result["messages"]
        final = all_messages[-1]
        reply_text = final.content if isinstance(final.content, str) else str(final.content)
        return reply_text, all_messages

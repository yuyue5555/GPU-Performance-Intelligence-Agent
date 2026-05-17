"""
agent/graph.py

LangGraph ReAct agent with MCP-style tool bindings.
The agent can reason over GPU benchmark data using tool calls in a loop.

Graph flow:
  START → agent_node → (tool_call?) → tool_node → agent_node → ... → END
"""
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.prompts import SYSTEM_PROMPT
from mcp_tools.benchmark_tools import ALL_TOOLS


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_agent():
    """Construct and compile the LangGraph ReAct agent."""

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> AgentState:
        """Core reasoning node: calls LLM with tool access."""
        messages = state["messages"]

        # Prepend system message if this is the first turn
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Route: continue to tools if LLM requested tool calls, else end."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# Singleton for reuse across API and CLI
agent = build_agent()


def run_agent(user_query: str, history: list[dict] | None = None) -> str:
    """
    Run the agent with a user query.
    
    Args:
        user_query: Natural language question
        history: Optional list of prior messages [{"role": "user"|"assistant", "content": str}]
    
    Returns:
        Agent's final response as a string.
    """
    messages = []
    for msg in (history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_query))

    result = agent.invoke({"messages": messages})
    final = result["messages"][-1]
    return final.content

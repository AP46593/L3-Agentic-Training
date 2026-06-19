"""
pyt5.py - Multi-agent orchestrator built with LangGraph StateGraph.
Explicit graph-based routing with full control over agent flow.
Agents: Weather, Calculator, Stock, Tavily Web Search.
Traced with Opik for observability.
"""

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from opik.integrations.langchain import OpikTracer

# Import tools
from tools.agent_tools import ask_weather_agent, ask_calc_agent, ask_stock_agent
from tools.tavily_search import tavily_search
from langchain_core.tools import tool


# --- Tavily Agent Tool (inline for orchestrator) ---
@tool
def ask_tavily_search_agent(query: str) -> str:
    """Delegate questions requiring live internet search to the Tavily web search agent.
    Use this when the user asks about current events, latest news, recent updates,
    or any information that requires up-to-date internet search.
    Pass the user's full question as the query.
    """
    from agents.tavily_search_agent import run as tavily_run
    print(f"\n  [Tavily Search Agent] Received: {query}")
    result = tavily_run(query)
    print(f"  [Tavily Search Agent] Returning: {result}")
    return result


# --- Configuration ---
MODEL = "gpt-oss:120b-cloud"
MAX_TOKENS = 500
TEMPERATURE = 0.7
SYSTEM_MESSAGE = (
    "You are a helpful orchestrator assistant. "
    "You can answer general questions directly. "
    "For weather-related questions, use the ask_weather_agent tool. "
    "For math calculations, use the ask_calc_agent tool. "
    "For stock prices, first try ask_stock_agent. If not found, use ask_tavily_search_agent. "
    "For current events, news, or live information, use ask_tavily_search_agent. "
    "Keep responses concise, under 500 characters. No markdown formatting."
)

# --- All available tools ---
tools = [ask_weather_agent, ask_calc_agent, ask_stock_agent, ask_tavily_search_agent]


# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --- LLM with tools bound ---
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS
).bind_tools(tools)


# --- Graph Nodes ---
def agent_node(state: AgentState) -> AgentState:
    """The main agent node - calls the LLM with tools bound."""
    messages = state["messages"]
    # Inject system message if not already present
    if not messages or messages[0].type != "system":
        messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}


# Tool node handles tool execution
tool_node = ToolNode(tools)


# --- Router: should we call a tool or end? ---
def should_continue(state: AgentState) -> str:
    """Decide whether to route to tools or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# --- Build the Graph ---
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # After tool execution, go back to agent

# Compile
orchestrator = graph.compile()

# --- Opik Tracer ---
opik_tracer = OpikTracer()


# --- Conversation Memory ---
conversation_history = []


def chat(user_input: str):
    print(f"\n{'='*60}")
    print(f"[User] {user_input}")
    print(f"{'='*60}")

    # Add user message to history
    conversation_history.append(HumanMessage(content=user_input))

    result = orchestrator.invoke(
        {"messages": conversation_history},
        config={"callbacks": [opik_tracer]}
    )

    # Find the last AI message with content (final answer)
    final_answer_idx = None
    for i in range(len(result["messages"]) - 1, -1, -1):
        msg = result["messages"][i]
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            final_answer_idx = i
            break

    for i, msg in enumerate(result["messages"]):
        if msg.type == "human" or msg.type == "system":
            continue
        elif msg.type == "ai" and msg.tool_calls:
            print(f"\n[Orchestrator → Delegating]")
            for tc in msg.tool_calls:
                print(f"  Tool: {tc['name']}")
                print(f"  Args: {tc['args']}")
        elif msg.type == "tool":
            print(f"\n[Agent Response ← {msg.name}]")
            print(f"  {msg.content[:300]}")
        elif msg.type == "ai" and msg.content:
            if i == final_answer_idx:
                print(f"\n[Orchestrator → Final Answer]")
                print(f"  {msg.content}")
                # Add assistant response to conversation history
                conversation_history.append(AIMessage(content=msg.content))

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import sys

    # Print graph and exit if --graph flag is passed
    if "--graph" in sys.argv:
        print(orchestrator.get_graph().draw_mermaid())
        sys.exit(0)

    print("=== LangGraph Orchestrator Agent ===")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print("Agents: Weather, Calculator, Stock, Web Search (Tavily)")
    print("Type 'quit' to exit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not prompt:
            continue
        chat(prompt)
        print()

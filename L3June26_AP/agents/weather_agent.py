"""
weather_agent.py - Weather specialist agent.
Callable by the orchestrator to handle weather-related queries.
Uses get_weather tool to fetch real weather data from Open-Meteo.
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools import get_weather

# --- Configuration ---
MODEL = "gpt-oss:120b-cloud"
MAX_TOKENS = 500
TEMPERATURE = 0.7
SYSTEM_MESSAGE = (
    "You are a weather specialist assistant. "
    "Use the get_weather tool to fetch current weather for any city the user asks about. "
    "If the city is not found, tell the user directly and ask for a valid city name."
)

# --- Agent Setup ---
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=SYSTEM_MESSAGE
)


def run(query: str) -> str:
    """Run the weather agent and return the final response as a string."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # Log internal tool calls
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    [Weather Agent → Tool Call] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"    [Weather Agent ← Tool Result] {msg.content}")
    # Return final AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content
    return "Weather agent could not generate a response."

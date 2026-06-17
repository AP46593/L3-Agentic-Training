"""
pyt3.py - Multi-agent orchestrator.
Primary chat agent that delegates to specialist agents:
  - Weather Agent (pyt2) for weather-related queries
More agents can be added as tools over time.
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from tools.agent_tools import ask_weather_agent

# --- Configuration ---
MODEL = "gpt-oss:120b-cloud"
MAX_TOKENS = 500
TEMPERATURE = 0.7
SYSTEM_MESSAGE = (
    "You are a helpful orchestrator assistant. "
    "You can answer general questions directly. "
    "For weather-related questions, delegate to the weather agent using the ask_weather_agent tool."
)


# --- Orchestrator Agent Setup ---
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS
)

orchestrator = create_react_agent(
    model=llm,
    tools=[ask_weather_agent],
    prompt=SYSTEM_MESSAGE
)


def chat(user_input: str):
    result = orchestrator.invoke({"messages": [{"role": "user", "content": user_input}]})
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n[Delegating] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"[Agent Response] {tc['name']}: {msg.content}")
        elif msg.type == "ai" and msg.content:
            print(f"\nAssistant: {msg.content}")


if __name__ == "__main__":
    print("=== Orchestrator Agent ===")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print("I can chat or delegate to specialist agents.")
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

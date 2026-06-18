"""
pyt3.py - Multi-agent orchestrator.
Primary chat agent that delegates to specialist agents:
  - Weather Agent (pyt2) for weather-related queries
More agents can be added as tools over time.
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools.agent_tools import ask_weather_agent, ask_calc_agent, ask_stock_agent

# --- Configuration ---
MODEL = "gpt-oss:120b-cloud"
MAX_TOKENS = 500
TEMPERATURE = 0.7
SYSTEM_MESSAGE = (
    "You are a helpful orchestrator assistant. "
    "You can answer general questions directly. "
    "For weather-related questions, delegate to the weather agent using the ask_weather_agent tool. "
    "For math calculations, delegate to the calculator agent using the ask_calc_agent tool. "
    "For stock prices or company share information, delegate to the stock agent using the ask_stock_agent tool."
)


# --- Orchestrator Agent Setup ---
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS
)

orchestrator = create_agent(
    model=llm,
    tools=[ask_weather_agent, ask_calc_agent, ask_stock_agent],
    system_prompt=SYSTEM_MESSAGE
)


def chat(user_input: str):
    print(f"\n{'='*60}")
    print(f"[User] {user_input}")
    print(f"{'='*60}")

    result = orchestrator.invoke({"messages": [{"role": "user", "content": user_input}]})

    for i, msg in enumerate(result["messages"]):
        if msg.type == "human":
            continue  # Already printed above
        elif msg.type == "ai" and msg.tool_calls:
            print(f"\n[Orchestrator → Delegating]")
            for tc in msg.tool_calls:
                print(f"  Tool: {tc['name']}")
                print(f"  Args: {tc['args']}")
        elif msg.type == "tool":
            print(f"\n[Agent Response ← {msg.name}]")
            print(f"  {msg.content}")
        elif msg.type == "ai" and msg.content:
            print(f"\n[Orchestrator → Final Answer]")
            print(f"  {msg.content}")

    print(f"\n{'='*60}")


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

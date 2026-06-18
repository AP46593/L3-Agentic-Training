"""
pyt2.py - LangGraph agent with a weather tool placeholder.
Uses create_react_agent from LangGraph with ChatOllama.
System message: 'You are a helpful Assistant.'
User provides a city to get weather for.
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools import get_weather

# --- Configuration ---
MODEL = "gpt-oss:120b-cloud"
MAX_TOKENS = 500
TEMPERATURE = 0.7
SYSTEM_MESSAGE = "You are a helpful Assistant. When a user asks about weather and the tool returns that the location could not be found, tell them directly that the city was not recognized and ask them to provide a valid city name."


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


def chat(user_input: str):
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    # Print all messages to show tool usage
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n[Tool Call] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"[Tool Result] {msg.content}")
        elif msg.type == "ai" and msg.content:
            print(f"\nAssistant: {msg.content}")


if __name__ == "__main__":
    print("=== Weather Agent ===")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print("Ask about the weather in any city. Type 'quit' to exit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not prompt:
            continue
        chat(prompt)
        print()

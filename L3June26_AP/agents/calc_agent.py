"""
calc_agent.py - Simple calculator specialist agent.
Handles basic math: add, subtract, multiply, divide, square, cube, sqrt, cbrt.
Declines unsupported complex calculations.
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools.calculator import simple_calc

# --- Configuration ---
MODEL = "gpt-oss:120b-cloud"
MAX_TOKENS = 500
TEMPERATURE = 0.0  # Deterministic for math
SYSTEM_MESSAGE = (
    "You are a simple calculator assistant. "
    "You can ONLY perform these operations using the simple_calc tool: "
    "add, subtract, multiply, divide, square, cube, square root, cube root. "
    "Always respond with just the plain result, no markdown formatting, no bold, no extra explanation. "
    "Example: '11 multiplied by 12 = 132' "
    "If the user asks for any calculation outside these operations "
    "(e.g., logarithms, trigonometry, integrals, derivatives, percentages, exponents beyond cube), "
    "respond: 'I don't currently have the capability for that calculation. "
    "I can help with: add, subtract, multiply, divide, square, cube, square root, and cube root.'"
)

# --- Agent Setup ---
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS
)

agent = create_agent(
    model=llm,
    tools=[simple_calc],
    system_prompt=SYSTEM_MESSAGE
)


def run(query: str) -> str:
    """Run the calc agent and return the final response as a string."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # Log internal tool calls
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    [Calc Agent → Tool Call] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"    [Calc Agent ← Tool Result] {msg.content}")
    # Return final AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content
    return "Calculator agent could not generate a response."

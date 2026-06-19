"""
tavily_search_agent.py - Web search specialist agent using Tavily.
Searches the internet for live, current information.
Summarizes results concisely within 500 characters.
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools.tavily_search import tavily_search

# --- Configuration ---
MODEL = "gpt-oss:120b-cloud"
MAX_TOKENS = 500
TEMPERATURE = 0.3
SYSTEM_MESSAGE = (
    "You are a web search specialist assistant. "
    "Use the tavily_search tool to search the internet for current, live information. "
    "After getting search results, summarize the key findings concisely. "
    "Keep your final response under 500 characters. "
    "Include relevant source info when possible. "
    "Respond in plain text, no markdown formatting."
)

# --- Agent Setup ---
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS
)

agent = create_agent(
    model=llm,
    tools=[tavily_search],
    system_prompt=SYSTEM_MESSAGE
)


def run(query: str) -> str:
    """Run the Tavily search agent and return the final response."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # Log internal tool calls
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    [Tavily Agent → Tool Call] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"    [Tavily Agent ← Tool Result] {msg.content[:200]}...")
    # Return final AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content[:500]
    return "Tavily search agent could not generate a response."

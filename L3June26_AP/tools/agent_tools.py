"""
agent_tools.py - Tool wrappers that delegate to specialist agents.
Import these into any orchestrator that needs to call sub-agents.
"""

from langchain_core.tools import tool


@tool
def ask_weather_agent(query: str) -> str:
    """Delegate weather-related questions to the weather specialist agent.
    Use this when the user asks about current weather, temperature, or conditions in a city.
    Pass the user's full question as the query.
    """
    from agents.weather_agent import run as weather_run
    return weather_run(query)

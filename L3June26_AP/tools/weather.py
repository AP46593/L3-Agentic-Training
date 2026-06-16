"""
weather.py - Tool to fetch current weather for a city using Open-Meteo API.
"""

import requests
import urllib3
from langchain_core.tools import tool

# Suppress SSL verification warnings (corporate proxy)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    try:
        # Step 1: Geocode city name to coordinates
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo = requests.get(geo_url, verify=False).json()

        if not geo.get("results"):
            return f"Could not find location: {city}"

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        # Step 2: Fetch current weather
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather = requests.get(weather_url, verify=False).json()
        cw = weather["current_weather"]

        return (
            f"Weather in {city}: {cw['temperature']}°C, "
            f"wind {cw['windspeed']} km/h, "
            f"weather code {cw['weathercode']}"
        )
    except Exception as e:
        return f"Error fetching weather for {city}: {e}"

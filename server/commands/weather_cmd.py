"""Weather command using Open-Meteo (free, no API key required).

Uses Open-Meteo geocoding API to resolve city names, then fetches
current conditions and today's high/low forecast.
https://open-meteo.com/
"""
import requests
from typing import Optional, Tuple

from .base import Command
from server.config import DEFAULT_LOCATION

# WMO weather interpretation codes → human-readable description
_WMO = {
    0:  "clear sky",
    1:  "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy",        48: "icy fog",
    51: "light drizzle", 53: "drizzle",     55: "heavy drizzle",
    61: "light rain",   63: "rain",          65: "heavy rain",
    71: "light snow",   73: "snow",          75: "heavy snow",
    77: "snow grains",
    80: "rain showers", 81: "showers",       82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _geocode(location: str) -> Optional[Tuple[float, float, str]]:
    """Return (lat, lon, display_name) for a location string, or None if not found."""
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        return None
    r = results[0]
    name = r.get("name", location)
    admin = r.get("admin1", "")
    country = r.get("country_code", "")
    display = f"{name}, {admin}" if admin else f"{name}, {country}"
    return r["latitude"], r["longitude"], display


class WeatherCommand(Command):
    name = "get_weather"
    description = (
        "Get current weather conditions and today's forecast for a location. "
        "No API key required."
    )

    @property
    def parameters(self) -> dict:
        return {
            "location": {
                "type": "string",
                "description": (
                    f"City name (e.g. 'Seattle' or 'London, UK'). "
                    f"Omit to use default ({DEFAULT_LOCATION or 'not configured'})."
                ),
            }
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, location: str = "", **_) -> str:
        location = location.strip() or DEFAULT_LOCATION
        if not location:
            return "No location specified and DEFAULT_LOCATION not set in config"
        if len(location) > 200:
            return "Location name too long"

        try:
            geo = _geocode(location)
            if not geo:
                return f"Location '{location}' not found"
            lat, lon, display_name = geo

            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                    "timezone": "auto",
                    "forecast_days": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            cw = data["current_weather"]
            temp = round(cw["temperature"])
            wind = round(cw["windspeed"])
            description = _WMO.get(cw.get("weathercode", 0), "unknown conditions")

            daily = data.get("daily", {})
            highs = daily.get("temperature_2m_max") or []
            lows  = daily.get("temperature_2m_min") or []
            precips = daily.get("precipitation_probability_max") or []
            high   = round(highs[0])   if highs   else None
            low    = round(lows[0])    if lows    else None
            precip = precips[0]        if precips else None

            result = f"{display_name}: {temp}°F, {description}, wind {wind} mph."
            if high is not None and low is not None:
                result += f" High {high}, low {low}."
            if precip is not None and precip > 20:
                result += f" {precip}% chance of precipitation."
            return result

        except requests.Timeout:
            return "Weather unavailable: request timed out"
        except requests.RequestException as e:
            return f"Weather unavailable: {e}"

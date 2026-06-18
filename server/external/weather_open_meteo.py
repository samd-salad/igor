"""Open-Meteo weather lookup. No API key required. Returns spoken-friendly
strings — never structured data — because the consumer is a voice tool."""
from __future__ import annotations
import logging

import requests

logger = logging.getLogger(__name__)


# https://open-meteo.com/en/docs#weathervariables
_WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
}


def _c_to_f(c: float) -> int:
    return round(c * 9 / 5 + 32)


class OpenMeteoWeather:
    def current(self, location: str) -> str:
        try:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
                timeout=5,
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                return f"I couldn't find {location}."
            place = results[0]
            lat, lon, name = place["latitude"], place["longitude"], place["name"]
            forecast = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon,
                        "current": "temperature_2m,weather_code"},
                timeout=5,
            )
            forecast.raise_for_status()
            cur = forecast.json().get("current", {})
            temp_c = cur.get("temperature_2m")
            code = cur.get("weather_code", 0)
            condition = _WEATHER_CODES.get(code, "unknown")
            return f"{_c_to_f(temp_c)}°F and {condition} in {name}."
        except Exception:
            logger.exception("Weather lookup failed for %s", location)
            return f"Weather lookup failed for {location}."

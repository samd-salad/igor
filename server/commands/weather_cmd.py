"""Weather command using OpenWeatherMap API."""
import os
import requests
from .base import Command

API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")
DEFAULT_LOCATION = os.environ.get("DEFAULT_LOCATION", "")


class WeatherCommand(Command):
    name = "get_weather"
    description = "Get current weather conditions and forecast for a location"

    @property
    def parameters(self) -> dict:
        return {
            "location": {
                "type": "string",
                "description": f"City name (e.g., 'Seattle' or 'London, UK'). Defaults to '{DEFAULT_LOCATION}' if not specified."
            }
        }

    def execute(self, location: str = "") -> str:
        location = location.strip() or DEFAULT_LOCATION

        if not API_KEY:
            return "Weather unavailable: OPENWEATHERMAP_API_KEY not set"

        if not location:
            return "No location specified and no default location configured"

        try:
            # Get current weather
            current_url = "https://api.openweathermap.org/data/2.5/weather"
            current_resp = requests.get(current_url, params={
                "q": location,
                "appid": API_KEY,
                "units": "imperial"
            }, timeout=10)

            if current_resp.status_code == 401:
                return "Weather unavailable: Invalid API key"
            if current_resp.status_code == 404:
                return f"Location '{location}' not found"
            current_resp.raise_for_status()

            current = current_resp.json()

            # Extract current conditions
            temp = round(current["main"]["temp"])
            feels_like = round(current["main"]["feels_like"])
            humidity = current["main"]["humidity"]
            description = current["weather"][0]["description"]
            city_name = current["name"]

            # Get forecast
            forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
            forecast_resp = requests.get(forecast_url, params={
                "q": location,
                "appid": API_KEY,
                "units": "imperial",
                "cnt": 8  # Next 24 hours (3-hour intervals)
            }, timeout=10)

            forecast_summary = ""
            if forecast_resp.status_code == 200:
                forecast = forecast_resp.json()
                # Get high/low for next 24 hours
                temps = [item["main"]["temp"] for item in forecast["list"]]
                high = round(max(temps))
                low = round(min(temps))
                forecast_summary = f" High {high}F, low {low}F today."

            result = f"{city_name}: {temp}F (feels like {feels_like}F), {description}, {humidity}% humidity.{forecast_summary}"
            return result

        except requests.Timeout:
            return "Weather unavailable: Request timed out"
        except requests.RequestException as e:
            return f"Weather unavailable: {e}"

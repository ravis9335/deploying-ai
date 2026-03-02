"""
Service 1: Weather API
Uses the free Open-Meteo API (no key required) with geocoding to fetch
real-time weather data for any city worldwide.
"""

import json
import requests


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _geocode_city(city: str, country_code: str = None) -> dict:
    """Look up latitude/longitude for a city using Open-Meteo geocoding."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    if country_code:
        params["country_code"] = country_code.upper()

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = data.get("results")
    if not results:
        return None
    return results[0]


def _fetch_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather from Open-Meteo for given coordinates."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "weather_code",
            "wind_speed_10m",
            "precipitation",
        ],
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def get_weather_info(city: str, country_code: str = None) -> str:
    """
    Fetch current weather for a city and return as a JSON string.
    This is the tool function exposed via function calling.
    """
    try:
        location = _geocode_city(city, country_code)
        if not location:
            return json.dumps({
                "error": f"Could not find location data for '{city}'. "
                         "Please check the city name and try again."
            })

        weather = _fetch_weather(location["latitude"], location["longitude"])
        current = weather.get("current", {})

        temp_c = current.get("temperature_2m")
        feels_c = current.get("apparent_temperature")
        code = current.get("weather_code", 0)

        result = {
            "location": f"{location.get('name', city)}, {location.get('country', '')}",
            "condition": WEATHER_CODES.get(code, "Unknown"),
            "temperature_celsius": temp_c,
            "temperature_fahrenheit": round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None,
            "feels_like_celsius": feels_c,
            "feels_like_fahrenheit": round(feels_c * 9 / 5 + 32, 1) if feels_c is not None else None,
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "precipitation_mm": current.get("precipitation", 0),
            "timezone": weather.get("timezone", ""),
        }
        return json.dumps(result)

    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "Unable to reach the weather service. Please check your internet connection."})
    except requests.exceptions.Timeout:
        return json.dumps({"error": "The weather service timed out. Please try again."})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error fetching weather: {str(e)}"})

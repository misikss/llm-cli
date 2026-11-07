"""
tools.py — Herramientas disponibles para Gemini (Clima en tiempo real y utilidades).
"""

import json
import logging
import urllib.parse
import urllib.request


logger = logging.getLogger(__name__)


def get_current_weather(city: str) -> str:
    """
    Obtiene el clima actual y en tiempo real de cualquier ciudad del mundo.
    Consulta la API gratuita de Open-Meteo mediante geocodificación automática.
    
    Args:
        city: Nombre de la ciudad o localidad (ej. 'Lima', 'Madrid', 'Ciudad de México').
    """
    try:
        # 1. Geocodificación para obtener latitud y longitud
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={urllib.parse.quote(city)}&count=1&language=es&format=json"
        )
        req = urllib.request.Request(geo_url, headers={"User-Agent": "LLM-CLI/1.0"})
        with urllib.request.urlopen(req, timeout=6) as res:
            geo_data = json.loads(res.read().decode("utf-8"))

        if not geo_data.get("results"):
            return f"No se encontró la ubicación de la ciudad '{city}'."

        loc = geo_data["results"][0]
        name = loc.get("name", city)
        country = loc.get("country", "")
        lat = loc["latitude"]
        lon = loc["longitude"]

        # 2. Consulta de clima actual
        w_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        )
        req_w = urllib.request.Request(w_url, headers={"User-Agent": "LLM-CLI/1.0"})
        with urllib.request.urlopen(req_w, timeout=6) as res_w:
            w_data = json.loads(res_w.read().decode("utf-8"))

        curr = w_data.get("current", {})
        temp = curr.get("temperature_2m", "N/A")
        humidity = curr.get("relative_humidity_2m", "N/A")
        wind = curr.get("wind_speed_10m", "N/A")

        return (
            f"Clima en tiempo real para {name}, {country}:\n"
            f"- Temperatura actual: {temp} °C\n"
            f"- Humedad relativa: {humidity}%\n"
            f"- Velocidad del viento: {wind} km/h"
        )
    except Exception as exc:
        logger.warning(
            "weather_lookup_failed error_type=%s",
            type(exc).__name__,
        )
        return f"No se pudo consultar el clima para {city}. Intenta nuevamente."


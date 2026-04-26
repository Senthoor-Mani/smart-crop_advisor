import os
import requests

BASE = "https://api.weatherapi.com/v1"

class WeatherAPIError(Exception):
    pass

def _key():
    k = os.getenv("WEATHERAPI_KEY", "").strip()
    if not k:
        raise WeatherAPIError("WEATHERAPI_KEY missing. Set it as an environment variable.")
    return k

def search_city(query: str, limit: int = 8):
    k = _key()
    url = f"{BASE}/search.json"
    params = {"key": k, "q": query}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise WeatherAPIError(f"Search failed ({r.status_code}): {r.text[:200]}")
    return r.json()[:limit]

def forecast(city: str, days: int = 14):
    k = _key()
    url = f"{BASE}/forecast.json"
    params = {"key": k, "q": city, "days": days, "aqi": "no", "alerts": "no"}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise WeatherAPIError(f"Forecast failed ({r.status_code}): {r.text[:200]}")
    return r.json()

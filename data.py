import os
import hashlib
import requests
import pandas as pd
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_URL = "https://api.open-meteo.com/v1"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1"

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "weather_code", "cloud_cover",
    "pressure_msl", "wind_speed_10m"
]

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "wind_speed_10m_max", "weather_code"
]


def _cache_key(lat, lon, start, end, freq):
    raw = f"{lat}_{lon}_{start}_{end}_{freq}"
    return hashlib.md5(raw.encode()).hexdigest() + ".parquet"


def _check_cache(key):
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def _save_cache(key, df):
    path = os.path.join(CACHE_DIR, key)
    df.to_parquet(path)


def fetch_forecast(lat, lon, days=7):
    """Fetch weather forecast from Open-Meteo (up to 16 days)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": days,
        "timezone": "auto"
    }
    resp = requests.get(f"{BASE_URL}/forecast", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


def fetch_current(lat, lon):
    """Fetch current weather conditions."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "weather_code", "cloud_cover", "pressure_msl", "wind_speed_10m"
        ]),
        "timezone": "auto"
    }
    resp = requests.get(f"{BASE_URL}/forecast", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["current"]


def fetch_history(lat, lon, start_date, end_date, freq="hourly"):
    """Fetch historical weather data. Caches results as parquet.

    Args:
        lat, lon: coordinates
        start_date, end_date: 'YYYY-MM-DD' strings
        freq: 'hourly' or 'daily'
    """
    key = _cache_key(lat, lon, start_date, end_date, freq)
    cached = _check_cache(key)
    if cached is not None:
        return cached

    variables = HOURLY_VARS if freq == "hourly" else DAILY_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
        freq: ",".join(variables),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto"
    }
    resp = requests.get(f"{ARCHIVE_URL}/archive", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data[freq])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    _save_cache(key, df)
    return df


def fetch_history_chunked(lat, lon, start_date, end_date, freq="hourly", chunk_months=3):
    """Fetch historical data in chunks to avoid API limits."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []

    current = start
    while current < end:
        chunk_end = min(current + pd.DateOffset(months=chunk_months) - timedelta(days=1), end)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        df = fetch_history(lat, lon, chunk_start_str, chunk_end_str, freq)
        chunks.append(df)
        current = chunk_end + timedelta(days=1)

    return pd.concat(chunks).sort_index()


# WMO weather code mapping to 6 classes
WMO_TO_CLASS = {}
for code in [0]:
    WMO_TO_CLASS[code] = 0  # Clear
for code in [1, 2, 3]:
    WMO_TO_CLASS[code] = 1  # Cloudy
for code in [45, 48]:
    WMO_TO_CLASS[code] = 2  # Fog
for code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
    WMO_TO_CLASS[code] = 3  # Rain
for code in [71, 73, 75, 77, 85, 86]:
    WMO_TO_CLASS[code] = 4  # Snow
for code in [95, 96, 99]:
    WMO_TO_CLASS[code] = 5  # Thunderstorm

CLASS_NAMES = ["Clear", "Cloudy", "Fog", "Rain", "Snow", "Thunderstorm"]


def weather_class(code):
    return WMO_TO_CLASS.get(code, 1)  # default to Cloudy if unknown

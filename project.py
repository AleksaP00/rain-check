"""RainCheck: predict whether it will rain tomorrow."""


import datetime
import re
import requests
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FEATURES = [
    "temperature_2m_max", 
    "temperature_2m_min", 
    "precipitation_sum", 
    "precipitation_hours", 
    "wind_speed_10m_max", 
    "sunshine_duration", 
    "relative_humidity_2m_mean", 
    "surface_pressure_mean", 
    "cloud_cover_mean",
]

class LocationNotFoundError(Exception):
    """City doesn't exist"""
    

def validate_date(s: str) -> str:
    """Return s unchanged if it is a real YYYY-MM-DD date, else raise ValueError."""

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise ValueError(f"Invalid date format {s}")

    datetime.date.fromisoformat(s)

    return s


def slugify(city: str) -> str:
    """Turn a city name into a file-friendly name, e.g. "Novi Sad" -> "novi_sad"."""

    city = re.sub(r"[^a-z0-9]+", "_", city.lower())

    city = city.strip("_")

    return city


def geocode(city: str) -> tuple[float, float, str]:
    """Return (latitude, longitude, timezone) of the first match for a city name."""
    
    response = requests.get(GEOCODING_URL, params = {"name": city, "count": 1}, timeout=10)
    response.raise_for_status()
    data = response.json()

    if "results" not in data:
        raise LocationNotFoundError(f"City not found: {city}")

    place = data['results'][0]
    return (place['latitude'], place['longitude'], place['timezone'])


def fetch_history(lat: float, lon: float, start: str, end: str) -> list[dict]:
    """Download daily weather history and return one dict per day."""

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(FEATURES),
        "timezone": "auto",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=10)
    response.raise_for_status()
    daily = response.json()["daily"]
    rows = []
    for i, date in enumerate(daily['time']):
        row = {"date": date}
        for feature in FEATURES:
            row[feature] = daily[feature][i]
        rows.append(row)

    return rows


def main() -> None:
    """Run the RainCheck program."""
    pass


if __name__ == "__main__":
    main()

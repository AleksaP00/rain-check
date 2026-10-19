"""RainCheck: predict whether it will rain tomorrow."""


import datetime
import re
import requests
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


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


def main() -> None:
    """Run the RainCheck program."""
    pass


if __name__ == "__main__":
    main()

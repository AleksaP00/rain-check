"""Data for RainCheck: input checks, Open-Meteo downloads, CSV files and the dataset."""

import csv
import datetime
import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

RAIN_THRESHOLD_MM = 1.0
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
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


class IncompleteDataError(Exception):
    """The weather service returned missing values."""


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

    response = requests.get(
        GEOCODING_URL, params={"name": city, "count": 1}, timeout=10
    )
    response.raise_for_status()
    data = response.json()

    if "results" not in data:
        raise LocationNotFoundError(f"City not found: {city}")

    place = data["results"][0]
    return (place["latitude"], place["longitude"], place["timezone"])


def daily_to_rows(daily: dict) -> list[dict]:
    """Turn the API's "daily" block (one list per variable) into one dict per day."""

    rows = []
    for i, date in enumerate(daily["time"]):
        row = {"date": date}
        for feature in FEATURES:
            row[feature] = daily[feature][i]
        rows.append(row)

    return rows


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
    return daily_to_rows(response.json()["daily"])


def fetch_today(lat: float, lon: float) -> dict:
    """Download today's forecast and return it as one cleaned row, like the history rows."""

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(FEATURES),
        "timezone": "auto",
        "forecast_days": 1,
    }
    response = requests.get(FORECAST_URL, params=params, timeout=10)
    response.raise_for_status()
    rows = clean_rows(daily_to_rows(response.json()["daily"]))

    if not rows:
        raise IncompleteDataError("Today's forecast has missing values")

    return rows[0]


def save_csv(rows: list[dict], path: str) -> None:
    """Write a list of dicts to a CSV file, creating the folder if needed."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: str) -> list[dict]:
    """Read a CSV file into a list of dicts (all values are strings)."""
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def location_path(csv_path: str) -> str:
    """Return the path of the JSON file with the city info that belongs to a CSV file."""
    return os.path.splitext(csv_path)[0] + ".json"


def clean_rows(rows: list[dict]) -> list[dict]:
    """Convert values to float and drop rows that are empty, invalid or impossible."""

    cleaned = []
    for row in rows:
        one_dic = {"date": row["date"]}
        ok = True
        for feature in FEATURES:
            try:
                one_dic[feature] = float(row[feature])
            except (ValueError, TypeError):
                ok = False

        if (
            ok == False
            or one_dic["precipitation_sum"] < 0
            or one_dic["precipitation_hours"] < 0
            or one_dic["relative_humidity_2m_mean"] < 0
            or one_dic["relative_humidity_2m_mean"] > 100
        ):
            ok = False

        if ok:
            cleaned.append(one_dic)

    logger.info("Dropped %d of %d rows", len(rows) - len(cleaned), len(rows))

    return cleaned


def make_examples(rows: list[dict]) -> tuple[list[list[float]], list[int], list[str]]:
    """Turn cleaned daily rows into (X, y, dates): today's features, tomorrow's rain label, today's date."""

    X = []
    y = []
    dates = []
    for i in range(len(rows) - 1):
        if datetime.date.fromisoformat(
            rows[i + 1]["date"]
        ) - datetime.date.fromisoformat(rows[i]["date"]) != datetime.timedelta(days=1):
            continue
        X.append([rows[i][feature] for feature in FEATURES])
        if rows[i + 1]["precipitation_sum"] >= RAIN_THRESHOLD_MM:
            y.append(1)
        else:
            y.append(0)
        dates.append(rows[i]["date"])

    if y:
        logger.info(
            "Rainy days: %.1f%% (%d of %d)", 100 * sum(y) / len(y), sum(y), len(y)
        )

    return X, y, dates


def time_split(
    X: list[list[float]], y: list[int], test_ratio: float = 0.2
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """Split examples by time: the first part is train, the last test_ratio part is test."""

    train_ratio = 1 - test_ratio
    number_of_train_examples = int(len(X) * train_ratio)

    (X_train, X_test, y_train, y_test) = (
        X[:number_of_train_examples],
        X[number_of_train_examples:],
        y[:number_of_train_examples],
        y[number_of_train_examples:],
    )

    return (X_train, X_test, y_train, y_test)


def load_split(path: str) -> tuple:
    """Read a CSV and return the time split as a dict.

    Keys: "train" and "test", each a tuple (X, y, dates).
    """

    rows = clean_rows(load_csv(path))
    X, y, dates = make_examples(rows)
    X_train, X_test, y_train, y_test = time_split(X, y)
    n_train = len(X_train)
    return {
        "train": (X_train, y_train, dates[:n_train]),
        "test": (X_test, y_test, dates[n_train:]),
    }

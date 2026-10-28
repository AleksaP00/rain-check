"""RainCheck: predict whether it will rain tomorrow."""


import datetime
import re
import requests
import os
import csv
import logging
import json
import argparse
import sys

from model import StandardScaler, KNNClassifier

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
    
    response = requests.get(GEOCODING_URL, params = {"name": city, "count": 1}, timeout=10)
    response.raise_for_status()
    data = response.json()

    if "results" not in data:
        raise LocationNotFoundError(f"City not found: {city}")

    place = data['results'][0]
    return (place['latitude'], place['longitude'], place['timezone'])


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

        if ok == False or one_dic["precipitation_sum"] < 0 or one_dic["precipitation_hours"] < 0 or one_dic["relative_humidity_2m_mean"] < 0 or one_dic["relative_humidity_2m_mean"] > 100:
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
        if datetime.date.fromisoformat(rows[i + 1]["date"])  - datetime.date.fromisoformat(rows[i]["date"]) != datetime.timedelta(days=1):
            continue
        X.append([rows[i][feature] for feature in FEATURES])
        if rows[i + 1]["precipitation_sum"] >= RAIN_THRESHOLD_MM:
            y.append(1)
        else:
            y.append(0)
        dates.append(rows[i]["date"])
    
    if y:
        logger.info("Rainy days: %.1f%% (%d of %d)", 100 * sum(y) / len(y), sum(y), len(y))

    return X, y, dates


def time_split(
    X: list[list[float]], y: list[int], test_ratio: float = 0.2
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """Split examples by time: the first part is train, the last test_ratio part is test."""

    train_ratio = 1 - test_ratio
    number_of_train_examples = int(len(X) * train_ratio)

    (X_train, X_test, y_train, y_test) = (X[:number_of_train_examples], X[number_of_train_examples:], y[:number_of_train_examples], y[number_of_train_examples:])

    return (X_train, X_test, y_train, y_test)


def confusion_counts(y_true: list[int], y_pred: list[int]) -> dict:
    """Count true/false positives and negatives for binary labels."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            counts["tp"] += 1
        elif actual == 0 and predicted == 1:
            counts["fp"] += 1
        elif actual == 0 and predicted == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1

    return counts


def classification_report(y_true: list[int], y_pred: list[int]) -> dict:
    """Return accuracy, precision, recall and F1; 0.0 when a division by zero."""
    counts = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def baseline_predictions(
    X_test_raw: list[list[float]], y_train: list[int]
) -> dict[str, list[int]]:
    """Return predictions of two simple baselines for every test example."""
    majority = 1 if sum(y_train) > len(y_train) / 2 else 0

    rain_column = FEATURES.index("precipitation_sum")
    persistence = [1 if row[rain_column] >= RAIN_THRESHOLD_MM else 0 for row in X_test_raw]

    return {"majority": [majority] * len(X_test_raw), "persistence": persistence}


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


def save_model(
    path: str, scaler: StandardScaler, model: KNNClassifier, metadata: dict
) -> None:
    """Save scaler, model and metadata into one JSON file."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    data = {"scaler": scaler.to_dict(), "model": model.to_dict(), "metadata": metadata}
    with open(path, "w") as file:
        json.dump(data, file)


def load_model(path: str) -> tuple[StandardScaler, KNNClassifier, dict]:
    """Load (scaler, model, metadata) from a JSON file made by save_model."""
    with open(path) as file:
        data = json.load(file)

    scaler = StandardScaler.from_dict(data["scaler"])
    model = KNNClassifier.from_dict(data["model"])
    return scaler, model, data["metadata"]


def location_path(csv_path: str) -> str:
    """Return the path of the JSON file with the city info that belongs to a CSV file."""
    return os.path.splitext(csv_path)[0] + ".json"


def cmd_fetch(args: argparse.Namespace) -> None:
    """fetch: find the city, download its history and save it as CSV plus city info."""

    lat, lon, timezone = geocode(args.city)
    logger.info("Found %s at (%.2f, %.2f)", args.city, lat, lon)

    rows = fetch_history(lat, lon, args.start, args.end)
    csv_path = os.path.join("data", slugify(args.city) + ".csv")
    save_csv(rows, csv_path)

    location = {
        "city": args.city,
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
    }
    with open(location_path(csv_path), "w") as file:
        json.dump(location, file)

    logger.info("Saved %d days to %s", len(rows), csv_path)


def load_split(path: str) -> tuple:
    """Read a CSV and return (X_train, X_test, y_train, y_test, dates), split by time."""

    rows = clean_rows(load_csv(path))
    X, y, dates = make_examples(rows)
    X_train, X_test, y_train, y_test = time_split(X, y)
    return X_train, X_test, y_train, y_test, dates


def evaluate_model(
    scaler: StandardScaler,
    model: KNNClassifier,
    X_test_raw: list[list[float]],
    y_test: list[int],
    y_train: list[int],
) -> dict[str, dict]:
    """Return a classification report for k-NN and for both baselines."""

    predictions = baseline_predictions(X_test_raw, y_train)
    predictions["k-NN"] = model.predict(scaler.transform(X_test_raw))
    return {name: classification_report(y_test, y_pred) for name, y_pred in predictions.items()}


def format_table(results: dict[str, dict]) -> str:
    """Format the reports as a text table, one row per predictor."""

    lines = [f"{'':12}{'accuracy':>10}{'precision':>10}{'recall':>10}{'f1':>10}"]
    for name in ["k-NN", "persistence", "majority"]:
        r = results[name]
        lines.append(
            f"{name:12}{r['accuracy']:>10.3f}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>10.3f}"
        )
    return "\n".join(lines)


def cmd_train(args: argparse.Namespace) -> None:
    """train: split by time, fit scaler and k-NN on the train part, save the model."""

    X_train, X_test, y_train, y_test, dates = load_split(args.data)

    # The scaler learns its mean and stdev from the train part only (no data leakage).
    scaler = StandardScaler()
    scaler.fit(X_train)
    model = KNNClassifier(args.k)
    model.fit(scaler.transform(X_train), y_train)

    results = evaluate_model(scaler, model, X_test, y_test, y_train)

    with open(location_path(args.data)) as file:
        metadata = json.load(file)
    metadata.update({
        "features": FEATURES,
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "date_range": [dates[0], dates[-1]],
        "metrics": results,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    save_model(args.model, scaler, model, metadata)

    logger.info("Model saved to %s", args.model)
    print(format_table(results))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """evaluate: compare the saved model with both baselines on the test part."""

    scaler, model, metadata = load_model(args.model)
    _, X_test, y_train, y_test, _ = load_split(args.data)

    results = evaluate_model(scaler, model, X_test, y_test, y_train)
    print(format_table(results))


def main() -> None:
    """Run the RainCheck program."""
    pass


if __name__ == "__main__":
    main()

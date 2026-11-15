"""RainCheck: predict whether it will rain tomorrow (command line interface)."""

import argparse
import datetime
import json
import logging
import os
import sys

import requests

from raincheck.data_handler import (
    FEATURES,
    RAIN_THRESHOLD_MM,
    IncompleteDataError,
    LocationNotFoundError,
    fetch_history,
    fetch_today,
    geocode,
    load_split,
    location_path,
    save_csv,
    slugify,
    validate_date,
)
from raincheck.evaluation_metrics import evaluate_model
from raincheck.model import KNNClassifier, StandardScaler

logger = logging.getLogger(__name__)


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


def format_table(results: dict[str, dict]) -> str:
    """Format the reports as a text table, one row per predictor."""

    lines = [f"{'':12}{'accuracy':>10}{'precision':>10}{'recall':>10}{'f1':>10}"]
    for name in ["k-NN", "persistence", "majority"]:
        r = results[name]
        lines.append(
            f"{name:12}{r['accuracy']:>10.3f}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>10.3f}"
        )
    return "\n".join(lines)


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
    metadata.update(
        {
            "features": FEATURES,
            "rain_threshold_mm": RAIN_THRESHOLD_MM,
            "date_range": [dates[0], dates[-1]],
            "metrics": results,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_model(args.model, scaler, model, metadata)

    logger.info("Model saved to %s", args.model)
    print(format_table(results))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """evaluate: compare the saved model with both baselines on the test part."""

    scaler, model, _ = load_model(args.model)
    _, X_test, y_train, y_test, _ = load_split(args.data)

    results = evaluate_model(scaler, model, X_test, y_test, y_train)
    print(format_table(results))


def cmd_predict(args: argparse.Namespace) -> None:
    """predict: get today's forecast for the model's city and predict tomorrow."""

    scaler, model, metadata = load_model(args.model)
    today = fetch_today(metadata["latitude"], metadata["longitude"])
    logger.debug("Today's weather: %s", today)

    # Same feature order as in training, taken from the model's own metadata.
    x = scaler.transform([[today[name] for name in metadata["features"]]])[0]
    probability = model.predict_proba_one(x)

    label = "rain" if probability >= 0.5 else "no rain"
    print(f"Tomorrow in {metadata['city']}: {label} (p={probability:.2f})")


def odd_int(text: str) -> int:
    """argparse type: a positive odd integer (same rule as KNNClassifier.k)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer")
    if value <= 0 or value % 2 == 0:
        raise argparse.ArgumentTypeError(
            f"k must be a positive odd integer, got {value}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser with the fetch, train, evaluate and predict commands."""

    # Options shared by every command live in one parent parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose", action="store_true", help="show debug messages"
    )

    parser = argparse.ArgumentParser(
        prog="raincheck", description="Predict whether it will rain tomorrow."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser(
        "fetch", parents=[common], help="download weather history"
    )
    fetch.add_argument("--city", required=True)
    fetch.add_argument("--start", required=True, type=validate_date, help="YYYY-MM-DD")
    fetch.add_argument("--end", required=True, type=validate_date, help="YYYY-MM-DD")
    fetch.set_defaults(func=cmd_fetch)

    train = commands.add_parser("train", parents=[common], help="train the k-NN model")
    train.add_argument("--data", required=True, help="CSV file made by fetch")
    train.add_argument("--k", type=odd_int, default=15)
    train.add_argument(
        "--model", default="models/model.json", help="where to save the model"
    )
    train.set_defaults(func=cmd_train)

    evaluate = commands.add_parser(
        "evaluate", parents=[common], help="compare the model with baselines"
    )
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", default="models/model.json")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = commands.add_parser("predict", parents=[common], help="predict tomorrow")
    predict.add_argument("--model", default="models/model.json")
    predict.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    """Parse the command line, set up logging and run the chosen command."""

    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        args.func(args)
    except (
        LocationNotFoundError,
        IncompleteDataError,
        requests.RequestException,
        FileNotFoundError,
    ) as error:
        logger.error("%s", error)
        sys.exit(1)

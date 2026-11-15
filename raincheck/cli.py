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
from raincheck.artifacts import load_artifacts, load_split_csv, save_artifacts
from raincheck.evaluation_metrics import evaluate_model
from raincheck.model import build_pipeline

logger = logging.getLogger(__name__)


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
    """train: split by time, fit the pipeline on the train part, save the model folder."""

    splits = load_split(args.data)
    X_train, y_train, dates_train = splits["train"]
    X_test, y_test, dates_test = splits["test"]

    # The pipeline's scaler learns mean and stdev from the train part only (no data leakage).
    pipeline = build_pipeline(args.k)
    pipeline.fit(X_train, y_train)

    results = evaluate_model(pipeline, X_test, y_test, y_train)

    with open(location_path(args.data)) as file:
        config = json.load(file)
    config.update(
        {
            "features": FEATURES,
            "rain_threshold_mm": RAIN_THRESHOLD_MM,
            "k": args.k,
            "train_dates": [dates_train[0], dates_train[-1]],
            "test_dates": [dates_test[0], dates_test[-1]],
            "metrics": results,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_artifacts(args.model, pipeline, config, splits)

    logger.info("Model saved to %s", args.model)
    print(format_table(results))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """evaluate: compare the saved model with both baselines on its saved test part."""

    pipeline, _ = load_artifacts(args.model)
    _, y_train, _ = load_split_csv(os.path.join(args.model, "train.csv"))
    X_test, y_test, _ = load_split_csv(os.path.join(args.model, "test.csv"))

    results = evaluate_model(pipeline, X_test, y_test, y_train)
    print(format_table(results))


def cmd_predict(args: argparse.Namespace) -> None:
    """predict: get today's forecast for the model's city and predict tomorrow."""

    pipeline, config = load_artifacts(args.model)
    today = fetch_today(config["latitude"], config["longitude"])
    logger.debug("Today's weather: %s", today)

    # Same feature order as in training, taken from the model's own config.
    x = [[today[name] for name in config["features"]]]
    # predict_proba gives one column per class; take the column of class 1 (rain).
    rain_column = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(x)[0][rain_column])

    label = "rain" if probability >= 0.5 else "no rain"
    print(f"Tomorrow in {config['city']}: {label} (p={probability:.2f})")


def odd_int(text: str) -> int:
    """argparse type: a positive odd integer (an odd k avoids tied votes)."""
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
        "--model", default="models/model", help="folder to save the model in"
    )
    train.set_defaults(func=cmd_train)

    evaluate = commands.add_parser(
        "evaluate", parents=[common], help="compare the model with baselines"
    )
    evaluate.add_argument("--model", default="models/model", help="model folder")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = commands.add_parser("predict", parents=[common], help="predict tomorrow")
    predict.add_argument("--model", default="models/model", help="model folder")
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

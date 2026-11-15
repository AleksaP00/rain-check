"""Saving and loading a trained model folder.

A model folder holds four files:
    model.pkl    the fitted scikit-learn pipeline
    config.json  city, coordinates, features, k, metrics, versions, dates
    train.csv    the training examples (date, features, label)
    test.csv     the test examples (date, features, label)
"""

import csv
import json
import logging
import os

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from raincheck.data_handler import FEATURES

logger = logging.getLogger(__name__)

MODEL_FILE = "model.pkl"
CONFIG_FILE = "config.json"


def save_split_csv(
    path: str, X: list[list[float]], y: list[int], dates: list[str]
) -> None:
    """Write examples to a CSV file: one row per day with date, features and label."""

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["date", *FEATURES, "label"])
        for date, features, label in zip(dates, X, y):
            writer.writerow([date, *features, label])


def load_split_csv(path: str) -> tuple[list[list[float]], list[int], list[str]]:
    """Read a CSV made by save_split_csv and return (X, y, dates)."""

    X, y, dates = [], [], []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            X.append([float(row[name]) for name in FEATURES])
            y.append(int(row["label"]))
            dates.append(row["date"])
    return X, y, dates


def save_artifacts(
    folder: str, pipeline: Pipeline, config: dict, splits: dict[str, tuple]
) -> None:
    """Save the pipeline, its config and the train/test examples into one folder."""

    os.makedirs(folder, exist_ok=True)

    joblib.dump(pipeline, os.path.join(folder, MODEL_FILE))

    config = {**config, "sklearn_version": sklearn.__version__}
    with open(os.path.join(folder, CONFIG_FILE), "w") as file:
        json.dump(config, file, indent=2)

    for name, (X, y, dates) in splits.items():
        save_split_csv(os.path.join(folder, f"{name}.csv"), X, y, dates)


def load_artifacts(folder: str) -> tuple[Pipeline, dict]:
    """Load (pipeline, config) from a folder made by save_artifacts.

    Only load folders you made yourself: a .pkl file can run code when it is loaded.
    """

    with open(os.path.join(folder, CONFIG_FILE)) as file:
        config = json.load(file)

    saved_version = config.get("sklearn_version")
    if saved_version != sklearn.__version__:
        logger.warning(
            "Model was saved with scikit-learn %s, but %s is installed",
            saved_version,
            sklearn.__version__,
        )

    pipeline = joblib.load(os.path.join(folder, MODEL_FILE))
    return pipeline, config

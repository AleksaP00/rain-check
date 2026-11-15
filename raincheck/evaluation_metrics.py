"""Evaluation for RainCheck: confusion counts, metrics and baselines."""

import logging

from sklearn.pipeline import Pipeline

from raincheck.data_handler import FEATURES, RAIN_THRESHOLD_MM

logger = logging.getLogger(__name__)


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
    persistence = [
        1 if row[rain_column] >= RAIN_THRESHOLD_MM else 0 for row in X_test_raw
    ]

    return {"majority": [majority] * len(X_test_raw), "persistence": persistence}


def evaluate_model(
    model: Pipeline,
    X_test_raw: list[list[float]],
    y_test: list[int],
    y_train: list[int],
) -> dict[str, dict]:
    """Return a classification report for k-NN and for both baselines.

    The model gets the raw (unscaled) features: the pipeline scales them itself.
    """

    predictions = baseline_predictions(X_test_raw, y_train)
    predictions["k-NN"] = [int(label) for label in model.predict(X_test_raw)]
    return {
        name: classification_report(y_test, y_pred)
        for name, y_pred in predictions.items()
    }

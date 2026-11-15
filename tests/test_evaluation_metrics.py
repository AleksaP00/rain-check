import pytest

from raincheck.data_handler import FEATURES
from raincheck.evaluation_metrics import (
    baseline_predictions,
    classification_report,
    confusion_counts,
    evaluate_model,
)
from raincheck.model import build_pipeline


def test_confusion_counts():
    y_true = [1, 0, 0, 1, 0, 1]
    y_pred = [1, 1, 0, 0, 0, 1]

    counts = confusion_counts(y_true, y_pred)

    assert counts == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    assert sum(counts.values()) == len(y_true)


def test_classification_report():
    report = classification_report([1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1])

    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["recall"] == pytest.approx(2 / 3)
    assert report["f1"] == pytest.approx(2 / 3)


def test_classification_report_zero_division():
    report = classification_report([0, 0, 0], [0, 0, 0])

    assert report == {"accuracy": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_baseline_predictions():
    rain_column = FEATURES.index("precipitation_sum")
    dry_day = [0.0] * len(FEATURES)
    wet_day = [0.0] * len(FEATURES)
    wet_day[rain_column] = 5.0

    result = baseline_predictions([dry_day, wet_day], [0, 0, 1])

    assert result == {"majority": [0, 0], "persistence": [0, 1]}

    result = baseline_predictions([dry_day], [1, 1, 0])

    assert result["majority"] == [1]


def test_evaluate_model():

    # Every column holds the same value, so persistence (which reads precipitation_sum) sees it too.
    X_train = [[0.0] * len(FEATURES) for _ in range(3)] + [
        [5.0] * len(FEATURES) for _ in range(2)
    ]
    y_train = [0, 0, 0, 1, 1]
    pipeline = build_pipeline(3)
    pipeline.fit(X_train, y_train)

    X_test = [[0.0] * len(FEATURES), [5.0] * len(FEATURES)]
    results = evaluate_model(pipeline, X_test, [0, 1], y_train)

    assert set(results) == {"k-NN", "persistence", "majority"}
    assert results["majority"]["recall"] == 0.0
    assert results["k-NN"]["accuracy"] == 1.0

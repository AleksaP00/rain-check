import pytest
import requests


from model import KNNClassifier, StandardScaler
from project import fetch_today, IncompleteDataError, save_model, load_model, validate_date, slugify, geocode, LocationNotFoundError, fetch_history, FEATURES, save_csv, load_csv, clean_rows, make_examples, time_split, confusion_counts, classification_report, baseline_predictions


def test_validate_date():

    assert validate_date("2026-10-18") == "2026-10-18"

    with pytest.raises(ValueError):
        validate_date("2026-02-30")

    with pytest.raises(ValueError):
        validate_date("17.10.2026")

    with pytest.raises(ValueError):
        validate_date("2026-13-01")


def test_slugify():

    assert slugify("Novi Sad") == "novi_sad"

    assert slugify(" Novi Sad!! ") == "novi_sad"


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def test_geocode(monkeypatch):

    def fake_get(*args, **kwargs):
        return FakeResponse({"results": [{"latitude": 1.5, "longitude": 2.5, "timezone": "Test/Zone"}]})

    monkeypatch.setattr(requests, "get", fake_get)

    assert geocode("Novi Sad") == (1.5, 2.5, "Test/Zone")


def test_geocode_not_found(monkeypatch):

    def fake_get(*args, **kwargs):
        return FakeResponse({})

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(LocationNotFoundError):
        geocode("kewrq")


def test_fetch_history(monkeypatch):

    def fake_get(*args, **kwargs):
        return FakeResponse({"daily": {"time": ["2026-10-01", "2026-10-02"], **{name: [1.0, 2.0] for name in FEATURES}}})

    monkeypatch.setattr(requests, "get", fake_get)

    rows = fetch_history(45.0, 19.0, "2026-10-01", "2026-10-02")

    assert len(rows) == 2
    assert rows[0]["date"] == "2026-10-01"
    assert rows[1]["precipitation_sum"] == 2.0


def test_csv_roundtrip(tmp_path):

    rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    save_csv(rows, str(tmp_path / "test.csv"))
    loaded = load_csv(str(tmp_path / "test.csv"))
    assert loaded == rows


def test_clean_rows():

    dictionary = {f: "1" for f in FEATURES}
    dictionary_true = {f: 1 for f in FEATURES}
    dictionary_1 = [{"date": "2020-10-04", **dictionary}]
    dictionary_2 = [{"date": "2020-10-04", **dictionary, "cloud_cover_mean": ""}]
    dictionary_3 = [{"date": "2020-10-04", **dictionary, "precipitation_sum": "-1"}]

    assert clean_rows(dictionary_1) == [{"date": "2020-10-04", **dictionary_true}]
    assert clean_rows(dictionary_2) == []
    assert clean_rows(dictionary_3) == []


def test_make_examples():

    dictionary = {f: 1 for f in FEATURES}
    rainy_tomorrow = [{"date" : "2020-10-04", **dictionary, "precipitation_sum": 0.0}, {"date" : "2020-10-05", **dictionary, "precipitation_sum": 5.0}]
    with_gap = [{"date" : "2020-10-04", **dictionary}, {"date" : "2020-10-06", **dictionary}, {"date" : "2020-10-07", **dictionary}]

    X, y, dates = make_examples(rainy_tomorrow)
    assert len(X) == 1 and y == [1]
    assert dates == ["2020-10-04"]

    X, y, dates = make_examples(with_gap)
    assert len(X) == 1
    assert dates == ["2020-10-06"]


def test_time_split():

    X = list(range(10))
    y = list(range(10))
    X_train, X_test, y_train, y_test = time_split(X, y)

    assert len(X_train) == 8 and len(X_test) == 2 and len(y_train) == 8 and len(y_test) == 2
    assert X_test == [8, 9] and X_train == [0, 1, 2, 3, 4, 5, 6, 7]
    assert y_test == [8, 9] and y_train == [0, 1, 2, 3, 4, 5, 6, 7]
    assert X_train + X_test == X and y_train + y_test == y

    X_train, X_test, y_train, y_test = time_split(X, y, 0.3)

    assert len(X_train) == 7 and len(X_test) == 3
    assert X_train + X_test == X and y_train + y_test == y


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


def test_save_load_model(tmp_path):

    X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [10.0, 100.0], [11.0, 110.0]]
    y = [0, 0, 0, 1, 1]
    scaler = StandardScaler()
    scaler.fit(X)
    model = KNNClassifier(3)
    model.fit(scaler.transform(X), y)

    path = str(tmp_path / "models" / "model.json")
    save_model(path, scaler, model, {"city": "Test"})
    loaded_scaler, loaded_model, metadata = load_model(path)

    assert metadata == {"city": "Test"}
    assert loaded_model.predict(loaded_scaler.transform(X)) == model.predict(scaler.transform(X))


def test_fetch_today(monkeypatch):

    def fake_get(*args, **kwargs):
        return FakeResponse({"daily": {"time": ["2026-10-28"], **{name: [3.0] for name in FEATURES}}})

    monkeypatch.setattr(requests, "get", fake_get)

    today = fetch_today(45.0, 19.0)

    assert today["date"] == "2026-10-28"
    assert today["cloud_cover_mean"] == 3.0


def test_fetch_today_missing_values(monkeypatch):

    def fake_get(*args, **kwargs):
        return FakeResponse({"daily": {"time": ["2026-10-28"], **{name: [None] for name in FEATURES}}})

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(IncompleteDataError):
        fetch_today(45.0, 19.0)

import pytest
import requests


from project import validate_date, slugify, geocode, LocationNotFoundError, fetch_history, FEATURES, save_csv, load_csv, clean_rows, make_examples


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

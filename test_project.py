import pytest
import requests


from project import validate_date, slugify, geocode, LocationNotFoundError, fetch_history, FEATURES



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
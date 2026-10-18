import pytest

from project import validate_date


def test_validate_date():
    
    assert validate_date("2026-10-18") == "2026-10-18"

    with pytest.raises(ValueError):
        validate_date("2026-02-30")

    with pytest.raises(ValueError):
        validate_date("17.10.2026")

    with pytest.raises(ValueError):
        validate_date("2026-13-01")


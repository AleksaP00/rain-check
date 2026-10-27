import pytest

from model import StandardScaler


def test_scaler_mean_and_stdev():
    X = [[170, 65], [180, 80], [160, 50]]
    scaler = StandardScaler()
    scaler.fit(X)
    scaled = scaler.transform(X)

    for column in zip(*scaled):
        assert sum(column) / len(column) == pytest.approx(0)

    row = scaler.transform([[180, 80]])[0]
    assert row == pytest.approx([1.2247, 1.2247], abs=1e-3)


def test_scaler_constant_column():
    scaler = StandardScaler()
    scaler.fit([[1, 5], [2, 5], [3, 5]])

    assert scaler.transform([[2, 5]]) == [[0.0, 0.0]]

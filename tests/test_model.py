import json

import pytest

from raincheck.model import KNNClassifier, StandardScaler


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


def test_scaler_dict_roundtrip():
    X = [[170, 65], [180, 80], [160, 50]]
    scaler = StandardScaler()
    scaler.fit(X)

    restored = StandardScaler.from_dict(scaler.to_dict())

    assert restored.means == scaler.means
    assert restored.stdevs == scaler.stdevs
    assert restored.transform(X) == scaler.transform(X)


def test_knn_rejects_bad_k():
    for bad_k in (4, -3, 0, 7.0):
        with pytest.raises(ValueError):
            KNNClassifier(bad_k)

    assert KNNClassifier(7).k == 7


def test_knn_predict_proba_one():
    model = KNNClassifier(3)
    model.fit([[0, 0], [1, 0], [0, 2], [5, 5]], [0, 1, 1, 0])

    assert model.predict_proba_one([0, 1]) == pytest.approx(2 / 3)

    model.k = 1
    assert model.predict_proba_one([5, 4]) == 0.0


def test_knn_predict():
    model = KNNClassifier(3)
    model.fit([[0, 0], [1, 0], [0, 2], [5, 5]], [0, 1, 1, 0])

    assert model.predict([[0, 1]]) == [1]
    assert model.predict([[0, 1]], threshold=0.8) == [0]


def test_knn_dict_roundtrip():
    X = [[0, 0], [1, 0], [0, 2], [5, 5]]
    y = [0, 1, 1, 0]
    model = KNNClassifier(3)
    model.fit(X, y)

    saved = json.loads(json.dumps(model.to_dict()))
    restored = KNNClassifier.from_dict(saved)

    assert restored.k == 3
    assert restored.predict_proba_one([0, 1]) == model.predict_proba_one([0, 1])
    assert restored.predict(X) == model.predict(X)

from raincheck.model import build_pipeline


def test_build_pipeline():
    pipeline = build_pipeline(7)

    assert list(pipeline.named_steps) == ["scaler", "knn"]
    assert pipeline.named_steps["knn"].n_neighbors == 7

    X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [10.0, 100.0], [11.0, 110.0]] * 2
    y = [0, 0, 0, 1, 1] * 2
    pipeline.fit(X, y)
    assert list(pipeline.predict([[2.5, 25.0], [10.5, 105.0]])) == [0, 1]

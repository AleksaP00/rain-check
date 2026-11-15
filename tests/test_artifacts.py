import json

from raincheck.artifacts import load_artifacts, load_split_csv, save_artifacts
from raincheck.data_handler import FEATURES
from raincheck.model import build_pipeline


def make_splits():
    dry = [0.0] * len(FEATURES)
    wet = [5.0] * len(FEATURES)
    train_dates = [f"2026-01-0{i}" for i in range(1, 6)]
    train = ([dry, dry, dry, wet, wet], [0, 0, 0, 1, 1], train_dates)
    test = ([dry, wet], [0, 1], ["2026-01-06", "2026-01-07"])
    return {"train": train, "test": test}


def test_save_load_artifacts(tmp_path):

    splits = make_splits()
    X_train, y_train, _ = splits["train"]
    X_test, _, _ = splits["test"]
    pipeline = build_pipeline(3)
    pipeline.fit(X_train, y_train)

    folder = str(tmp_path / "models" / "test_model")
    save_artifacts(folder, pipeline, {"city": "Test"}, splits)
    loaded, config = load_artifacts(folder)

    assert list(loaded.predict(X_test)) == list(pipeline.predict(X_test))
    assert config["city"] == "Test"
    assert "sklearn_version" in config

    with open(tmp_path / "models" / "test_model" / "config.json") as file:
        assert json.load(file) == config


def test_split_csv_roundtrip(tmp_path):

    splits = make_splits()
    X_train, y_train, _ = splits["train"]
    pipeline = build_pipeline(3).fit(X_train, y_train)
    save_artifacts(str(tmp_path), pipeline, {}, splits)

    for name in ["train", "test"]:
        assert load_split_csv(str(tmp_path / f"{name}.csv")) == splits[name]

import argparse
import sys

import pytest

from raincheck.cli import build_parser, format_table, main, odd_int


def test_format_table():

    report = {"accuracy": 0.75, "precision": 0.5, "recall": 0.25, "f1": 1 / 3}
    table = format_table({"k-NN": report, "persistence": report, "majority": report})
    lines = table.splitlines()

    assert len(lines) == 4
    assert lines[1].startswith("k-NN")
    assert "0.333" in lines[1]


def test_odd_int():

    assert odd_int("7") == 7

    for bad in ["4", "0", "-3", "seven"]:
        with pytest.raises(argparse.ArgumentTypeError):
            odd_int(bad)


def test_build_parser():

    args = build_parser().parse_args(["train", "--data", "x.csv", "--k", "7"])
    assert args.k == 7
    assert args.command == "train"
    assert args.verbose is False

    assert build_parser().parse_args(["predict", "-v"]).verbose is True

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--data", "x.csv", "--k", "4"])

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["fetch", "--city", "X", "--start", "2026-02-30", "--end", "2026-03-01"]
        )


def test_main_missing_file_exits_with_1(monkeypatch):

    monkeypatch.setattr(
        sys,
        "argv",
        ["raincheck", "evaluate", "--model", "no_such_folder"],
    )

    with pytest.raises(SystemExit) as info:
        main()

    assert info.value.code == 1

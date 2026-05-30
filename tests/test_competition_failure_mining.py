from __future__ import annotations

from dataclasses import replace
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint import (  # noqa: E402
    CompetitionFailureMiningError,
    mine_competition_failures,
    run_competition_train_eval,
)


ROMAN_PROMPT = """In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:
11 -> XI
15 -> XV
Now, write the number 38 in the Wonderland numeral system."""


def write_train(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "answer"])
        writer.writeheader()
        writer.writerow({"id": "ok", "prompt": ROMAN_PROMPT, "answer": "XXXVIII"})
        writer.writerow({"id": "wrong", "prompt": ROMAN_PROMPT, "answer": "I"})
        writer.writerow({"id": "bad", "prompt": "no examples", "answer": "1"})


def test_mines_family_accuracy_and_abstentions(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    out = tmp_path / "out"
    write_train(train)
    run_competition_train_eval(train, out)

    report = mine_competition_failures(out)

    assert report.prediction_count == 2
    assert report.abstention_count == 0
    assert report.accuracy == 0.5
    assert report.accuracy_by_family["roman_numeral"] == 0.5
    assert report.wrong_predictions_by_family["roman_numeral"] == 1
    assert dict(report.top_error_reasons)


def test_forged_hash_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    out = tmp_path / "out"
    write_train(train)
    run_competition_train_eval(train, out)
    report = mine_competition_failures(out)

    with pytest.raises(CompetitionFailureMiningError):
        replace(report, report_hash="forged")


def test_missing_artifact_rejected(tmp_path: Path) -> None:
    with pytest.raises(CompetitionFailureMiningError):
        mine_competition_failures(tmp_path)

from __future__ import annotations

from dataclasses import replace
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.adapter_training_data.solver_distillation import (  # noqa: E402
    SolverDistillationError,
    build_solver_distilled_examples,
    distill_solver_predictions,
)
from nemotron_engine.competition_sprint import parse_competition_prompt  # noqa: E402


ROMAN_PROMPT = """In Alice's Wonderland, numbers are secretly converted into a different numeral system.
11 -> XI
15 -> XV
Now, write the number 38 in the Wonderland numeral system."""


def test_distills_only_solver_correct_examples(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "answer"])
        writer.writeheader()
        writer.writerow({"id": "ok", "prompt": ROMAN_PROMPT, "answer": "XXXVIII"})
        writer.writerow({"id": "wrong", "prompt": ROMAN_PROMPT, "answer": "I"})
    problems = (
        parse_competition_prompt("ok", ROMAN_PROMPT, "XXXVIII"),
        parse_competition_prompt("wrong", ROMAN_PROMPT, "I"),
    )

    records = distill_solver_predictions(problems, corpus_source_path=str(csv_path))
    examples = build_solver_distilled_examples(problems, records)

    assert [record.correct for record in records] == [True, False]
    assert [example.problem_id for example in examples] == ["ok"]
    assert examples[0].source == "solver_distilled"


def test_distillation_requires_labels() -> None:
    problem = parse_competition_prompt("unlabeled", ROMAN_PROMPT, None)

    with pytest.raises(SolverDistillationError):
        distill_solver_predictions((problem,))


def test_distillation_record_hash_forgery_rejected(tmp_path: Path) -> None:
    problem = parse_competition_prompt("ok", ROMAN_PROMPT, "XXXVIII")
    record = distill_solver_predictions((problem,))[0]

    with pytest.raises(SolverDistillationError):
        replace(record, record_hash="forged")

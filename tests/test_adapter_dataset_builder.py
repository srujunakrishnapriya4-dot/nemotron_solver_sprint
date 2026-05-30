from __future__ import annotations

from dataclasses import replace
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.adapter_training_data.dataset_builder import AdapterTrainingDataError, build_adapter_training_datasets  # noqa: E402


ROMAN_PROMPT = """In Alice's Wonderland, numbers are secretly converted into a different numeral system.
11 -> XI
15 -> XV
Now, write the number 38 in the Wonderland numeral system."""

BIT_PROMPT = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
00000000 -> 11111111
11111111 -> 00000000
Now, determine the output for: 00110100"""

UNIT_PROMPT = """In Alice's Wonderland, a secret unit conversion is applied to measurements.
1.00 m becomes 3.00
2.00 m becomes 5.00
3.00 m becomes 7.00
Now, convert the following measurement: 4.00 m"""


def write_train(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "answer"])
        writer.writeheader()
        writer.writerow({"id": "roman1", "prompt": ROMAN_PROMPT, "answer": "XXXVIII"})
        writer.writerow({"id": "roman2", "prompt": ROMAN_PROMPT, "answer": "XXXVIII"})
        writer.writerow({"id": "bit1", "prompt": BIT_PROMPT, "answer": "11001011"})
        writer.writerow({"id": "bit2", "prompt": BIT_PROMPT, "answer": "11001011"})
        writer.writerow({"id": "unit1", "prompt": UNIT_PROMPT, "answer": "9.00"})
        writer.writerow({"id": "unit2", "prompt": UNIT_PROMPT, "answer": "9.00"})
        writer.writerow({"id": "bad", "prompt": "no examples", "answer": "1"})


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_builds_all_adapter_training_artifacts(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    out = tmp_path / "artifacts"
    write_train(train)

    report = build_adapter_training_datasets(
        train,
        out,
        seed=11,
        validation_per_family=1,
        hard_family_caps={"bit_manipulation": 2, "unit_conversion": 2},
    )

    expected = {
        "train_direct.jsonl",
        "train_family_tagged.jsonl",
        "train_solver_distilled.jsonl",
        "train_hard_oversampled.jsonl",
        "validation_family_balanced.jsonl",
        "dataset_manifest.json",
    }
    assert expected <= {path.name for path in out.iterdir()}
    assert report.parsed_count == 6
    assert report.parse_error_count == 1
    assert report.validation_count == 3
    assert report.file_counts["train_direct.jsonl"] == 3
    assert all("Family:" in row["messages"][0]["content"] for row in read_jsonl(out / "train_family_tagged.jsonl"))
    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_test_labels_used"] is True
    assert "train_direct.jsonl" in manifest["artifact_hashes"]


def test_dataset_report_hash_forgery_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    out = tmp_path / "artifacts"
    write_train(train)
    report = build_adapter_training_datasets(train, out, validation_per_family=1)

    with pytest.raises(AdapterTrainingDataError):
        replace(report, report_hash="forged")

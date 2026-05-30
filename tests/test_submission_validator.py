from __future__ import annotations

import json
from io import BytesIO
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.submission.submission_validator import (  # noqa: E402
    SubmissionRow,
    SubmissionValidationError,
    render_submission_csv,
    validate_adapter_dir,
    validate_submission_csv,
    validate_submission_rows,
    validate_submission_zip,
)


def _mock_adapter_config(monkeypatch: pytest.MonkeyPatch, text: str | None) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: text is not None)
    monkeypatch.setattr(Path, "is_file", lambda self: text is not None)
    monkeypatch.setattr(Path, "read_text", lambda self, encoding="utf-8": text or "")


def _zip_bytes(entries: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_adapter_dir_missing_config_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_adapter_config(monkeypatch, None)

    report = validate_adapter_dir("adapter")

    assert report.valid is False
    assert "Missing adapter_config.json" in report.errors[0]


def test_adapter_dir_invalid_json_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_adapter_config(monkeypatch, "{not-json")

    report = validate_adapter_dir("adapter")

    assert report.valid is False
    assert "Invalid JSON" in report.errors[0]


@pytest.mark.parametrize("key", ["r", "rank", "lora_rank"])
def test_top_level_rank_keys_over_32_are_rejected(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    _mock_adapter_config(monkeypatch, json.dumps({key: 33}))

    report = validate_adapter_dir("adapter")

    assert report.valid is False
    assert f"{key}=33 exceeds" in report.errors[0]


def test_nested_peft_config_rank_over_32_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_adapter_config(monkeypatch, json.dumps({"peft_config": {"r": 64}}))

    report = validate_adapter_dir("adapter")

    assert report.valid is False
    assert "peft_config.r=64 exceeds" in report.errors[0]


def test_zip_without_adapter_config_rejected() -> None:
    report = validate_submission_zip(_zip_bytes({"weights.bin": "placeholder"}))

    assert report.valid is False
    assert "Missing adapter_config.json" in report.errors[0]


def test_zip_with_valid_adapter_config_accepted() -> None:
    report = validate_submission_zip(_zip_bytes({"adapter/adapter_config.json": json.dumps({"r": 32})}))

    assert report.valid is True
    assert report.errors == ()
    assert report.adapter_config_path == "adapter/adapter_config.json"


def test_validate_submission_rows_and_csv_compatibility() -> None:
    report = validate_submission_rows(
        [{"id": "10", "answer": "210"}, {"id": "2", "answer": 17}],
        expected_problem_ids=["2", "10"],
    )

    assert report.valid is True
    assert report.rows == (SubmissionRow(id="2", answer=17), SubmissionRow(id="10", answer=210))
    assert render_submission_csv(report.rows) == "id,answer\n2,17\n10,210\n"

    bad_header = validate_submission_csv("problem_id,answer\n2,17\n")
    assert bad_header.valid is False
    assert "header" in bad_header.errors[0]

    with pytest.raises(SubmissionValidationError):
        validate_submission_rows([{"id": "2", "answer": True}], fail_fast=True)

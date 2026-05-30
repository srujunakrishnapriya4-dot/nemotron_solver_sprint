from __future__ import annotations

import json
import sys
from pathlib import Path
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaggle_sprint4.kaggle_behavioral_eval import (  # noqa: E402
    BehavioralEvalError,
    compare_parent_child,
    extract_final_answer,
    validate_behavioral_report,
)
from kaggle_sprint4.kaggle_package_adapter import AdapterPackagingError, create_submission_zip  # noqa: E402
from kaggle_sprint4.kaggle_validate_adapter import validate_adapter_and_zip  # noqa: E402


def validation_rows() -> list[dict]:
    return [
        {"problem_id": "a", "family": "bit", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\\boxed{1010}"}]},
        {"problem_id": "b", "family": "roman", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "XII"}]},
    ]


def write_adapter(root: Path) -> None:
    root.mkdir()
    (root / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(b"x" * 2048)


def write_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("adapter_config.json", "{}")
        archive.writestr("adapter_model.safetensors", b"x" * 2048)


def test_extract_final_answer_supports_boxed_and_final_line() -> None:
    assert extract_final_answer("work\n\\boxed{XXXVIII}") == "XXXVIII"
    assert extract_final_answer("reason\nfinal answer") == "final answer"


def test_behavioral_report_parser_rejects_child_worse_than_parent() -> None:
    report = compare_parent_child(validation_rows(), {"a": "1010", "b": "XII"}, {"a": "0000", "b": "XII"})

    with pytest.raises(BehavioralEvalError):
        validate_behavioral_report(report)


def test_behavioral_report_allows_non_regressing_child() -> None:
    report = compare_parent_child(validation_rows(), {"a": "1010", "b": "XII"}, {"a": "\\boxed{1010}", "b": "XII"})

    assert validate_behavioral_report(report) is True


def test_packaging_rejects_missing_behavioral_eval_by_default(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    write_adapter(adapter)

    with pytest.raises(AdapterPackagingError):
        create_submission_zip(adapter, tmp_path / "submission.zip")


def test_packaging_allows_baseline_adapter_only(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    write_adapter(adapter)

    output = create_submission_zip(adapter, tmp_path / "submission.zip", baseline_adapter_only=True)

    assert output.is_file()


def test_validation_does_not_print_ready_when_behavior_unknown(tmp_path: Path, capsys) -> None:
    adapter = tmp_path / "adapter"
    write_adapter(adapter)
    zip_path = tmp_path / "submission.zip"
    write_zip(zip_path)

    assert validate_adapter_and_zip(adapter, zip_path) is False
    output = capsys.readouterr().out
    assert "STRUCTURE_READY" in output
    assert "BEHAVIOR_UNKNOWN" in output
    assert "READY_TO_SUBMIT" not in output


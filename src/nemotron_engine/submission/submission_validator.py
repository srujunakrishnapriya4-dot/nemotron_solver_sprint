"""Deterministic Pass 1 submission and adapter validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import io
import json
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate


MAX_LORA_RANK = 32
RANK_KEYS = ("r", "rank", "lora_rank")


class SubmissionValidationError(ValueError):
    """Raised when fail-fast validation is requested."""


@dataclass(frozen=True)
class SubmissionRow:
    id: str
    answer: int


@dataclass(frozen=True)
class SubmissionValidationResult:
    valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    rows: tuple[SubmissionRow, ...] = field(default_factory=tuple)
    adapter_config: Mapping[str, Any] | None = None
    adapter_config_path: str | None = None

    def raise_for_errors(self) -> None:
        if self.errors:
            raise SubmissionValidationError("; ".join(self.errors))


SubmissionValidationReport = SubmissionValidationResult


def validate_adapter_dir(path: str | Path) -> SubmissionValidationResult:
    """Validate a LoRA adapter directory without attempting model reload."""

    root = Path(path)
    config_path = root / "adapter_config.json"
    if not config_path.exists():
        return SubmissionValidationResult(
            valid=False,
            errors=(f"Missing adapter_config.json in adapter directory: {root}",),
            adapter_config_path=str(config_path),
        )
    if not config_path.is_file():
        return SubmissionValidationResult(
            valid=False,
            errors=(f"adapter_config.json is not a file: {config_path}",),
            adapter_config_path=str(config_path),
        )

    try:
        config = _load_json_text(config_path.read_text(encoding="utf-8"), source=str(config_path))
    except SubmissionValidationError as exc:
        return SubmissionValidationResult(
            valid=False,
            errors=(str(exc),),
            adapter_config_path=str(config_path),
        )
    return _validate_adapter_config(config, adapter_config_path=str(config_path))


def validate_submission_zip(path: str | Path | bytes | BinaryIO) -> SubmissionValidationResult:
    """Validate an adapter zip and locate exactly one adapter_config.json inside it."""

    source_name = "<in-memory zip>" if isinstance(path, bytes) or hasattr(path, "read") else str(path)
    zip_source: str | Path | io.BytesIO | BinaryIO
    if isinstance(path, bytes):
        zip_source = io.BytesIO(path)
    elif hasattr(path, "read"):
        zip_source = path
    else:
        zip_path = Path(path)
        if not zip_path.exists():
            return SubmissionValidationResult(valid=False, errors=(f"Submission zip does not exist: {zip_path}",))
        zip_source = zip_path
    try:
        with ZipFile(zip_source) as archive:
            names = sorted(info.filename for info in archive.infolist() if not info.is_dir())
            unsafe = [name for name in names if _is_unsafe_zip_name(name)]
            if unsafe:
                return SubmissionValidationResult(
                    valid=False,
                    errors=(f"Unsafe zip member path: {unsafe[0]}",),
                )
            config_names = [name for name in names if Path(name).name == "adapter_config.json"]
            if not config_names:
                return SubmissionValidationResult(
                    valid=False,
                    errors=("Missing adapter_config.json in submission zip.",),
                )
            if len(config_names) > 1:
                return SubmissionValidationResult(
                    valid=False,
                    errors=(f"Multiple adapter_config.json files found: {config_names}",),
                )
            config_name = config_names[0]
            raw = archive.read(config_name).decode("utf-8")
    except BadZipFile as exc:
        return SubmissionValidationResult(valid=False, errors=(f"Invalid zip file: {source_name}",))
    except UnicodeDecodeError as exc:
        return SubmissionValidationResult(valid=False, errors=(f"adapter_config.json is not valid UTF-8: {exc}",))

    try:
        config = _load_json_text(raw, source=config_name)
    except SubmissionValidationError as exc:
        return SubmissionValidationResult(valid=False, errors=(str(exc),), adapter_config_path=config_name)
    return _validate_adapter_config(config, adapter_config_path=config_name)


def validate_submission_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_problem_ids: Sequence[str] | None = None,
    fail_fast: bool = False,
) -> SubmissionValidationResult:
    rows: list[SubmissionRow] = []
    errors: list[str] = []
    seen: set[str] = set()

    for index, record in enumerate(records, start=1):
        if "id" not in record or "answer" not in record:
            errors.append(f"Row {index} must contain 'id' and 'answer'.")
            if fail_fast:
                return _finish_rows(rows, errors, fail_fast)
            continue

        problem_id = str(record["id"]).strip()
        if not problem_id:
            errors.append(f"Row {index} has an empty id.")
            if fail_fast:
                return _finish_rows(rows, errors, fail_fast)
            continue
        if "," in problem_id or "\n" in problem_id or "\r" in problem_id:
            errors.append(f"Row {index} has an unsafe id: {problem_id!r}.")
            if fail_fast:
                return _finish_rows(rows, errors, fail_fast)
            continue
        if problem_id in seen:
            errors.append(f"Duplicate problem id: {problem_id}.")
            if fail_fast:
                return _finish_rows(rows, errors, fail_fast)
            continue

        try:
            answer = normalize_answer_candidate(record["answer"])
        except ValueError as exc:
            errors.append(f"Row {index} has invalid answer: {exc}")
            if fail_fast:
                return _finish_rows(rows, errors, fail_fast)
            continue

        seen.add(problem_id)
        rows.append(SubmissionRow(id=problem_id, answer=answer))

    if expected_problem_ids is not None:
        expected = {str(item).strip() for item in expected_problem_ids}
        actual = {row.id for row in rows}
        missing = sorted(expected - actual, key=_problem_id_sort_key)
        extra = sorted(actual - expected, key=_problem_id_sort_key)
        if missing:
            errors.append(f"Missing expected problem ids: {missing}.")
        if extra:
            errors.append(f"Unexpected problem ids: {extra}.")

    return _finish_rows(rows, errors, fail_fast)


def validate_submission_csv(
    csv_text: str,
    *,
    expected_problem_ids: Sequence[str] | None = None,
    fail_fast: bool = False,
) -> SubmissionValidationResult:
    reader = csv.DictReader(io.StringIO(csv_text or ""))
    if reader.fieldnames != ["id", "answer"]:
        report = SubmissionValidationResult(
            valid=False,
            errors=(f"Submission CSV header must be exactly ['id', 'answer']; got {reader.fieldnames}.",),
        )
        if fail_fast:
            report.raise_for_errors()
        return report
    return validate_submission_rows(
        reader,
        expected_problem_ids=expected_problem_ids,
        fail_fast=fail_fast,
    )


def render_submission_csv(rows: Sequence[SubmissionRow]) -> str:
    ordered = sorted(rows, key=lambda row: _problem_id_sort_key(row.id))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["id", "answer"], lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow({"id": row.id, "answer": row.answer})
    return buffer.getvalue()


def _load_json_text(text: str, *, source: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SubmissionValidationError(f"Invalid JSON in {source}: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise SubmissionValidationError(f"adapter_config.json must contain a JSON object: {source}")
    return payload


def _validate_adapter_config(
    config: Mapping[str, Any],
    *,
    adapter_config_path: str,
) -> SubmissionValidationResult:
    errors = _rank_errors(config)
    return SubmissionValidationResult(
        valid=not errors,
        errors=tuple(errors),
        adapter_config=dict(config),
        adapter_config_path=adapter_config_path,
    )


def _rank_errors(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in RANK_KEYS:
        if key in config:
            _append_rank_error(errors, key, config[key])
    peft_config = config.get("peft_config")
    if isinstance(peft_config, Mapping) and "r" in peft_config:
        _append_rank_error(errors, "peft_config.r", peft_config["r"])
    return sorted(errors)


def _append_rank_error(errors: list[str], key: str, value: Any) -> None:
    if isinstance(value, bool):
        errors.append(f"{key} must be an integer rank, got boolean.")
        return
    try:
        rank = int(value)
    except (TypeError, ValueError):
        errors.append(f"{key} must be an integer rank, got {value!r}.")
        return
    if rank > MAX_LORA_RANK:
        errors.append(f"{key}={rank} exceeds maximum allowed LoRA rank {MAX_LORA_RANK}.")


def _is_unsafe_zip_name(name: str) -> bool:
    path = Path(name)
    return path.is_absolute() or ".." in path.parts


def _finish_rows(
    rows: list[SubmissionRow],
    errors: list[str],
    fail_fast: bool,
) -> SubmissionValidationResult:
    report = SubmissionValidationResult(
        valid=not errors,
        rows=tuple(sorted(rows, key=lambda row: _problem_id_sort_key(row.id))),
        errors=tuple(errors),
    )
    if fail_fast:
        report.raise_for_errors()
    return report


def _problem_id_sort_key(problem_id: str) -> tuple[int, int | str, str]:
    value = problem_id.strip()
    if value.isdigit():
        return (0, int(value), value)
    return (1, value, value)


__all__ = [
    "MAX_LORA_RANK",
    "SubmissionRow",
    "SubmissionValidationError",
    "SubmissionValidationReport",
    "SubmissionValidationResult",
    "render_submission_csv",
    "validate_adapter_dir",
    "validate_submission_csv",
    "validate_submission_rows",
    "validate_submission_zip",
]

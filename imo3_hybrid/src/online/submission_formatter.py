"""
Submission formatting for the online Kaggle runtime path.

This module is intentionally strict:
- It emits typed submission rows only.
- It does not silently coerce unsafe answer types.
- It preserves deterministic ordering and CSV formatting.
- It keeps rich debug/evidence sidecars separate from competition rows.

The goal is to make the final online boundary competition-safe and schema-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import re
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from src.aggregation.final_selector import FinalSelectionResult
from src.common.constants import ANSWER_MAX, ANSWER_MIN
from src.common.schemas import FinalPrediction


_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class SubmissionFormattingError(ValueError):
    """Raised when a final prediction cannot be safely converted to a submission row."""


class SubmissionRow(BaseModel):
    id: str
    answer: int = Field(ge=ANSWER_MIN, le=ANSWER_MAX)


class SubmissionDebugRow(BaseModel):
    problem_id: str
    submission_answer: int = Field(ge=ANSWER_MIN, le=ANSWER_MAX)
    confidence: float = Field(ge=0.0, le=1.0)
    method_used: str = ""
    winning_answer_raw: str = ""
    winning_answer_canonical: str = ""
    num_branches_generated: int = Field(default=0, ge=0)
    num_branches_survived: int = Field(default=0, ge=0)
    solve_time_sec: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    rationale_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubmissionArtifactBundle(BaseModel):
    rows: list[SubmissionRow] = Field(default_factory=list)
    debug_rows: list[SubmissionDebugRow] = Field(default_factory=list)

    def submission_records(self) -> list[dict[str, Any]]:
        return [row.model_dump() for row in self.rows]

    def debug_records(self) -> list[dict[str, Any]]:
        return [row.model_dump() for row in self.debug_rows]


@dataclass(frozen=True)
class SubmissionFormatterConfig:
    enforce_answer_range: bool = True
    include_debug_sidecar: bool = False
    sort_rows: bool = True
    require_unique_problem_ids: bool = True
    strict_problem_id: bool = True


def format_submission_row(
    selection: FinalSelectionResult | FinalPrediction,
    *,
    config: SubmissionFormatterConfig | None = None,
) -> SubmissionRow:
    """
    Convert a final selected answer into a strict Kaggle submission row.

    Accepted inputs:
    - FinalSelectionResult from aggregation.final_selector
    - FinalPrediction from common.schemas
    """
    resolved = config or SubmissionFormatterConfig()
    prediction = _extract_prediction(selection)
    problem_id = _validate_problem_id(prediction.problem_id, strict=resolved.strict_problem_id)
    answer = _extract_final_answer(selection)
    answer_int = _coerce_answer_to_submission_int(
        answer,
        enforce_answer_range=resolved.enforce_answer_range,
    )
    return SubmissionRow(id=problem_id, answer=answer_int)


def format_debug_row(
    selection: FinalSelectionResult | FinalPrediction,
    *,
    config: SubmissionFormatterConfig | None = None,
) -> SubmissionDebugRow:
    """
    Build a rich sidecar row for logging or offline debugging.

    This never changes the competition submission row and remains separate from it.
    """
    resolved = config or SubmissionFormatterConfig()
    prediction = _extract_prediction(selection)
    problem_id = _validate_problem_id(prediction.problem_id, strict=resolved.strict_problem_id)
    answer = _extract_final_answer(selection)
    answer_int = _coerce_answer_to_submission_int(
        answer,
        enforce_answer_range=resolved.enforce_answer_range,
    )

    winning_answer_raw = ""
    winning_answer_canonical = ""
    warnings: list[str] = []
    rationale_summary = ""
    metadata: dict[str, Any] = {}

    if isinstance(selection, FinalSelectionResult):
        winning_answer_raw = selection.prediction.winning_cluster.answer
        winning_answer_canonical = selection.prediction.winning_cluster.answer_canonical
        warnings = [str(w.value if hasattr(w, "value") else w) for w in selection.warnings]
        rationale_summary = selection.rationale.summary
        metadata = dict(selection.metadata)

    return SubmissionDebugRow(
        problem_id=problem_id,
        submission_answer=answer_int,
        confidence=_clamp01(float(prediction.confidence)),
        method_used=str(prediction.method_used),
        winning_answer_raw=str(winning_answer_raw),
        winning_answer_canonical=str(winning_answer_canonical),
        num_branches_generated=max(0, int(prediction.num_branches_generated)),
        num_branches_survived=max(0, int(prediction.num_branches_survived)),
        solve_time_sec=max(0.0, float(prediction.solve_time_sec)),
        warnings=warnings,
        rationale_summary=rationale_summary,
        metadata=metadata,
    )


def format_submission_batch(
    selections: Sequence[FinalSelectionResult | FinalPrediction],
    *,
    expected_problem_ids: Sequence[str] | None = None,
    config: SubmissionFormatterConfig | None = None,
) -> SubmissionArtifactBundle:
    """
    Format a batch of final results into submission rows and an optional debug sidecar.

    This function enforces:
    - validated problem_id / answer pairing
    - deterministic ordering
    - duplicate-id detection
    """
    resolved = config or SubmissionFormatterConfig()

    rows = [format_submission_row(item, config=resolved) for item in selections]
    if resolved.require_unique_problem_ids:
        _ensure_unique_problem_ids(rows)

    if expected_problem_ids is not None:
        _validate_expected_problem_ids(rows, expected_problem_ids)

    if resolved.sort_rows:
        rows = sort_submission_rows(rows)

    debug_rows: list[SubmissionDebugRow] = []
    if resolved.include_debug_sidecar:
        debug_rows = [format_debug_row(item, config=resolved) for item in selections]
        if resolved.sort_rows:
            debug_rows = sort_debug_rows(debug_rows)

    return SubmissionArtifactBundle(rows=rows, debug_rows=debug_rows)


def sort_submission_rows(rows: Sequence[SubmissionRow]) -> list[SubmissionRow]:
    """Deterministically sort submission rows by integer-aware problem-id order."""
    return sorted(rows, key=lambda row: _problem_id_sort_key(row.id))


def sort_debug_rows(rows: Sequence[SubmissionDebugRow]) -> list[SubmissionDebugRow]:
    """Deterministically sort debug rows by integer-aware problem-id order."""
    return sorted(rows, key=lambda row: _problem_id_sort_key(row.problem_id))


def render_submission_csv(rows: Sequence[SubmissionRow]) -> str:
    """
    Serialize strict Kaggle submission rows to CSV text with header: id,answer.
    """
    ordered = sort_submission_rows(rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["id", "answer"], lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow({"id": row.id, "answer": row.answer})
    return buffer.getvalue()


def render_debug_sidecar_csv(rows: Sequence[SubmissionDebugRow]) -> str:
    """
    Serialize debug/evidence rows to CSV text. This is intentionally separate from
    Kaggle submission rows and may contain richer metadata.
    """
    ordered = sort_debug_rows(rows)
    buffer = io.StringIO()
    fieldnames = [
        "problem_id",
        "submission_answer",
        "confidence",
        "method_used",
        "winning_answer_raw",
        "winning_answer_canonical",
        "num_branches_generated",
        "num_branches_survived",
        "solve_time_sec",
        "warnings",
        "rationale_summary",
        "metadata",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        payload = row.model_dump()
        payload["warnings"] = "|".join(payload["warnings"])
        payload["metadata"] = _stable_dict_string(payload["metadata"])
        writer.writerow(payload)
    return buffer.getvalue()


def submission_rows_from_records(records: Iterable[Mapping[str, Any]]) -> list[SubmissionRow]:
    """
    Validate pre-built row-like mappings and convert them into typed SubmissionRow objects.
    """
    rows: list[SubmissionRow] = []
    for record in records:
        if "id" not in record or "answer" not in record:
            raise SubmissionFormattingError(
                f"Submission record must contain 'id' and 'answer' keys. Got keys={sorted(record.keys())}"
            )
        rows.append(
            SubmissionRow(
                id=_validate_problem_id(record["id"], strict=True),
                answer=_coerce_answer_to_submission_int(record["answer"], enforce_answer_range=True),
            )
        )
    _ensure_unique_problem_ids(rows)
    return sort_submission_rows(rows)


def _extract_prediction(selection: FinalSelectionResult | FinalPrediction) -> FinalPrediction:
    if isinstance(selection, FinalPrediction):
        return selection
    if isinstance(selection, FinalSelectionResult):
        return selection.prediction
    raise SubmissionFormattingError(
        f"Unsupported final-selection type for submission formatting: {type(selection)!r}"
    )


def _extract_final_answer(selection: FinalSelectionResult | FinalPrediction) -> Any:
    """
    Extract the selected answer without mutating the source object.

    Priority:
    1. FinalPrediction.final_answer (already typed by aggregation)
    2. For FinalSelectionResult only, selected cluster canonical/raw answer for diagnostics,
       but only if final_answer is missing or invalid.
    """
    prediction = _extract_prediction(selection)
    candidate_answer: Any = prediction.final_answer

    try:
        return _coerce_answer_to_submission_int(candidate_answer, enforce_answer_range=True)
    except SubmissionFormattingError:
        if isinstance(selection, FinalSelectionResult):
            cluster = selection.prediction.winning_cluster
            fallback_candidates = (
                cluster.answer_canonical,
                cluster.answer,
            )
            for candidate in fallback_candidates:
                try:
                    return _coerce_answer_to_submission_int(candidate, enforce_answer_range=True)
                except SubmissionFormattingError:
                    continue
        raise


def _coerce_answer_to_submission_int(
    answer: Any,
    *,
    enforce_answer_range: bool,
) -> int:
    """
    Safely convert an answer-like object to the competition integer answer.

    Rules:
    - bool is rejected explicitly
    - int is accepted directly
    - exact integer strings are accepted
    - float is accepted only if mathematically integral; otherwise rejected
    - all other lossy conversions are rejected

    This avoids silently discarding information from canonicalized answers like
    fractions or symbolic strings.
    """
    if isinstance(answer, bool):
        raise SubmissionFormattingError("Boolean answers are invalid for competition submission.")

    if isinstance(answer, int):
        value = answer
    elif isinstance(answer, str):
        text = answer.strip()
        if not text:
            raise SubmissionFormattingError("Empty answer string cannot be converted safely.")
        if not _INTEGER_RE.fullmatch(text):
            raise SubmissionFormattingError(
                f"Unsafe non-integer answer string for submission: {answer!r}"
            )
        value = int(text)
    elif isinstance(answer, float):
        if not answer.is_integer():
            raise SubmissionFormattingError(
                f"Unsafe non-integral float answer for submission: {answer!r}"
            )
        value = int(answer)
    else:
        raise SubmissionFormattingError(
            f"Unsupported answer type for submission: type={type(answer)!r}, value={answer!r}"
        )

    if enforce_answer_range and not (ANSWER_MIN <= value <= ANSWER_MAX):
        raise SubmissionFormattingError(
            f"Answer {value} is outside competition range [{ANSWER_MIN}, {ANSWER_MAX}]."
        )
    return value


def _validate_problem_id(problem_id: Any, *, strict: bool) -> str:
    if problem_id is None:
        raise SubmissionFormattingError("Problem id cannot be None.")
    value = str(problem_id).strip()
    if not value:
        raise SubmissionFormattingError("Problem id cannot be empty.")
    if strict and "," in value:
        raise SubmissionFormattingError(f"Problem id contains invalid CSV delimiter: {value!r}")
    return value


def _ensure_unique_problem_ids(rows: Sequence[SubmissionRow]) -> None:
    seen: set[str] = set()
    dupes: list[str] = []
    for row in rows:
        if row.id in seen:
            dupes.append(row.id)
        seen.add(row.id)
    if dupes:
        ordered_dupes = sorted(set(dupes), key=_problem_id_sort_key)
        raise SubmissionFormattingError(
            f"Duplicate problem ids in submission rows: {ordered_dupes}"
        )


def _validate_expected_problem_ids(
    rows: Sequence[SubmissionRow],
    expected_problem_ids: Sequence[str],
) -> None:
    expected = [_validate_problem_id(pid, strict=True) for pid in expected_problem_ids]
    actual = [row.id for row in rows]
    if len(actual) != len(expected):
        raise SubmissionFormattingError(
            f"Submission row count {len(actual)} does not match expected count {len(expected)}."
        )

    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set, key=_problem_id_sort_key)
    extra = sorted(actual_set - expected_set, key=_problem_id_sort_key)
    if missing or extra:
        raise SubmissionFormattingError(
            f"Submission problem-id mismatch. missing={missing} extra={extra}"
        )


def _problem_id_sort_key(problem_id: str) -> tuple[int, int | str, str]:
    """
    Deterministic helper that sorts numeric ids numerically and falls back to lexical order.

    Examples:
    - "2" before "10"
    - "A1" by lexical order after numeric ids
    """
    value = problem_id.strip()
    if value.isdigit():
        return (0, int(value), value)
    return (1, value, value)


def _stable_dict_string(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "{}"
    parts = []
    for key in sorted(payload):
        parts.append(f"{key}={payload[key]!r}")
    return "{" + ", ".join(parts) + "}"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "SubmissionArtifactBundle",
    "SubmissionDebugRow",
    "SubmissionFormatterConfig",
    "SubmissionFormattingError",
    "SubmissionRow",
    "format_debug_row",
    "format_submission_batch",
    "format_submission_row",
    "render_debug_sidecar_csv",
    "render_submission_csv",
    "sort_debug_rows",
    "sort_submission_rows",
    "submission_rows_from_records",
]
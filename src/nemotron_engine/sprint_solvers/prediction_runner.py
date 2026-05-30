"""Deterministic Sprint 1 prediction runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate

from .arithmetic_solver import solve_arithmetic_problem
from .bit_solver import solve_bit_problem
from .dsl_synthesizer import solve_dsl_problem
from .family_router import route_family
from .mapping_solver import solve_mapping_problem
from .modular_solver import solve_modular_problem
from .problem_parser import parse_problem
from .sequence_solver import solve_sequence_problem
from .solver_base import ParsedProblem, SolverResult
from .string_solver import solve_string_problem


class PredictionRunnerError(ValueError):
    """Raised when a Sprint prediction run or artifact is invalid."""


_ARTIFACT_NAMES = (
    "solver_attempts.jsonl",
    "predictions.jsonl",
    "disagreements.jsonl",
    "abstentions.jsonl",
    "family_metrics.json",
)
_FORBIDDEN_ROW_FIELDS = {"expected_answer", "gold", "target_answer", "correct_answer"}
_LEAKAGE_ERROR_MARKER = "forbidden target/gold leakage"
_SOLVERS: dict[str, Callable[[ParsedProblem], SolverResult]] = {
    "bit_solver": solve_bit_problem,
    "mapping_solver": solve_mapping_problem,
    "arithmetic_solver": solve_arithmetic_problem,
    "sequence_solver": solve_sequence_problem,
    "string_solver": solve_string_problem,
    "modular_solver": solve_modular_problem,
    "dsl_synthesizer": solve_dsl_problem,
}
_SOLVER_FAMILY = {
    "bit_solver": "bit",
    "mapping_solver": "mapping",
    "arithmetic_solver": "arithmetic",
    "sequence_solver": "sequence",
    "string_solver": "string",
    "modular_solver": "modular",
    "dsl_synthesizer": "dsl",
}


@dataclass(frozen=True)
class PredictionRecord:
    id: str
    answer: int
    source: str
    solver: str
    family: str
    prediction_hash: str = ""

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise PredictionRunnerError("prediction id must be non-empty.")
        if self.source not in {"symbolic", "model_fallback"}:
            raise PredictionRunnerError("prediction source is invalid.")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "answer", normalize_answer_candidate(self.answer))
        object.__setattr__(self, "solver", str(self.solver))
        object.__setattr__(self, "family", str(self.family))
        _set_or_check_hash(self, "prediction_hash")


@dataclass(frozen=True)
class SolverAttemptRecord:
    problem_id: str
    solver_name: str
    status: str
    prediction: str | None
    reason: str
    candidates_tested: int
    verified_count: int
    rejected_count: int
    result_hash: str

    def __post_init__(self) -> None:
        if not str(self.problem_id).strip() or not str(self.solver_name).strip():
            raise PredictionRunnerError("solver attempt identifiers must be non-empty.")
        object.__setattr__(self, "problem_id", str(self.problem_id))
        object.__setattr__(self, "solver_name", str(self.solver_name))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "prediction", None if self.prediction is None else str(self.prediction))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "candidates_tested", int(self.candidates_tested))
        object.__setattr__(self, "verified_count", int(self.verified_count))
        object.__setattr__(self, "rejected_count", int(self.rejected_count))
        object.__setattr__(self, "result_hash", str(self.result_hash))


@dataclass(frozen=True)
class DisagreementRecord:
    problem_id: str
    solver_predictions: Mapping[str, Any]
    solver_names: tuple[str, ...]
    reason: str
    disagreement_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", str(self.problem_id))
        object.__setattr__(self, "solver_predictions", dict(self.solver_predictions))
        object.__setattr__(self, "solver_names", tuple(str(item) for item in self.solver_names))
        object.__setattr__(self, "reason", str(self.reason))
        _set_or_check_hash(self, "disagreement_hash")


@dataclass(frozen=True)
class AbstentionRecord:
    problem_id: str
    reason: str
    parser_failed: bool
    fallback_allowed: bool
    abstention_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", str(self.problem_id))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "parser_failed", bool(self.parser_failed))
        object.__setattr__(self, "fallback_allowed", bool(self.fallback_allowed))
        _set_or_check_hash(self, "abstention_hash")


@dataclass(frozen=True)
class PredictionRunReport:
    total_rows: int
    parsed_count: int
    prediction_count: int
    symbolic_prediction_count: int
    fallback_prediction_count: int
    abstention_count: int
    disagreement_count: int
    invalid_prediction_count: int
    solver_status_counts: Mapping[str, Mapping[str, int]]
    family_counts: Mapping[str, int]
    report_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "total_rows",
            "parsed_count",
            "prediction_count",
            "symbolic_prediction_count",
            "fallback_prediction_count",
            "abstention_count",
            "disagreement_count",
            "invalid_prediction_count",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise PredictionRunnerError(f"{field_name} cannot be negative.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "solver_status_counts", _normalize_nested_counts(self.solver_status_counts))
        object.__setattr__(self, "family_counts", _normalize_counts(self.family_counts))
        if self.prediction_count != self.symbolic_prediction_count + self.fallback_prediction_count:
            raise PredictionRunnerError("prediction_count must equal symbolic plus fallback predictions.")
        if self.total_rows != self.prediction_count + self.abstention_count:
            raise PredictionRunnerError("total_rows must equal predictions plus abstentions.")
        if self.disagreement_count > self.abstention_count:
            raise PredictionRunnerError("disagreement_count cannot exceed abstention_count.")
        if self.invalid_prediction_count > self.abstention_count:
            raise PredictionRunnerError("invalid_prediction_count cannot exceed abstention_count.")
        _set_or_check_hash(self, "report_hash")


def run_predictions(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    allow_model_fallback: bool = False,
    model_fallback: Callable[..., Any] | None = None,
) -> PredictionRunReport:
    """Run routed symbolic solvers and write deterministic Sprint artifacts."""

    rows = _read_input_jsonl(input_path)
    out_dir = _prepare_output_dir(output_dir)
    artifacts = {name: _artifact_path(out_dir, name) for name in _ARTIFACT_NAMES}

    attempts: list[SolverAttemptRecord] = []
    predictions: list[PredictionRecord] = []
    disagreements: list[DisagreementRecord] = []
    abstentions: list[AbstentionRecord] = []
    solver_status_counts: dict[str, dict[str, int]] = {}
    family_counts: dict[str, int] = {}
    parsed_count = 0
    invalid_prediction_count = 0

    for index, row in enumerate(rows, start=1):
        problem_id, prompt = _safe_problem_identity(row, index)
        parsed: ParsedProblem | None = None
        parser_failed = False
        parser_failure_reason = ""
        route_solvers: tuple[str, ...] = ("dsl_synthesizer",)
        route_families: tuple[str, ...] = ()
        results: list[tuple[str, SolverResult]] = []

        try:
            parsed = parse_problem({"id": problem_id, "prompt": prompt})
            parsed_count += 1
            decision = route_family(parsed)
            route_solvers = tuple(name for name in decision.ordered_solvers if name in _SOLVERS)
            route_families = decision.matched_families or ("unknown",)
        except Exception as exc:
            parser_failed = True
            parser_failure_reason = str(exc)

        for family in route_families or ("parse_failed",):
            family_counts[family] = family_counts.get(family, 0) + 1

        if parsed is not None:
            for solver_name in route_solvers:
                result = _SOLVERS[solver_name](parsed)
                results.append((solver_name, result))
                attempts.append(_attempt_from_result(problem_id, solver_name, result))
                _increment_status(solver_status_counts, solver_name, result.status)

        disagreement = _disagreement_record(problem_id, results)
        if disagreement is not None:
            disagreements.append(disagreement)
            abstentions.append(AbstentionRecord(problem_id, "symbolic_disagreement", parser_failed=False, fallback_allowed=False))
            continue

        solved = [(solver_name, result) for solver_name, result in results if result.status == "solved" and result.prediction is not None]
        if solved:
            prediction_text = solved[0][1].prediction
            try:
                answer = normalize_answer_candidate(prediction_text)
            except ValueError:
                invalid_prediction_count += 1
                abstentions.append(AbstentionRecord(problem_id, "invalid_symbolic_prediction", parser_failed=False, fallback_allowed=False))
                continue
            solver_names = tuple(solver_name for solver_name, _ in solved)
            predictions.append(
                PredictionRecord(
                    id=problem_id,
                    answer=answer,
                    source="symbolic",
                    solver="+".join(solver_names),
                    family="+".join(_SOLVER_FAMILY.get(name, "unknown") for name in solver_names),
                )
            )
            continue

        parser_forbidden_leakage = parser_failed and _LEAKAGE_ERROR_MARKER in parser_failure_reason
        fallback_allowed = bool(allow_model_fallback and model_fallback is not None and not parser_forbidden_leakage)
        if fallback_allowed:
            fallback_value = _call_model_fallback(model_fallback, problem_id=problem_id, raw_prompt=prompt, parsed_problem=parsed)
            try:
                answer = normalize_answer_candidate(fallback_value)
            except ValueError:
                invalid_prediction_count += 1
                abstentions.append(AbstentionRecord(problem_id, "invalid_fallback_prediction", parser_failed=parser_failed, fallback_allowed=True))
                continue
            predictions.append(PredictionRecord(id=problem_id, answer=answer, source="model_fallback", solver="model_fallback", family="fallback"))
        else:
            if parser_forbidden_leakage:
                reason = "parser_forbidden_leakage"
            else:
                reason = "parser_failed" if parser_failed else "no_symbolic_solution"
            abstentions.append(AbstentionRecord(problem_id, reason, parser_failed=parser_failed, fallback_allowed=False))

    report = PredictionRunReport(
        total_rows=len(rows),
        parsed_count=parsed_count,
        prediction_count=len(predictions),
        symbolic_prediction_count=sum(1 for item in predictions if item.source == "symbolic"),
        fallback_prediction_count=sum(1 for item in predictions if item.source == "model_fallback"),
        abstention_count=len(abstentions),
        disagreement_count=len(disagreements),
        invalid_prediction_count=invalid_prediction_count,
        solver_status_counts=solver_status_counts,
        family_counts=family_counts,
    )

    _write_jsonl(artifacts["solver_attempts.jsonl"], (_record_to_dict(item) for item in attempts))
    _write_jsonl(artifacts["predictions.jsonl"], (_record_to_dict(item) for item in predictions))
    _write_jsonl(artifacts["disagreements.jsonl"], (_record_to_dict(item) for item in disagreements))
    _write_jsonl(artifacts["abstentions.jsonl"], (_record_to_dict(item) for item in abstentions))
    _write_json(artifacts["family_metrics.json"], _record_to_dict(report))
    validate_prediction_run_report(report, out_dir)
    return report


def validate_prediction_run_report(report: PredictionRunReport, output_dir: str | Path | None = None) -> PredictionRunReport:
    """Validate report hash and, when provided, artifact counts."""

    _validate_hash(report, "report_hash")
    if output_dir is None:
        return report
    out_dir = Path(output_dir).resolve()
    predictions = _read_artifact_jsonl(_artifact_path(out_dir, "predictions.jsonl"))
    disagreements = _read_artifact_jsonl(_artifact_path(out_dir, "disagreements.jsonl"))
    abstentions = _read_artifact_jsonl(_artifact_path(out_dir, "abstentions.jsonl"))
    attempts = _read_artifact_jsonl(_artifact_path(out_dir, "solver_attempts.jsonl"))
    metrics = _read_json(_artifact_path(out_dir, "family_metrics.json"))
    if attempts is None:
        raise PredictionRunnerError("solver attempts artifact missing.")
    disagreement_ids = {str(row["problem_id"]) for row in disagreements}
    prediction_ids = {str(row["id"]) for row in predictions}
    if disagreement_ids & prediction_ids:
        raise PredictionRunnerError("prediction exists for a disagreement problem.")
    symbolic_count = 0
    fallback_count = 0
    for row in predictions:
        try:
            normalize_answer_candidate(row["answer"])
        except (KeyError, ValueError) as exc:
            raise PredictionRunnerError("prediction artifact contains invalid answer.") from exc
        source = row.get("source")
        if source == "symbolic":
            symbolic_count += 1
        elif source == "model_fallback":
            fallback_count += 1
        else:
            raise PredictionRunnerError("prediction artifact contains invalid source.")
    invalid_count = sum(1 for row in abstentions if str(row.get("reason", "")).startswith("invalid_"))
    recomputed = {
        "prediction_count": len(predictions),
        "symbolic_prediction_count": symbolic_count,
        "fallback_prediction_count": fallback_count,
        "abstention_count": len(abstentions),
        "disagreement_count": len(disagreements),
        "invalid_prediction_count": invalid_count,
        "total_rows": len(predictions) + len(abstentions),
    }
    for field_name, value in recomputed.items():
        if getattr(report, field_name) != value:
            raise PredictionRunnerError(f"{field_name} does not match artifacts.")
        if metrics.get(field_name) != value:
            raise PredictionRunnerError(f"family_metrics {field_name} does not match artifacts.")
    if metrics.get("report_hash") != report.report_hash:
        raise PredictionRunnerError("family_metrics report_hash does not match report.")
    forbidden = _FORBIDDEN_ROW_FIELDS & set().union(*(row.keys() for row in predictions), set())
    if forbidden:
        raise PredictionRunnerError("prediction artifact contains forbidden eval-only fields.")
    return report


def _attempt_from_result(problem_id: str, solver_name: str, result: SolverResult) -> SolverAttemptRecord:
    return SolverAttemptRecord(
        problem_id=problem_id,
        solver_name=solver_name,
        status=result.status,
        prediction=result.prediction,
        reason=_result_reason(result),
        candidates_tested=len(result.verified_candidates) + len(result.rejected_candidates),
        verified_count=len(result.verified_candidates),
        rejected_count=len(result.rejected_candidates),
        result_hash=result.result_hash,
    )


def _result_reason(result: SolverResult) -> str:
    reasons: list[str] = []
    reasons.extend(str(item) for item in result.errors if str(item).strip())
    for candidate in result.rejected_candidates:
        reason = candidate.metadata.get("rejection_reason")
        if reason and str(reason) not in reasons:
            reasons.append(str(reason))
    if reasons:
        return ",".join(reasons)
    return result.status


def _disagreement_record(problem_id: str, results: list[tuple[str, SolverResult]]) -> DisagreementRecord | None:
    disagreeing = [(name, result) for name, result in results if result.status == "disagreement"]
    solved = [(name, result) for name, result in results if result.status == "solved" and result.prediction is not None]
    solved_predictions = {result.prediction for _, result in solved}
    if len(solved_predictions) <= 1 and not disagreeing:
        return None
    solver_predictions: dict[str, Any] = {}
    for solver_name, result in solved:
        solver_predictions[solver_name] = result.prediction
    for solver_name, result in disagreeing:
        solver_predictions[solver_name] = sorted({candidate.target_prediction for candidate in result.verified_candidates})
    reason = "symbolic_solver_disagreement" if disagreeing else "solved_solvers_disagree"
    return DisagreementRecord(
        problem_id=problem_id,
        solver_predictions=solver_predictions,
        solver_names=tuple(name for name, _ in [*solved, *disagreeing]),
        reason=reason,
    )


def _call_model_fallback(model_fallback: Callable[..., Any] | None, *, problem_id: str, raw_prompt: str, parsed_problem: ParsedProblem | None) -> Any:
    if model_fallback is None:
        return None
    return model_fallback(problem_id=problem_id, raw_prompt=raw_prompt, parsed_problem=parsed_problem)


def _safe_problem_identity(row: Mapping[str, Any], index: int) -> tuple[str, str]:
    if not isinstance(row, Mapping):
        raise PredictionRunnerError(f"input row {index} is not an object.")
    problem_id = row.get("id", row.get("problem_id"))
    prompt = row.get("prompt", row.get("raw_prompt"))
    if problem_id is None or not str(problem_id).strip():
        raise PredictionRunnerError(f"input row {index} is missing id/problem_id.")
    if prompt is None or not str(prompt).strip():
        raise PredictionRunnerError(f"input row {index} is missing prompt/raw_prompt.")
    return str(problem_id).strip(), str(prompt).strip()


def _prepare_output_dir(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _artifact_path(output_dir: Path, name: str) -> Path:
    if name not in _ARTIFACT_NAMES:
        raise PredictionRunnerError(f"unknown artifact: {name}")
    path = (output_dir / name).resolve()
    if path.parent != output_dir.resolve():
        raise PredictionRunnerError("refusing to write outside output_dir.")
    return path


def _read_input_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PredictionRunnerError(f"malformed JSONL on line {line_number}: {exc.msg}") from exc
                if not isinstance(item, Mapping):
                    raise PredictionRunnerError(f"input row {line_number} is not an object.")
                rows.append(item)
    except OSError as exc:
        raise PredictionRunnerError(f"cannot read input JSONL: {input_path}") from exc
    return rows


def _read_artifact_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows: list[dict[str, Any]] = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PredictionRunnerError(f"malformed JSONL in {path.name} line {line_number}: {exc.msg}") from exc
                if not isinstance(item, dict):
                    raise PredictionRunnerError(f"malformed JSONL in {path.name} line {line_number}: row is not an object.")
                rows.append(item)
            return rows
    except OSError as exc:
        raise PredictionRunnerError(f"cannot read artifact: {path.name}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionRunnerError(f"cannot read artifact: {path.name}") from exc
    if not isinstance(item, dict):
        raise PredictionRunnerError(f"artifact {path.name} is not a JSON object.")
    return item


def _write_jsonl(path: Path, rows: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")


def _record_to_dict(record: Any) -> dict[str, Any]:
    return {item.name: getattr(record, item.name) for item in fields(record)}


def _increment_status(counts: dict[str, dict[str, int]], solver_name: str, status: str) -> None:
    solver_counts = counts.setdefault(solver_name, {})
    solver_counts[status] = solver_counts.get(status, 0) + 1


def _normalize_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def _normalize_nested_counts(counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {str(key): _normalize_counts(value) for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = _payload_hash(instance, hash_field)
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise PredictionRunnerError(f"{hash_field} does not match payload.")


def _validate_hash(instance: object, hash_field: str) -> None:
    if getattr(instance, hash_field) != _payload_hash(instance, hash_field):
        raise PredictionRunnerError(f"{hash_field} does not match payload.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "AbstentionRecord",
    "DisagreementRecord",
    "PredictionRecord",
    "PredictionRunReport",
    "PredictionRunnerError",
    "SolverAttemptRecord",
    "run_predictions",
    "validate_prediction_run_report",
]

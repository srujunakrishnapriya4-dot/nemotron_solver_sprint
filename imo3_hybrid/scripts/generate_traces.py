from __future__ import annotations

"""
Production CLI for generating raw reasoning traces.

This script is intentionally more than a casual loop over examples. It owns:
- deterministic input selection
- explicit runtime budget handling
- solve invocation through the online inference stack
- raw trace capture into a distillation-consumable JSONL format
- provenance / manifest emission
- partial-failure recording without corrupting the full run
- deterministic artifact naming and summary reporting

The raw trace JSONL written by this script is directly consumable by
src.offline.trace_distillation.normalize_trace_source / distill_traces because
it emits mapping records with the fields that module accepts.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW_TRACE_SCHEMA_VERSION = "raw_reasoning_trace.v1"
RUN_MANIFEST_SCHEMA_VERSION = "trace_generation_run.v1"


@dataclass(frozen=True)
class ProblemRecord:
    problem_id: str
    problem_text: str
    ordinal: int
    source_path: str
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalBudgetTracker:
    total_budget_s: float
    reserve_s: float = 0.0
    spent_s: float = 0.0

    def remaining_s(self) -> float:
        return max(0.0, float(self.total_budget_s) - float(self.reserve_s) - float(self.spent_s))

    def per_problem_cap(self, remaining_problems: int) -> float:
        if remaining_problems <= 0:
            return 0.0
        return max(0.0, self.remaining_s() / float(remaining_problems))

    def with_spent(self, additional_spent_s: float) -> "LocalBudgetTracker":
        return LocalBudgetTracker(
            total_budget_s=float(self.total_budget_s),
            reserve_s=float(self.reserve_s),
            spent_s=max(0.0, float(self.spent_s) + float(additional_spent_s)),
        )


@dataclass
class RunState:
    run_dir: Path
    raw_trace_path: Path
    solve_results_path: Path
    failures_path: Path
    summary_path: Path
    manifest_path: Path
    distillation_dir: Path
    raw_trace_count: int = 0
    solve_result_count: int = 0
    failure_count: int = 0
    problem_success_count: int = 0
    problem_failure_count: int = 0
    problem_partial_count: int = 0


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        self._handle.write(_stable_json(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _coerce_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _coerce_jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _coerce_jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return _coerce_jsonable(vars(value))
    if hasattr(value, "value"):
        return getattr(value, "value")
    return _normalize_text(str(value))


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw reasoning traces through the current online inference stack.",
    )
    parser.add_argument("--input-path", required=True, help="Dataset file: CSV, JSONL, JSON, or TXT.")
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "jsonl", "json", "txt"),
        default="auto",
        help="Explicitly set the input format. Default: infer from extension.",
    )
    parser.add_argument("--id-field", default="id", help="Problem id field for CSV/JSON inputs.")
    parser.add_argument(
        "--text-field",
        default="problem",
        help="Problem text field for CSV/JSON inputs. Falls back to 'text' when missing.",
    )
    parser.add_argument(
        "--selection-ids",
        nargs="*",
        default=None,
        help="Explicit problem ids to keep. Order is preserved from the input file.",
    )
    parser.add_argument(
        "--selection-ids-file",
        default=None,
        help="Optional newline-delimited file containing problem ids to keep.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive start index after loading.")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end index after loading.")
    parser.add_argument("--max-problems", type=int, default=None, help="Maximum problems to run.")
    parser.add_argument(
        "--shuffle-selection",
        action="store_true",
        help="Shuffle the selected problems deterministically before truncation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for selection and runtime.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Explicit artifact root directory. A deterministic run subdirectory is created inside it.",
    )
    parser.add_argument(
        "--run-name",
        default="generate_traces",
        help="Human-readable run prefix used in deterministic directory naming.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow removing a pre-existing deterministic run directory before writing artifacts.",
    )
    parser.add_argument(
        "--continue-on-problem-failure",
        action="store_true",
        default=False,
        help="Continue after problem-level failures instead of stopping the run.",
    )
    parser.add_argument(
        "--emit-distilled-artifact",
        action="store_true",
        help="Also run src.offline.trace_distillation on the raw JSONL after generation.",
    )
    parser.add_argument(
        "--include-solve-debug",
        action="store_true",
        help="Persist solve-stage summaries/debug metadata into solve_results.jsonl.",
    )
    parser.add_argument(
        "--total-time-budget-s",
        type=float,
        default=None,
        help="Optional total wall-clock budget for the entire run.",
    )
    parser.add_argument(
        "--reserve-time-s",
        type=float,
        default=0.0,
        help="Reserved wall-clock buffer held back from the total run budget.",
    )
    parser.add_argument(
        "--per-problem-time-limit-s",
        type=float,
        default=None,
        help="Reserved for future engine preemption support. The current script rejects this flag because the runtime cannot enforce it safely.",
    )
    parser.add_argument(
        "--engine-config-json",
        default=None,
        help="Optional JSON object merged into InferenceEngineConfig.",
    )
    parser.add_argument(
        "--strict-engine-import",
        action="store_true",
        help="Fail immediately if the online inference stack cannot be imported. Default behavior records the bootstrap failure and exits cleanly.",
    )
    return parser.parse_args(argv)


def _infer_input_format(path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix in {".json"}:
        return "json"
    if suffix in {".txt"}:
        return "txt"
    raise ValueError(f"Unsupported input extension for auto-detection: {path}")


def _read_selection_ids(args: argparse.Namespace) -> set[str]:
    chosen: set[str] = set()
    for item in args.selection_ids or []:
        text = _normalize_text(item)
        if text:
            chosen.add(text)
    if args.selection_ids_file:
        selection_path = Path(args.selection_ids_file)
        for line in selection_path.read_text(encoding="utf-8").splitlines():
            text = _normalize_text(line)
            if text:
                chosen.add(text)
    return chosen


def _load_problem_records(args: argparse.Namespace) -> list[ProblemRecord]:
    input_path = Path(args.input_path).resolve()
    fmt = _infer_input_format(input_path, args.input_format)
    id_field = args.id_field
    text_field = args.text_field

    records: list[ProblemRecord] = []
    if fmt == "csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                problem_text = _normalize_text(row.get(text_field) or row.get("text") or row.get("problem_text"))
                if not problem_text:
                    continue
                problem_id = _normalize_text(row.get(id_field) or row.get("problem_id") or f"problem_{idx:05d}")
                meta = {k: v for k, v in row.items() if k not in {id_field, text_field, "text", "problem_text"}}
                records.append(
                    ProblemRecord(
                        problem_id=problem_id,
                        problem_text=problem_text,
                        ordinal=idx,
                        source_path=str(input_path),
                        source_metadata=_coerce_jsonable(meta),
                    )
                )
    elif fmt == "jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if isinstance(row, str):
                    problem_text = _normalize_text(row)
                    problem_id = f"problem_{idx:05d}"
                    meta: dict[str, Any] = {}
                else:
                    problem_text = _normalize_text(row.get(text_field) or row.get("text") or row.get("problem_text"))
                    if not problem_text:
                        continue
                    problem_id = _normalize_text(row.get(id_field) or row.get("problem_id") or f"problem_{idx:05d}")
                    meta = {k: v for k, v in row.items() if k not in {id_field, text_field, "text", "problem_text"}}
                records.append(
                    ProblemRecord(
                        problem_id=problem_id,
                        problem_text=problem_text,
                        ordinal=idx,
                        source_path=str(input_path),
                        source_metadata=_coerce_jsonable(meta),
                    )
                )
    elif fmt == "json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            maybe_records = payload.get("records") or payload.get("problems") or payload.get("items")
            if maybe_records is None:
                raise ValueError("JSON input must be a list or contain 'records'/'problems'/'items'.")
            payload = maybe_records
        if not isinstance(payload, list):
            raise ValueError("JSON input must decode to a list of records.")
        for idx, row in enumerate(payload):
            if isinstance(row, str):
                problem_text = _normalize_text(row)
                problem_id = f"problem_{idx:05d}"
                meta = {}
            else:
                problem_text = _normalize_text(row.get(text_field) or row.get("text") or row.get("problem_text"))
                if not problem_text:
                    continue
                problem_id = _normalize_text(row.get(id_field) or row.get("problem_id") or f"problem_{idx:05d}")
                meta = {k: v for k, v in row.items() if k not in {id_field, text_field, "text", "problem_text"}}
            records.append(
                ProblemRecord(
                    problem_id=problem_id,
                    problem_text=problem_text,
                    ordinal=idx,
                    source_path=str(input_path),
                    source_metadata=_coerce_jsonable(meta),
                )
            )
    elif fmt == "txt":
        for idx, line in enumerate(input_path.read_text(encoding="utf-8").splitlines()):
            problem_text = _normalize_text(line)
            if not problem_text:
                continue
            records.append(
                ProblemRecord(
                    problem_id=f"problem_{idx:05d}",
                    problem_text=problem_text,
                    ordinal=idx,
                    source_path=str(input_path),
                    source_metadata={},
                )
            )
    else:
        raise ValueError(f"Unsupported input format: {fmt}")

    selection_ids = _read_selection_ids(args)
    if selection_ids:
        records = [record for record in records if record.problem_id in selection_ids]

    start_index = max(0, int(args.start_index or 0))
    end_index = None if args.end_index is None else max(start_index, int(args.end_index))
    records = records[start_index:end_index]

    if args.shuffle_selection:
        rng = random.Random(int(args.seed))
        records = list(records)
        rng.shuffle(records)

    if args.max_problems is not None:
        records = records[: max(0, int(args.max_problems))]

    return records


def _run_identity(args: argparse.Namespace, records: Sequence[ProblemRecord]) -> str:
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_name": _normalize_text(args.run_name),
        "seed": int(args.seed),
        "input_path": str(Path(args.input_path).resolve()),
        "input_format": _infer_input_format(Path(args.input_path).resolve(), args.input_format),
        "id_field": args.id_field,
        "text_field": args.text_field,
        "selected_problem_ids": [record.problem_id for record in records],
        "problem_count": len(records),
        "time_budget": {
            "total_time_budget_s": args.total_time_budget_s,
            "reserve_time_s": args.reserve_time_s,
            "per_problem_time_limit_s": args.per_problem_time_limit_s,
        },
        "continue_on_problem_failure": bool(args.continue_on_problem_failure),
        "emit_distilled_artifact": bool(args.emit_distilled_artifact),
        "engine_config_json": args.engine_config_json,
    }
    digest = _sha1_text(_stable_json(payload))[:16]
    return f"{_normalize_text(args.run_name).replace(' ', '_')}_{digest}"


def _prepare_run_state(args: argparse.Namespace, run_id: str) -> RunState:
    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / run_id
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Deterministic output directory already exists: {run_dir}. Use --overwrite to replace it."
            )
        for child in sorted(run_dir.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunState(
        run_dir=run_dir,
        raw_trace_path=run_dir / "raw_traces.jsonl",
        solve_results_path=run_dir / "solve_results.jsonl",
        failures_path=run_dir / "failures.jsonl",
        summary_path=run_dir / "summary.json",
        manifest_path=run_dir / "manifest.json",
        distillation_dir=run_dir / "distilled",
    )


def _build_engine(args: argparse.Namespace) -> tuple[Any | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "import_ok": False,
        "inference_engine_module": "src.online.inference_engine",
        "engine_config": {},
        "bootstrap_error": None,
    }
    try:
        module = importlib.import_module("src.online.inference_engine")
        InferenceEngine = getattr(module, "InferenceEngine")
        InferenceEngineConfig = getattr(module, "InferenceEngineConfig")
        config_data = {"deterministic_seed": int(args.seed)}
        if args.engine_config_json:
            extra = json.loads(args.engine_config_json)
            if not isinstance(extra, dict):
                raise ValueError("--engine-config-json must decode to a JSON object.")
            config_data.update(extra)
        diagnostics["engine_config"] = _coerce_jsonable(config_data)
        engine = InferenceEngine(config=InferenceEngineConfig(**config_data))
        diagnostics["import_ok"] = True
        diagnostics["engine_type"] = f"{engine.__class__.__module__}.{engine.__class__.__name__}"
        return engine, diagnostics
    except Exception as exc:  # pragma: no cover
        diagnostics["bootstrap_error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if args.strict_engine_import:
            raise
        return None, diagnostics


def _build_tracker(args: argparse.Namespace) -> Any | None:
    if args.total_time_budget_s is None:
        return None
    try:
        adaptive = importlib.import_module("src.online.adaptive_budget")
        tracker_cls = getattr(adaptive, "GlobalBudgetTracker")
        return tracker_cls(total_budget_s=float(args.total_time_budget_s), reserve_s=float(args.reserve_time_s), spent_s=0.0)
    except Exception:
        return LocalBudgetTracker(
            total_budget_s=float(args.total_time_budget_s),
            reserve_s=float(args.reserve_time_s),
            spent_s=0.0,
        )


def _tracker_remaining_s(tracker: Any | None) -> float | None:
    if tracker is None:
        return None
    if hasattr(tracker, "remaining_s"):
        try:
            return float(tracker.remaining_s())
        except Exception:
            return None
    return None


def _tracker_per_problem_cap(tracker: Any | None, remaining_problems: int) -> float | None:
    if tracker is None:
        return None
    if hasattr(tracker, "per_problem_cap"):
        try:
            return float(tracker.per_problem_cap(remaining_problems))
        except Exception:
            return None
    return None


def _tracker_with_spent(tracker: Any | None, elapsed_s: float) -> Any | None:
    if tracker is None:
        return None
    if hasattr(tracker, "with_spent"):
        try:
            return tracker.with_spent(float(elapsed_s))
        except Exception:
            return tracker
    return tracker


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _serialize_stage_records(stage_records: Sequence[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in stage_records or []:
        if hasattr(record, "model_dump"):
            try:
                out.append(_coerce_jsonable(record.model_dump(mode="json")))
                continue
            except TypeError:
                out.append(_coerce_jsonable(record.model_dump()))
                continue
        out.append(_coerce_jsonable(record))
    return out


def _step_to_mapping(step: Any) -> dict[str, Any]:
    return {
        "step_id": getattr(step, "step_id", None),
        "index": int(getattr(step, "index", 0)),
        "kind": _enum_value(getattr(step, "kind", None)),
        "phase": _enum_value(getattr(step, "phase", None)),
        "description": getattr(step, "description", ""),
        "operator_name": getattr(step, "operator_name", None),
        "node_id": getattr(step, "node_id", None),
        "state_fingerprint": getattr(step, "state_fingerprint", None),
        "summary_text": getattr(step, "summary_text", None),
    }


def _evidence_to_mapping(evidence: Any, kind: str) -> dict[str, Any]:
    payload = _coerce_jsonable(evidence)
    if isinstance(payload, dict):
        payload.setdefault("kind", kind)
        return payload
    return {"kind": kind, "value": payload}


def _branch_route_snapshot(branch: Any) -> dict[str, Any]:
    route = getattr(branch, "route", None)
    parsed = getattr(branch, "parsed_problem", None)
    return {
        "problem_type": _coerce_jsonable(getattr(route, "problem_type", {})),
        "archetypes": _coerce_jsonable(getattr(route, "archetypes", {})),
        "difficulty": _coerce_jsonable(_enum_value(getattr(route, "difficulty", None))),
        "difficulty_score": _coerce_jsonable(getattr(route, "difficulty_score", None)),
        "branch_budget": _coerce_jsonable(getattr(route, "branch_budget", None)),
        "retrieval_depth": _coerce_jsonable(getattr(route, "retrieval_depth", None)),
        "answer_type": _coerce_jsonable(getattr(parsed, "answer_type", None)),
        "target": _coerce_jsonable(getattr(parsed, "target", None)),
        "domain": _coerce_jsonable(_enum_value(getattr(parsed, "domain", None))),
    }


def _extract_branch_trace_records(
    solve_result: Any,
    *,
    record: ProblemRecord,
    run_id: str,
    problem_elapsed_s: float,
) -> list[dict[str, Any]]:
    branch_result = getattr(solve_result, "branch_result", None)
    if branch_result is None:
        return []

    extracted: list[dict[str, Any]] = []
    all_branches = list(getattr(branch_result, "all_branches", ()) or ())
    surviving_ids = {getattr(branch, "branch_id", None) for branch in getattr(branch_result, "surviving_branches", ()) or ()}
    critiqued_ids = set(getattr(branch_result, "critiqued_branch_ids", ()) or ())

    winning_branch_ids = set(
        getattr(getattr(solve_result, "final_selection", None), "winning_branch_ids", ())
        or getattr(getattr(branch_result, "metadata", {}), "get", lambda *args, **kwargs: [])("winning_branch_ids", [])
    )

    for order, branch in enumerate(all_branches):
        candidate = branch.current_candidate() if hasattr(branch, "current_candidate") else None
        active_steps = list(branch.active_steps()) if hasattr(branch, "active_steps") else list(getattr(branch, "steps", ()) or ())
        retrieval_items = list(branch.active_retrieval_evidence()) if hasattr(branch, "active_retrieval_evidence") else []
        symbolic_items = list(branch.active_symbolic_evidence()) if hasattr(branch, "active_symbolic_evidence") else []
        verifier_items = list(branch.active_verifier_evidence()) if hasattr(branch, "active_verifier_evidence") else []
        critique_items = list(branch.active_critique_evidence()) if hasattr(branch, "active_critique_evidence") else []
        score_breakdown = getattr(branch, "score_breakdown", None)
        active_cursor = getattr(branch, "active_cursor", None)

        verifier_score = 0.0
        logical_consistency = 0.0
        completeness = 0.0
        repairability = 0.0
        if verifier_items:
            verifier_score = max(float(getattr(item, "probability", 0.0)) for item in verifier_items)
            logical_consistency = max(float(getattr(item, "logical_consistency", 0.0)) for item in verifier_items)
            completeness = max(float(getattr(item, "completeness", 0.0)) for item in verifier_items)
            repairability = max(float(getattr(item, "repairability", 0.0)) for item in verifier_items)
        elif score_breakdown is not None:
            verifier_score = float(getattr(score_breakdown, "verifier_probability", 0.0))

        symbolic_score = 0.0
        symbolic_valid = False
        if symbolic_items:
            symbolic_score = max(float(getattr(item, "score", 0.0)) for item in symbolic_items)
            symbolic_valid = any(bool(getattr(item, "passed", False)) for item in symbolic_items)

        operator_sequence = [str(getattr(step, "operator_name", "")).strip() for step in active_steps if getattr(step, "operator_name", None)]
        operator_sequence = [item for item in operator_sequence if item]

        evidence: list[dict[str, Any]] = []
        evidence.extend(_evidence_to_mapping(item, "retrieval") for item in retrieval_items)
        evidence.extend(_evidence_to_mapping(item, "symbolic") for item in symbolic_items)
        evidence.extend(_evidence_to_mapping(item, "verifier") for item in verifier_items)
        evidence.extend(_evidence_to_mapping(item, "critique") for item in critique_items)

        answer_raw = getattr(candidate, "raw_answer", None) if candidate is not None else None
        answer_canonical = getattr(candidate, "canonical_answer", None) if candidate is not None else None
        branch_id = getattr(branch, "branch_id", f"{record.problem_id}::branch::{order:03d}")
        metadata_get = getattr(getattr(branch, "metadata", {}), "get", lambda *args, **kwargs: None)

        trace_record = {
            "schema_version": RAW_TRACE_SCHEMA_VERSION,
            "record_type": "raw_branch_trace",
            "run_id": run_id,
            "problem_id": record.problem_id,
            "problem_text": record.problem_text,
            "problem_source_metadata": _coerce_jsonable(record.source_metadata),
            "problem_ordinal": int(record.ordinal),
            "branch_id": branch_id,
            "trace_id": branch_id,
            "root_branch_id": getattr(branch, "root_branch_id", None),
            "parent_branch_id": getattr(branch, "parent_branch_id", None),
            "lineage": list(getattr(branch, "lineage", ()) or ()),
            "phase": _enum_value(getattr(branch, "phase", None)),
            "steps": [_step_to_mapping(step) for step in active_steps],
            "evidence": evidence,
            "answer": answer_raw,
            "answer_raw": answer_raw,
            "answer_canonical": answer_canonical,
            "verifier_score": verifier_score,
            "symbolic_score": symbolic_score,
            "logical_consistency": logical_consistency,
            "completeness": completeness,
            "repairability": repairability,
            "branch_score": float(branch.composite_score()) if hasattr(branch, "composite_score") else 0.0,
            "symbolic_valid": bool(symbolic_valid),
            "retrieval_used": bool(retrieval_items),
            "self_critiqued": bool(critique_items or (branch_id in critiqued_ids)),
            "repaired": bool(getattr(active_cursor, "repair_count", 0) or 0),
            "repair_count": int(getattr(active_cursor, "repair_count", 0) or 0),
            "failure_type": _normalize_text(str(metadata_get("failure_type") or metadata_get("failure_reason") or "")) or None,
            "failure_location": _normalize_text(str(metadata_get("failure_location") or "")) or None,
            "operator_sequence": operator_sequence,
            "archetypes": sorted(
                str(key)
                for key, value in (getattr(getattr(branch, "route", None), "archetypes", {}) or {}).items()
                if float(value) > 0.0
            ),
            "route_snapshot": _branch_route_snapshot(branch),
            "is_surviving_branch": bool(branch_id in surviving_ids),
            "is_winning_branch": bool(branch_id in winning_branch_ids),
            "problem_elapsed_s": round(float(problem_elapsed_s), 6),
            "solve_status": _enum_value(getattr(solve_result, "status", None)),
            "stage_records": _serialize_stage_records(getattr(solve_result, "stage_records", None)),
        }
        extracted.append(trace_record)
    return extracted


def _make_problem_result_record(
    solve_result: Any,
    *,
    record: ProblemRecord,
    run_id: str,
    elapsed_s: float,
    include_debug: bool,
) -> dict[str, Any]:
    final_prediction = getattr(solve_result, "final_prediction", None)
    final_selection = getattr(solve_result, "final_selection", None)
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "record_type": "solve_result",
        "run_id": run_id,
        "problem_id": record.problem_id,
        "problem_ordinal": int(record.ordinal),
        "status": _enum_value(getattr(solve_result, "status", None)),
        "ok": bool(getattr(solve_result, "ok", False)),
        "elapsed_s": round(float(elapsed_s), 6),
        "failure": _coerce_jsonable(getattr(solve_result, "failure", None)),
        "final_prediction": _coerce_jsonable(final_prediction),
        "final_selection": _coerce_jsonable(final_selection),
        "route": _coerce_jsonable(getattr(solve_result, "route", None)),
        "retrieved_trace_count": len(getattr(solve_result, "retrieved_traces", []) or []),
        "branch_count": len(getattr(getattr(solve_result, "branch_result", None), "all_branches", []) or []),
        "surviving_branch_count": len(getattr(getattr(solve_result, "branch_result", None), "surviving_branches", []) or []),
        "stage_records": _serialize_stage_records(getattr(solve_result, "stage_records", None)),
    }
    if include_debug:
        payload["debug_artifact"] = _coerce_jsonable(getattr(solve_result, "debug_artifact", None))
    return payload


def _make_failure_record(
    *,
    run_id: str,
    record: ProblemRecord,
    failure_code: str,
    failure_message: str,
    elapsed_s: float,
    exc: BaseException | None = None,
    stage_records: Sequence[Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "record_type": "problem_failure",
        "run_id": run_id,
        "problem_id": record.problem_id,
        "problem_ordinal": int(record.ordinal),
        "elapsed_s": round(float(elapsed_s), 6),
        "failure_code": failure_code,
        "failure_message": failure_message,
        "stage_records": _serialize_stage_records(stage_records),
    }
    if exc is not None:
        payload["exception"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    return payload


def _write_manifest(
    args: argparse.Namespace,
    state: RunState,
    *,
    run_id: str,
    records: Sequence[ProblemRecord],
    engine_diagnostics: Mapping[str, Any],
    tracker: Any | None,
) -> None:
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_by": "scripts/generate_traces.py",
        "repo_root": str(REPO_ROOT),
        "input": {
            "path": str(Path(args.input_path).resolve()),
            "format": _infer_input_format(Path(args.input_path).resolve(), args.input_format),
            "id_field": args.id_field,
            "text_field": args.text_field,
            "selected_problem_ids": [record.problem_id for record in records],
            "problem_count": len(records),
        },
        "seed": int(args.seed),
        "time_budget": {
            "total_time_budget_s": args.total_time_budget_s,
            "reserve_time_s": args.reserve_time_s,
            "per_problem_time_limit_s": args.per_problem_time_limit_s,
            "tracker_type": None if tracker is None else f"{tracker.__class__.__module__}.{tracker.__class__.__name__}",
            "tracker_remaining_s_initial": _tracker_remaining_s(tracker),
        },
        "behavior": {
            "continue_on_problem_failure": bool(args.continue_on_problem_failure),
            "emit_distilled_artifact": bool(args.emit_distilled_artifact),
            "include_solve_debug": bool(args.include_solve_debug),
            "overwrite": bool(args.overwrite),
        },
        "engine": _coerce_jsonable(engine_diagnostics),
        "artifacts": {
            "raw_trace_path": str(state.raw_trace_path),
            "solve_results_path": str(state.solve_results_path),
            "failures_path": str(state.failures_path),
            "summary_path": str(state.summary_path),
            "distillation_dir": str(state.distillation_dir),
        },
    }
    state.manifest_path.write_text(_stable_json(manifest) + "\n", encoding="utf-8")


def _finalize_summary(
    state: RunState,
    *,
    run_id: str,
    started_at: float,
    records: Sequence[ProblemRecord],
    tracker: Any | None,
    engine_diagnostics: Mapping[str, Any],
    distillation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "elapsed_s": round(time.perf_counter() - started_at, 6),
        "problem_count": len(records),
        "raw_trace_count": state.raw_trace_count,
        "solve_result_count": state.solve_result_count,
        "failure_count": state.failure_count,
        "problem_success_count": state.problem_success_count,
        "problem_partial_count": state.problem_partial_count,
        "problem_failure_count": state.problem_failure_count,
        "tracker_remaining_s_final": _tracker_remaining_s(tracker),
        "engine_import_ok": bool(engine_diagnostics.get("import_ok", False)),
        "distillation": _coerce_jsonable(distillation or {}),
        "artifacts": {
            "manifest_path": str(state.manifest_path),
            "raw_trace_path": str(state.raw_trace_path),
            "solve_results_path": str(state.solve_results_path),
            "failures_path": str(state.failures_path),
            "summary_path": str(state.summary_path),
        },
    }
    state.summary_path.write_text(_stable_json(summary) + "\n", encoding="utf-8")
    return summary


def _run_optional_distillation(state: RunState) -> dict[str, Any]:
    from src.offline.trace_distillation import distill_traces, write_distilled_artifact

    raw_records: list[dict[str, Any]] = []
    with state.raw_trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            raw_records.append(json.loads(stripped))

    artifact, report = distill_traces(raw_records)
    write_result = write_distilled_artifact(artifact, state.distillation_dir)
    payload = {
        "artifact_metadata": artifact.metadata.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "write_result": write_result.model_dump(mode="json"),
    }
    (state.distillation_dir / "distillation_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (state.distillation_dir / "distillation_summary.json").write_text(_stable_json(payload) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.per_problem_time_limit_s is not None:
        print(
            "[generate_traces] ERROR: --per-problem-time-limit-s is not supported by the current inference-engine contract. "
            "Use --total-time-budget-s/--reserve-time-s for enforced runtime budgeting.",
            file=sys.stderr,
        )
        return 2
    _set_deterministic_seed(int(args.seed))
    records = _load_problem_records(args)
    if not records:
        raise SystemExit("No problems selected after applying the explicit dataset/input selection.")

    run_id = _run_identity(args, records)
    state = _prepare_run_state(args, run_id)
    engine, engine_diagnostics = _build_engine(args)
    tracker = _build_tracker(args)
    _write_manifest(args, state, run_id=run_id, records=records, engine_diagnostics=engine_diagnostics, tracker=tracker)

    started_at = time.perf_counter()

    with JsonlWriter(state.raw_trace_path) as raw_writer, JsonlWriter(state.solve_results_path) as solve_writer, JsonlWriter(state.failures_path) as failure_writer:
        if engine is None:
            failure_writer.write(
                {
                    "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                    "record_type": "bootstrap_failure",
                    "run_id": run_id,
                    "failure_code": "engine_import_failed",
                    "failure_message": "Could not import the current online inference stack.",
                    "engine_diagnostics": _coerce_jsonable(engine_diagnostics),
                }
            )
            state.failure_count += 1
            summary = _finalize_summary(
                state,
                run_id=run_id,
                started_at=started_at,
                records=records,
                tracker=tracker,
                engine_diagnostics=engine_diagnostics,
                distillation=None,
            )
            print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
            return 2

        for idx, record in enumerate(records):
            remaining_problems = len(records) - idx
            remaining_before = _tracker_remaining_s(tracker)
            cap_before = _tracker_per_problem_cap(tracker, remaining_problems)
            if remaining_before is not None and remaining_before <= 0.0:
                failure_payload = _make_failure_record(
                    run_id=run_id,
                    record=record,
                    failure_code="global_time_budget_exhausted",
                    failure_message="Stopped before solving because the remaining run budget was exhausted.",
                    elapsed_s=0.0,
                )
                failure_writer.write(failure_payload)
                state.failure_count += 1
                state.problem_failure_count += 1
                if not args.continue_on_problem_failure:
                    break
                continue

            solve_started = time.perf_counter()
            try:
                solve_result = engine.solve_problem(
                    record.problem_text,
                    problem_id=record.problem_id,
                    remaining_problems=remaining_problems,
                    budget_tracker=tracker,
                )
                elapsed_s = time.perf_counter() - solve_started
                tracker = _tracker_with_spent(tracker, elapsed_s)

                trace_records = _extract_branch_trace_records(
                    solve_result,
                    record=record,
                    run_id=run_id,
                    problem_elapsed_s=elapsed_s,
                )
                for trace_record in trace_records:
                    raw_writer.write(trace_record)
                    state.raw_trace_count += 1

                solve_payload = _make_problem_result_record(
                    solve_result,
                    record=record,
                    run_id=run_id,
                    elapsed_s=elapsed_s,
                    include_debug=bool(args.include_solve_debug),
                )
                solve_payload["runtime_budget"] = {
                    "remaining_before_s": remaining_before,
                    "per_problem_cap_before_s": cap_before,
                    "remaining_after_s": _tracker_remaining_s(tracker),
                    "per_problem_time_limit_s": args.per_problem_time_limit_s,
                }
                solve_writer.write(solve_payload)
                state.solve_result_count += 1

                status_value = _enum_value(getattr(solve_result, "status", None))
                if bool(getattr(solve_result, "ok", False)):
                    if str(status_value) == "partial_success":
                        state.problem_partial_count += 1
                    else:
                        state.problem_success_count += 1
                else:
                    state.problem_failure_count += 1
                    failure_writer.write(
                        _make_failure_record(
                            run_id=run_id,
                            record=record,
                            failure_code=str(getattr(getattr(solve_result, "failure", None), "code", "solve_failure")),
                            failure_message=str(getattr(getattr(solve_result, "failure", None), "message", "solve failure")),
                            elapsed_s=elapsed_s,
                            stage_records=getattr(solve_result, "stage_records", None),
                        )
                    )
                    state.failure_count += 1
                    if not args.continue_on_problem_failure:
                        break
            except Exception as exc:  # pragma: no cover
                elapsed_s = time.perf_counter() - solve_started
                tracker = _tracker_with_spent(tracker, elapsed_s)
                failure_writer.write(
                    _make_failure_record(
                        run_id=run_id,
                        record=record,
                        failure_code="uncaught_problem_exception",
                        failure_message=str(exc),
                        elapsed_s=elapsed_s,
                        exc=exc,
                    )
                )
                state.failure_count += 1
                state.problem_failure_count += 1
                if not args.continue_on_problem_failure:
                    break

    distillation_payload = None
    if args.emit_distilled_artifact and state.raw_trace_count > 0:
        distillation_payload = _run_optional_distillation(state)

    summary = _finalize_summary(
        state,
        run_id=run_id,
        started_at=started_at,
        records=records,
        tracker=tracker,
        engine_diagnostics=engine_diagnostics,
        distillation=distillation_payload,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if state.problem_failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

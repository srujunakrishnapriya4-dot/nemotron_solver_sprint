from __future__ import annotations

"""Production CLI for deterministic retrieval index construction.

This script builds the retrieval artifact consumed by the runtime trace retriever.
It intentionally does more than a notebook helper:
- parses a real CLI
- validates source inputs and config
- loads supported offline/retrieval source artifacts
- resolves optional problem-text lookup records
- builds a deterministic dense + lexical retrieval index
- writes an explicit build manifest with provenance
- enforces a no-overwrite default with actionable failures
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from pydantic import BaseModel, Field, ValidationError, model_validator

# Ensure the repository root is importable when the script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import OFFLINE_ARTIFACT_VERSION
from src.common.schemas import BranchTrace, ParsedProblem
from src.offline.trace_distillation import DistilledTraceRecord, load_distilled_artifact
from src.retrieval.embedder import MathEmbedder
from src.retrieval.index_builder import IndexedTraceRecord, RetrievalIndexBuilder


SCRIPT_VERSION = "build_indexes.v1"
DEFAULT_OUTPUT_ROOT = "artifacts/retrieval"
DEFAULT_INDEX_PREFIX = "retrieval_index"
DEFAULT_STRUCTURAL_FEATURES = ("dense_embeddings", "lexical_tokens", "tags", "operators", "snippets")
SUPPORTED_INPUT_KINDS = {
    "auto",
    "raw_trace_artifact",
    "raw_trace_jsonl",
    "branch_trace_jsonl",
    "distilled_artifact",
    "distilled_jsonl",
    "indexed_records_jsonl",
    "generic_jsonl",
    "generic_json",
}


class BuildIndexesError(RuntimeError):
    """Raised for actionable CLI/build failures."""


class InputSourceSpec(BaseModel):
    path: str
    kind: str = "auto"
    label: str | None = None

    @model_validator(mode="after")
    def _validate_kind(self) -> "InputSourceSpec":
        if self.kind not in SUPPORTED_INPUT_KINDS:
            raise ValueError(
                f"Unsupported input kind '{self.kind}'. Supported: {sorted(SUPPORTED_INPUT_KINDS)}"
            )
        return self


class BuildIndexesConfig(BaseModel):
    inputs: list[InputSourceSpec] = Field(default_factory=list)
    problem_lookup: str | None = None
    output_root: str = DEFAULT_OUTPUT_ROOT
    index_prefix: str = DEFAULT_INDEX_PREFIX
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_device: str = "cpu"
    batch_size: int = Field(default=64, ge=1)
    fallback_dimension: int = Field(default=384, ge=8)
    overwrite: bool = False
    strict_problem_text: bool = False
    dry_run: bool = False

    @model_validator(mode="after")
    def _validate_inputs(self) -> "BuildIndexesConfig":
        if not self.inputs:
            raise ValueError("At least one --input source must be provided.")
        return self


class SourceLoadResult(BaseModel):
    spec: InputSourceSpec
    kind_resolved: str
    records: list[IndexedTraceRecord]
    missing_problem_text_count: int = 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(payload: Any) -> str:
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:16]


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BuildIndexesError(f"Config file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(raw) if raw.strip() else {}
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise BuildIndexesError(
                    f"YAML config requested but PyYAML is unavailable: {path}"
                )
            data = yaml.safe_load(raw) or {}
        else:
            raise BuildIndexesError(
                f"Unsupported config format '{path.suffix}'. Use .json/.yaml/.yml"
            )
    except BuildIndexesError:
        raise
    except Exception as exc:
        raise BuildIndexesError(f"Failed to parse config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BuildIndexesError(f"Config file must contain a mapping/object: {path}")
    return data


def _resolve_config(args: argparse.Namespace) -> BuildIndexesConfig:
    payload: dict[str, Any] = {}
    if args.config is not None:
        payload.update(_read_config_file(Path(args.config)))

    cli_inputs: list[dict[str, Any]] = []
    if args.input:
        kinds = list(args.input_kind or [])
        while len(kinds) < len(args.input):
            kinds.append("auto")
        labels = list(args.input_label or [])
        while len(labels) < len(args.input):
            labels.append(None)
        cli_inputs = [
            {"path": p, "kind": kinds[idx], "label": labels[idx]}
            for idx, p in enumerate(args.input)
        ]

    overrides = {
        "inputs": cli_inputs or None,
        "problem_lookup": args.problem_lookup,
        "output_root": args.output_dir,
        "index_prefix": args.index_prefix,
        "artifact_version": args.artifact_version,
        "embedder_model": args.embedder_model,
        "embedder_device": args.device,
        "batch_size": args.batch_size,
        "fallback_dimension": args.fallback_dimension,
        "overwrite": True if args.overwrite else None,
        "strict_problem_text": True if args.strict_problem_text else None,
        "dry_run": True if args.dry_run else None,
    }
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value

    try:
        return BuildIndexesConfig.model_validate(payload)
    except ValidationError as exc:
        raise BuildIndexesError(f"Invalid build_indexes configuration:\n{exc}") from exc


def _load_json_or_jsonl(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise BuildIndexesError(
                        f"Invalid JSONL in {path} at line {line_no}: {exc}"
                    ) from exc
        return rows

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BuildIndexesError(f"Invalid JSON file {path}: {exc}") from exc
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("records", "items", "rows", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        raise BuildIndexesError(f"Unsupported JSON payload in {path}: expected object or list")

    raise BuildIndexesError(f"Unsupported source file extension for {path}; use .json or .jsonl")


def _load_problem_lookup(path: str | None) -> dict[str, ParsedProblem]:
    if not path:
        return {}
    src = (REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not src.exists():
        raise BuildIndexesError(f"Problem lookup file not found: {src}")
    rows = _load_json_or_jsonl(src)
    lookup: dict[str, ParsedProblem] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BuildIndexesError(f"Problem lookup row {idx} in {src} is not an object")
        try:
            problem = ParsedProblem.model_validate(dict(row))
        except ValidationError as exc:
            raise BuildIndexesError(
                f"Problem lookup row {idx} in {src} is not a valid ParsedProblem:\n{exc}"
            ) from exc
        lookup[problem.problem_id] = problem
    if not lookup:
        raise BuildIndexesError(f"Problem lookup file is empty: {src}")
    return lookup


def _join_step_texts(steps: Sequence[Any]) -> str:
    parts: list[str] = []
    for step in steps:
        text = None
        if isinstance(step, Mapping):
            text = step.get("text") or step.get("description") or step.get("summary")
        else:
            text = getattr(step, "text", None) or getattr(step, "description", None)
        clean = " ".join(str(text or "").split()).strip()
        if clean:
            parts.append(clean)
    return "\n".join(parts)


def _looks_like_generated_raw_trace(payload: Mapping[str, Any]) -> bool:
    record_type = str(payload.get("record_type") or "").strip().lower()
    if record_type == "raw_branch_trace":
        return True
    return (
        "problem_id" in payload
        and "problem_text" in payload
        and isinstance(payload.get("steps"), list)
        and ("route_snapshot" in payload or "operator_sequence" in payload)
    )


def _generated_raw_trace_rows(path: Path) -> list[Any]:
    if path.is_dir():
        raw_path = path / "raw_traces.jsonl"
    elif path.name == "manifest.json":
        raw_path = path.with_name("raw_traces.jsonl")
    else:
        raise BuildIndexesError(f"Unsupported raw trace artifact path: {path}")
    if not raw_path.exists():
        raise BuildIndexesError(f"Raw trace artifact missing raw_traces.jsonl: {path}")
    return _load_json_or_jsonl(raw_path)


def _operator_names_from_steps(steps: Sequence[Any]) -> list[str]:
    operators: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        name = str(step.get("operator_name") or step.get("operator_used") or "").strip()
        if name:
            operators.append(name)
    return operators


def _derive_problem_text(problem_id: str, problem_lookup: Mapping[str, ParsedProblem], *candidates: Any) -> str:
    from_lookup = problem_lookup.get(problem_id)
    if from_lookup is not None and from_lookup.raw_text.strip():
        return from_lookup.raw_text
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, Mapping):
            for key in (
                "problem_text",
                "raw_text",
                "problem",
                "prompt",
                "source_problem_text",
            ):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def _to_indexed_from_branch_trace(trace: BranchTrace, problem_lookup: Mapping[str, ParsedProblem]) -> IndexedTraceRecord:
    problem = problem_lookup.get(trace.problem_id)
    problem_text = _derive_problem_text(trace.problem_id, problem_lookup)
    tags: set[str] = set()
    if problem is not None:
        tags.add(problem.domain.value)
        tags.update(problem.likely_archetypes[:4])
        if problem.target:
            tags.add(problem.target.lower().replace(" ", "_"))
        if problem.parity_cues:
            tags.add("parity")
        if problem.symmetries:
            tags.add("symmetry")
    if trace.failure_type is not None:
        tags.add(f"failure::{trace.failure_type.value}")
    if trace.symbolic_valid:
        tags.add("symbolic_valid")
    if trace.repaired:
        tags.add("repaired")

    snippets: list[str] = []
    if problem is not None:
        snippets.extend(problem.constraints[:4])
        snippets.extend(problem.unknowns[:2])
        snippets.extend(problem.knowns[:2])
    for step in trace.steps[:4]:
        if step.description:
            snippets.append(step.description[:180])

    repair_snippets: list[str] = []
    if trace.failure_location:
        repair_snippets.append(trace.failure_location)
    if trace.failure_type is not None:
        repair_snippets.append(trace.failure_type.value)
    if trace.repaired:
        repair_snippets.append("repaired_branch")

    return IndexedTraceRecord(
        trace_id=trace.branch_id,
        problem_id=trace.problem_id,
        problem_text=problem_text,
        solution_text=trace.full_reasoning or _join_step_texts(trace.steps),
        answer=str(trace.answer or ""),
        domain=problem.domain.value if problem is not None else "mixed",
        difficulty="medium",
        archetypes=[trace.archetype_used] if trace.archetype_used else [],
        operators_used=list(trace.operator_sequence),
        source="branch_trace",
        tags=sorted(tags),
        subproblem_snippets=[s for s in snippets if s][:8],
        repair_snippets=repair_snippets[:4],
        metadata={
            "branch_score": trace.branch_score,
            "verifier_score": trace.verifier_score,
            "symbolic_valid": trace.symbolic_valid,
            "repaired": trace.repaired,
            "tool_consistency": trace.tool_consistency,
            "generation_time_sec": trace.generation_time_sec,
        },
    )


def _to_indexed_from_distilled(record: DistilledTraceRecord, problem_lookup: Mapping[str, ParsedProblem]) -> IndexedTraceRecord:
    problem = problem_lookup.get(record.problem_id)
    problem_text = _derive_problem_text(
        record.problem_id,
        problem_lookup,
        record.metadata,
        record.provenance.metadata,
    )
    solution_text = _join_step_texts(record.steps)
    evidence_tags = [f"evidence::{ev.kind}" for ev in record.evidence[:4] if getattr(ev, "kind", None)]
    tags = sorted(
        {
            str(record.classification.value),
            str(record.quality_tier.value),
            str(record.split.value),
            *(record.archetypes or []),
            *evidence_tags,
            *( [record.outcome.failure_type] if record.outcome.failure_type else [] ),
        }
    )
    domain = problem.domain.value if problem is not None else str(record.metadata.get("domain") or "mixed")
    difficulty = str(record.metadata.get("difficulty") or "medium")
    snippets = [step.text[:180] for step in record.steps[:6] if step.text]
    repair_snippets = [
        s
        for s in [record.outcome.failure_location, record.outcome.failure_type, "repaired" if record.outcome.repaired else None]
        if s
    ][:4]
    return IndexedTraceRecord(
        trace_id=record.trace_id,
        problem_id=record.problem_id,
        problem_text=problem_text,
        solution_text=solution_text,
        answer=str(record.outcome.answer_canonical or record.outcome.answer_raw or ""),
        domain=domain,
        difficulty=difficulty,
        archetypes=list(record.archetypes),
        operators_used=list(record.operator_sequence),
        source="distilled_trace",
        tags=tags,
        subproblem_snippets=snippets,
        repair_snippets=repair_snippets,
        metadata={
            "record_id": record.record_id,
            "classification": record.classification.value,
            "quality_tier": record.quality_tier.value,
            "split": record.split.value,
            "branch_id": record.branch_id,
            "verifier_score": record.outcome.verifier_score,
            "symbolic_valid": record.outcome.symbolic_valid,
            "branch_score": record.outcome.branch_score,
            "repair_count": record.outcome.repair_count,
            "retrieval_used": record.outcome.retrieval_used,
        },
    )


def _to_indexed_from_generated_raw_trace(
    payload: Mapping[str, Any],
    problem_lookup: Mapping[str, ParsedProblem],
) -> IndexedTraceRecord:
    problem_id = str(payload.get("problem_id") or "").strip()
    if not problem_id:
        raise BuildIndexesError("Generated raw trace record is missing problem_id")

    problem = problem_lookup.get(problem_id)
    route_snapshot = payload.get("route_snapshot") if isinstance(payload.get("route_snapshot"), Mapping) else {}
    steps = [item for item in (payload.get("steps") or []) if isinstance(item, Mapping)]
    operator_sequence = [str(item).strip() for item in payload.get("operator_sequence", []) if str(item).strip()]
    if not operator_sequence:
        operator_sequence = _operator_names_from_steps(steps)

    problem_text = _derive_problem_text(problem_id, problem_lookup, payload, route_snapshot)
    solution_text = _join_step_texts(steps)
    if not solution_text:
        solution_text = " ".join(
            str(payload.get(key) or "").strip()
            for key in ("reasoning_text", "full_reasoning")
            if str(payload.get(key) or "").strip()
        )

    raw_archetypes = payload.get("archetypes")
    archetypes: list[str] = []
    if isinstance(raw_archetypes, Sequence) and not isinstance(raw_archetypes, (str, bytes)):
        archetypes = sorted({str(item).strip() for item in raw_archetypes if str(item).strip()})
    elif isinstance(route_snapshot.get("archetypes"), Mapping):
        archetypes = sorted(
            str(key).strip()
            for key, value in route_snapshot["archetypes"].items()
            if str(key).strip() and float(value or 0.0) > 0.0
        )

    tags: set[str] = {"raw_trace"}
    if problem is not None:
        tags.add(problem.domain.value)
    domain = (
        problem.domain.value
        if problem is not None
        else str(route_snapshot.get("domain") or "mixed").strip() or "mixed"
    )
    difficulty = str(route_snapshot.get("difficulty") or "medium").strip() or "medium"
    tags.update(archetypes[:4])
    if bool(payload.get("symbolic_valid", False)):
        tags.add("symbolic_valid")
    if bool(payload.get("repaired", False)):
        tags.add("repaired")
    failure_type = str(payload.get("failure_type") or "").strip()
    if failure_type:
        tags.add(f"failure::{failure_type}")

    repair_snippets = [
        snippet
        for snippet in (
            str(payload.get("failure_location") or "").strip(),
            failure_type,
            "repaired_branch" if bool(payload.get("repaired", False)) else "",
        )
        if snippet
    ][:4]

    return IndexedTraceRecord(
        trace_id=str(payload.get("trace_id") or payload.get("branch_id") or problem_id),
        problem_id=problem_id,
        problem_text=problem_text,
        solution_text=solution_text,
        answer=str(payload.get("answer_canonical") or payload.get("answer_raw") or payload.get("answer") or ""),
        domain=domain,
        difficulty=difficulty,
        archetypes=archetypes,
        operators_used=operator_sequence,
        source="raw_trace",
        tags=sorted(tags),
        subproblem_snippets=[
            snippet
            for snippet in (
                problem_text,
                *[str(step.get("description") or "")[:180] for step in steps[:6]],
            )
            if snippet
        ][:8],
        repair_snippets=repair_snippets,
        metadata={
            "branch_id": str(payload.get("branch_id") or ""),
            "record_type": str(payload.get("record_type") or ""),
            "run_id": str(payload.get("run_id") or ""),
            "verifier_score": payload.get("verifier_score"),
            "symbolic_valid": bool(payload.get("symbolic_valid", False)),
            "branch_score": payload.get("branch_score"),
            "repair_count": payload.get("repair_count"),
            "retrieval_used": bool(payload.get("retrieval_used", False)),
            "solve_status": payload.get("solve_status"),
        },
    )


def _coerce_mapping_to_record(payload: Mapping[str, Any], problem_lookup: Mapping[str, ParsedProblem]) -> IndexedTraceRecord:
    if {"trace_id", "problem_id", "problem_text", "solution_text"}.issubset(payload.keys()):
        return IndexedTraceRecord(
            trace_id=str(payload["trace_id"]),
            problem_id=str(payload["problem_id"]),
            problem_text=str(payload.get("problem_text") or ""),
            solution_text=str(payload.get("solution_text") or ""),
            answer=str(payload.get("answer") or ""),
            domain=str(payload.get("domain") or "mixed"),
            difficulty=str(payload.get("difficulty") or "medium"),
            archetypes=[str(x) for x in payload.get("archetypes", [])],
            operators_used=[str(x) for x in payload.get("operators_used", [])],
            source=str(payload.get("source") or "generic"),
            tags=[str(x) for x in payload.get("tags", [])],
            subproblem_snippets=[str(x) for x in payload.get("subproblem_snippets", [])],
            repair_snippets=[str(x) for x in payload.get("repair_snippets", [])],
            metadata={k: v for k, v in payload.items() if k not in {
                "trace_id", "problem_id", "problem_text", "solution_text", "answer",
                "domain", "difficulty", "archetypes", "operators_used", "source",
                "tags", "subproblem_snippets", "repair_snippets",
            }},
        )
    if _looks_like_generated_raw_trace(payload):
        return _to_indexed_from_generated_raw_trace(payload, problem_lookup)
    if "full_reasoning" in payload and "branch_id" in payload and "problem_id" in payload:
        try:
            return _to_indexed_from_branch_trace(BranchTrace.model_validate(dict(payload)), problem_lookup)
        except ValidationError as exc:
            raise BuildIndexesError(f"Invalid BranchTrace payload: {exc}") from exc
    if "outcome" in payload and "provenance" in payload and "record_id" in payload:
        try:
            return _to_indexed_from_distilled(DistilledTraceRecord.model_validate(dict(payload)), problem_lookup)
        except ValidationError as exc:
            raise BuildIndexesError(f"Invalid DistilledTraceRecord payload: {exc}") from exc
    raise BuildIndexesError(
        "Unsupported record payload. Expected IndexedTraceRecord-like fields, BranchTrace fields, or DistilledTraceRecord fields."
    )


def _detect_input_kind(path: Path) -> str:
    if path.is_dir():
        if (path / "manifest.json").exists() and (path / "records.jsonl").exists():
            return "distilled_artifact"
        if (path / "manifest.json").exists() and (path / "raw_traces.jsonl").exists():
            return "raw_trace_artifact"
        raise BuildIndexesError(
            f"Unsupported directory source {path}. Expected a distilled artifact directory or a generate_traces run directory with raw_traces.jsonl"
        )
    if not path.exists():
        raise BuildIndexesError(f"Input source not found: {path}")
    if path.name == "manifest.json" and path.parent.is_dir() and (path.parent / "records.jsonl").exists():
        return "distilled_artifact"
    if path.name == "manifest.json" and path.parent.is_dir() and (path.parent / "raw_traces.jsonl").exists():
        return "raw_trace_artifact"
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise BuildIndexesError(
            f"Unsupported input file type for {path}. Supported: .json, .jsonl, or distilled artifact directory"
        )
    rows = _load_json_or_jsonl(path)
    if not rows:
        raise BuildIndexesError(f"Input source is empty: {path}")
    first = rows[0]
    if not isinstance(first, Mapping):
        raise BuildIndexesError(f"Input source must contain JSON objects: {path}")
    keys = set(first.keys())
    if {"trace_id", "problem_id", "problem_text", "solution_text"}.issubset(keys):
        return "indexed_records_jsonl"
    if _looks_like_generated_raw_trace(first):
        return "raw_trace_jsonl"
    if {"branch_id", "problem_id", "full_reasoning"}.issubset(keys):
        return "branch_trace_jsonl"
    if {"record_id", "provenance", "outcome"}.issubset(keys):
        return "distilled_jsonl"
    return "generic_jsonl" if path.suffix.lower() == ".jsonl" else "generic_json"


def _load_source(spec: InputSourceSpec, problem_lookup: Mapping[str, ParsedProblem]) -> SourceLoadResult:
    raw_path = Path(spec.path)
    path = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path)
    path = path.resolve()
    kind = spec.kind if spec.kind != "auto" else _detect_input_kind(path)

    records: list[IndexedTraceRecord] = []
    if kind == "distilled_artifact":
        artifact = load_distilled_artifact(path)
        records = [_to_indexed_from_distilled(record, problem_lookup) for record in artifact.records]
    elif kind == "raw_trace_artifact":
        rows = _generated_raw_trace_rows(path)
        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise BuildIndexesError(f"Row {idx} in raw trace artifact {path} is not a JSON object")
            records.append(_to_indexed_from_generated_raw_trace(row, problem_lookup))
    else:
        rows = _load_json_or_jsonl(path)
        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise BuildIndexesError(f"Row {idx} in {path} is not a JSON object")
            if kind == "branch_trace_jsonl":
                try:
                    branch_trace = BranchTrace.model_validate(dict(row))
                except ValidationError as exc:
                    raise BuildIndexesError(
                        f"Row {idx} in {path} is not a valid BranchTrace:\n{exc}"
                    ) from exc
                records.append(_to_indexed_from_branch_trace(branch_trace, problem_lookup))
            elif kind == "distilled_jsonl":
                try:
                    distilled = DistilledTraceRecord.model_validate(dict(row))
                except ValidationError as exc:
                    raise BuildIndexesError(
                        f"Row {idx} in {path} is not a valid DistilledTraceRecord:\n{exc}"
                    ) from exc
                records.append(_to_indexed_from_distilled(distilled, problem_lookup))
            elif kind == "raw_trace_jsonl":
                records.append(_to_indexed_from_generated_raw_trace(row, problem_lookup))
            else:
                records.append(_coerce_mapping_to_record(row, problem_lookup))

    if not records:
        raise BuildIndexesError(f"No retrieval records loaded from {path}")

    missing_problem_text_count = sum(1 for record in records if not record.problem_text.strip())
    return SourceLoadResult(
        spec=InputSourceSpec(path=str(path), kind=spec.kind, label=spec.label),
        kind_resolved=kind,
        records=records,
        missing_problem_text_count=missing_problem_text_count,
    )


def _ensure_clean_output_dir(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise BuildIndexesError(
            f"Output artifact directory already exists: {path}. Re-run with --overwrite to replace it."
        )
    shutil.rmtree(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_manifest(
    *,
    config: BuildIndexesConfig,
    artifact_id: str,
    artifact_dir: Path,
    source_results: Sequence[SourceLoadResult],
    builder: RetrievalIndexBuilder,
    build_started_at: str,
    build_finished_at: str,
) -> dict[str, Any]:
    artifact_meta = builder.artifact_metadata
    index_dir = artifact_dir / "index"
    output_paths = {
        "artifact_dir": str(artifact_dir),
        "index_dir": str(index_dir),
        "records_pkl": str(index_dir / "records.pkl"),
        "token_sets_pkl": str(index_dir / "token_sets.pkl"),
        "dense_npy": str(index_dir / "dense.npy"),
        "artifact_meta_json": str(index_dir / "artifact_meta.json"),
        "manifest_json": str(artifact_dir / "manifest.json"),
    }
    faiss_path = index_dir / "index.faiss"
    if faiss_path.exists():
        output_paths["faiss_index"] = str(faiss_path)

    return {
        "artifact_kind": "retrieval_index_build",
        "script_version": SCRIPT_VERSION,
        "artifact_version": config.artifact_version,
        "artifact_id": artifact_id,
        "created_at_utc": build_finished_at,
        "build_started_at_utc": build_started_at,
        "build_finished_at_utc": build_finished_at,
        "config": {
            "embedder_model": config.embedder_model,
            "embedder_device": config.embedder_device,
            "batch_size": config.batch_size,
            "fallback_dimension": config.fallback_dimension,
            "strict_problem_text": config.strict_problem_text,
            "overwrite": config.overwrite,
            "problem_lookup": config.problem_lookup,
        },
        "input_sources": [
            {
                "path": result.spec.path,
                "label": result.spec.label,
                "declared_kind": result.spec.kind,
                "resolved_kind": result.kind_resolved,
                "record_count": len(result.records),
                "missing_problem_text_count": result.missing_problem_text_count,
            }
            for result in source_results
        ],
        "record_count": len(builder.records),
        "problem_count": len({record.problem_id for record in builder.records}),
        "embedding_mode": {
            "backend": builder.embedder.backend,
            "model_name": config.embedder_model,
            "device": config.embedder_device,
            "dimension": artifact_meta.embedding_dimension if artifact_meta else config.fallback_dimension,
        },
        "structural_features_enabled": list(DEFAULT_STRUCTURAL_FEATURES),
        "index_metadata": asdict(artifact_meta) if artifact_meta is not None else {},
        "output_paths": output_paths,
        "provenance": {
            "repo_root": str(REPO_ROOT),
            "cwd": os.getcwd(),
            "python": sys.version,
        },
    }


def build_index(config: BuildIndexesConfig) -> tuple[Path | None, dict[str, Any]]:
    logging.info("Resolving problem lookup and retrieval sources")
    problem_lookup = _load_problem_lookup(config.problem_lookup)
    source_results = [_load_source(spec, problem_lookup) for spec in config.inputs]

    all_records: list[IndexedTraceRecord] = []
    missing_problem_text_count = 0
    for result in source_results:
        all_records.extend(result.records)
        missing_problem_text_count += result.missing_problem_text_count

    if not all_records:
        raise BuildIndexesError("No retrieval records available after source loading.")

    if config.strict_problem_text and missing_problem_text_count > 0:
        raise BuildIndexesError(
            "Problem text is missing for one or more records. "
            f"Loaded {missing_problem_text_count} records without problem text; provide --problem-lookup or disable --strict-problem-text."
        )
    if missing_problem_text_count > 0:
        logging.warning(
            "Loaded %d records without problem text. Retrieval will still build, but those rows will have weaker dense/lexical context.",
            missing_problem_text_count,
        )

    deterministic_payload = {
        "artifact_version": config.artifact_version,
        "index_prefix": config.index_prefix,
        "embedder_model": config.embedder_model,
        "embedder_device": config.embedder_device,
        "fallback_dimension": config.fallback_dimension,
        "batch_size": config.batch_size,
        "problem_lookup": str(config.problem_lookup or ""),
        "inputs": [
            {
                "path": result.spec.path,
                "label": result.spec.label,
                "resolved_kind": result.kind_resolved,
                "record_count": len(result.records),
            }
            for result in source_results
        ],
        "record_identity": [
            {
                "trace_id": record.trace_id,
                "problem_id": record.problem_id,
                "source": record.source,
            }
            for record in sorted(all_records, key=lambda r: (r.problem_id, r.trace_id, r.source))
        ],
    }
    artifact_id = f"{config.index_prefix}_{_fingerprint(deterministic_payload)}"
    output_root = (REPO_ROOT / config.output_root).resolve() if not Path(config.output_root).is_absolute() else Path(config.output_root)
    artifact_dir = output_root / artifact_id

    logging.info("Prepared %d records across %d sources", len(all_records), len(source_results))
    if config.dry_run:
        return None, {
            "artifact_id": artifact_id,
            "artifact_dir": str(artifact_dir),
            "record_count": len(all_records),
            "problem_count": len({record.problem_id for record in all_records}),
            "source_count": len(source_results),
            "missing_problem_text_count": missing_problem_text_count,
            "mode": "dry_run",
        }

    _ensure_clean_output_dir(artifact_dir, overwrite=config.overwrite)
    output_root.mkdir(parents=True, exist_ok=True)

    build_started_at = _utc_now_iso()
    temp_dir = output_root / f".tmp_{artifact_id}_{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        embedder = MathEmbedder(
            model_name=config.embedder_model,
            device=config.embedder_device,
            batch_size=config.batch_size,
            fallback_dimension=config.fallback_dimension,
        )
        builder = RetrievalIndexBuilder(embedder=embedder, index_path=str(temp_dir / "index"))
        builder.build_from_records(all_records, save=True)
        build_finished_at = _utc_now_iso()
        manifest = _build_manifest(
            config=config,
            artifact_id=artifact_id,
            artifact_dir=temp_dir,
            source_results=source_results,
            builder=builder,
            build_started_at=build_started_at,
            build_finished_at=build_finished_at,
        )
        _write_json(temp_dir / "manifest.json", manifest)
        os.replace(temp_dir, artifact_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    summary = {
        "artifact_id": artifact_id,
        "artifact_dir": str(artifact_dir),
        "record_count": len(all_records),
        "problem_count": len({record.problem_id for record in all_records}),
        "source_count": len(source_results),
        "missing_problem_text_count": missing_problem_text_count,
        "embedder_backend": embedder.backend,
        "embedder_model": config.embedder_model,
    }
    return artifact_dir, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the production retrieval index artifact used by runtime retrieval.",
    )
    parser.add_argument("--config", type=str, help="Optional JSON/YAML config file.")
    parser.add_argument(
        "--input",
        action="append",
        help="Input source path. Repeat for multiple inputs. Supported: distilled artifact dir/manifest, JSONL, JSON.",
    )
    parser.add_argument(
        "--input-kind",
        action="append",
        help="Input kind for the corresponding --input. Default: auto.",
        choices=sorted(SUPPORTED_INPUT_KINDS),
    )
    parser.add_argument(
        "--input-label",
        action="append",
        help="Optional human-readable label for the corresponding --input.",
    )
    parser.add_argument(
        "--problem-lookup",
        type=str,
        help="Optional ParsedProblem JSON/JSONL file used to recover problem text and domains.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}")
    parser.add_argument("--index-prefix", type=str, default=None, help=f"Artifact directory prefix. Default: {DEFAULT_INDEX_PREFIX}")
    parser.add_argument("--artifact-version", type=str, default=None, help=f"Version tag to record in manifest. Default: {OFFLINE_ARTIFACT_VERSION}")
    parser.add_argument("--embedder-model", type=str, default=None, help="Embedding model name.")
    parser.add_argument("--device", type=str, default=None, help="Embedder device, e.g. cpu/cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size.")
    parser.add_argument("--fallback-dimension", type=int, default=None, help="Deterministic fallback embedding dimension.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing deterministic artifact directory if present.")
    parser.add_argument("--strict-problem-text", action="store_true", help="Fail if any record lacks problem text after source resolution.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and sources without writing artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = _resolve_config(args)
        artifact_dir, summary = build_index(config)
        if config.dry_run:
            logging.info(
                "Dry run succeeded: artifact_id=%s records=%d problems=%d output_dir=%s",
                summary["artifact_id"],
                summary["record_count"],
                summary["problem_count"],
                summary["artifact_dir"],
            )
        else:
            logging.info(
                "Retrieval index build complete: artifact_id=%s records=%d problems=%d sources=%d dir=%s",
                summary["artifact_id"],
                summary["record_count"],
                summary["problem_count"],
                summary["source_count"],
                summary["artifact_dir"],
            )
            logging.info("Manifest: %s", Path(summary["artifact_dir"]) / "manifest.json")
        return 0
    except BuildIndexesError as exc:
        logging.error(str(exc))
        return 2
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logging.exception("Unexpected failure while building retrieval index: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

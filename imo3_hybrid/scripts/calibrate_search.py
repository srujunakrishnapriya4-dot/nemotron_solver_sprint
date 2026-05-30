from __future__ import annotations

"""Production CLI entrypoint for search/verifier/aggregation calibration.

Merged implementation:
- keeps the older manifest-backed, selector-aware, richer calibration workflow
- adds newer repair-pass signal-surface expectations so decomposed runtime
  evidence is used explicitly for scoring and later robustness analysis
"""

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
from collections import defaultdict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.aggregation.canonicalize import canonicalize_answer
from src.aggregation.entropy_weighting import EntropyWeightingConfig, score_answer_clusters
from src.aggregation.final_selector import FinalSelectorConfig, select_final_answer
from src.common.constants import ANSWER_MAX, ANSWER_MIN, OFFLINE_ARTIFACT_VERSION
from src.common.schemas import BranchTrace, CandidateAnswer, ParsedProblem, ReasoningStep

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

LOGGER = logging.getLogger("calibrate_search")
SCRIPT_SCHEMA_VERSION = "calibrate_search_script.v2"
ARTIFACT_KIND = "search_calibration_run"
DEFAULT_OUTPUT_DIR = "artifacts/calibration/search"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class RunStatus(str, Enum):
    COMPLETED = "completed"
    READY = "ready"
    INVALID_DATASET = "invalid_dataset"
    MISSING_INPUT = "missing_input"


class BranchSelectionMode(str, Enum):
    ANY = "any"
    REQUIRE_SYMBOLIC = "require_symbolic"
    PREFER_SYMBOLIC = "prefer_symbolic"


class CalibrationScriptConfig(StrictModel):
    run_name: str = "search_calibration"
    input_paths: list[str] = Field(default_factory=list)
    output_dir: str = DEFAULT_OUTPUT_DIR
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    seed: int = 1729
    overwrite: bool = False
    max_problems: int | None = Field(default=None, ge=1)
    verifier_thresholds: list[float] = Field(default_factory=lambda: [0.45, 0.55, 0.65])
    branch_budgets: list[int] = Field(default_factory=lambda: [16, 32, 64])
    entropy_betas: list[float] = Field(default_factory=lambda: [0.18, 0.24, 0.30])
    min_branch_scores: list[float] = Field(default_factory=lambda: [0.0, 0.20])
    selection_modes: list[BranchSelectionMode] = Field(default_factory=lambda: [BranchSelectionMode.ANY, BranchSelectionMode.PREFER_SYMBOLIC])
    retrieval_compatibility_floors: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4])
    repairability_biases: list[float] = Field(default_factory=lambda: [0.0, 0.1])
    operator_reliability_floors: list[float] = Field(default_factory=lambda: [0.0, 0.2])
    proof_burden_penalties: list[float] = Field(default_factory=lambda: [0.0, 0.05])
    top_n_configs: int = Field(default=5, ge=1)
    write_csv: bool = True

    @model_validator(mode="after")
    def _validate_lists(self) -> "CalibrationScriptConfig":
        if not self.input_paths:
            raise ValueError("At least one input path is required")
        if not self.verifier_thresholds or not self.branch_budgets or not self.entropy_betas or not self.min_branch_scores:
            raise ValueError("Calibration grid lists cannot be empty")
        return self


class CalibrationBranchRecord(StrictModel):
    branch_id: str
    answer: str = ""
    verifier_score: float = 0.0
    tool_consistency: float = 0.0
    branch_score: float = 0.0
    symbolic_valid: bool = False
    full_reasoning: str = ""
    repaired: bool = False
    self_critiqued: bool = False
    repair_count: int = 0
    retrieval_used: bool = False
    operator_sequence: list[str] = Field(default_factory=list)
    archetype_used: str | None = None
    generation_time_sec: float = 0.0
    logical_consistency: float = 0.0
    completeness: float = 0.0
    repairability: float = 0.0
    answer_correctness_likelihood: float = 0.0
    symbolic_agreement: float = 0.0
    step_quality: float = 0.0
    prefix_quality: float = 0.0
    prm_prefix_quality: float = 0.0
    prm_step_quality: float = 0.0
    retrieval_compatibility: float = 0.0
    retrieval_support: float = 0.0
    operator_reliability: float = 0.0
    discharge_fraction: float = 0.0
    open_obligation_burden: float = 0.0
    verifier_decomposition: dict[str, float] = Field(default_factory=dict)
    route_compute_signals: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationExample(StrictModel):
    problem_id: str
    raw_text: str = ""
    gold_answer: str = ""
    answer_type: str = "non_negative_integer"
    domain: str = "mixed"
    branches: list[CalibrationBranchRecord] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def split_name(self) -> str:
        raw = self.source_metadata.get("split") or self.source_metadata.get("source_split") or self.source_metadata.get("calibration_split") or "unspecified"
        return str(raw)


class InputValidationReport(StrictModel):
    valid: bool
    num_examples: int = 0
    num_problem_ids: int = 0
    num_branches: int = 0
    duplicate_problem_ids: list[str] = Field(default_factory=list)
    missing_gold_problem_ids: list[str] = Field(default_factory=list)
    empty_branch_problem_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CalibrationParameterSet(StrictModel):
    verifier_threshold: float
    branch_budget: int
    entropy_beta: float
    min_branch_score: float
    selection_mode: BranchSelectionMode
    retrieval_compatibility_floor: float = 0.0
    repairability_bias: float = 0.0
    operator_reliability_floor: float = 0.0
    proof_burden_penalty: float = 0.0

    def stable_id(self) -> str:
        return _stable_hash("params", self.model_dump(mode="json"))


class PerProblemResult(StrictModel):
    problem_id: str
    gold_answer: str
    predicted_answer: str
    correct: bool
    candidate_count: int
    filtered_branch_count: int
    total_branch_count: int
    winner_confidence: float
    method_used: str
    selection_warning_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationScoreSummary(StrictModel):
    parameter_id: str
    parameters: dict[str, Any]
    num_examples: int
    num_correct: int
    accuracy: float
    mean_confidence: float = 0.0
    mean_filtered_branch_count: float = 0.0
    mean_candidate_count: float = 0.0
    solved_nonzero_rate: float = 0.0
    score: float = 0.0
    split_summaries: dict[str, Any] = Field(default_factory=dict)
    robustness_penalty: float = 0.0
    signal_summary: dict[str, float] = Field(default_factory=dict)


class CalibrationRunManifest(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    schema_version: str = SCRIPT_SCHEMA_VERSION
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    run_name: str
    run_id: str
    status: str
    input_sources: list[dict[str, Any]] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    parameter_grid_size: int = 0
    best_parameter_id: str | None = None
    best_summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    split_summary: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedInputSource:
    requested_path: str
    resolved_path: str
    source_kind: str
    row_count: int


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(row) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass


def _normalize_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_rows_from_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("examples", "records", "items", "problems", "rows", "evaluations"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [payload]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix in {".parquet", ".pq"}:
        if pd is None:
            raise RuntimeError("pandas is unavailable; cannot read parquet input")
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported input file type: {path.suffix}")


def _resolve_input_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.is_dir():
        manifest = path / "manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for key in ("examples_path", "records_path", "evaluation_path", "jsonl_path"):
                candidate = payload.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    resolved = (path / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate)
                    return _load_rows_from_path(resolved), "manifest_dir"
        records = path / "records.jsonl"
        if records.exists():
            return _load_rows_from_path(records), "records_dir"
        raise FileNotFoundError(f"No supported manifest or records.jsonl found in directory: {path}")
    if path.name == "manifest.json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("examples_path", "records_path", "evaluation_path", "jsonl_path"):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                resolved = (path.parent / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate)
                return _load_rows_from_path(resolved), "manifest_file"
        raise ValueError(f"Manifest does not declare a supported records path: {path}")
    return _load_rows_from_path(path), path.suffix.lower().lstrip(".") or "file"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_steps(value: Any) -> list[ReasoningStep]:
    if not isinstance(value, list):
        return []
    steps: list[ReasoningStep] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            steps.append(
                ReasoningStep(
                    step_num=int(item.get("step_num", len(steps) + 1)),
                    description=str(item.get("description", item.get("text", ""))),
                    operator_used=str(item.get("operator_used")) if item.get("operator_used") is not None else None,
                    symbolic_expression=str(item.get("symbolic_expression")) if item.get("symbolic_expression") is not None else None,
                    symbolic_valid=_parse_bool(item.get("symbolic_valid", True)),
                    python_code=str(item.get("python_code")) if item.get("python_code") is not None else None,
                    python_result=str(item.get("python_result")) if item.get("python_result") is not None else None,
                )
            )
        except Exception:
            continue
    return steps


def _coerce_branch_trace(problem_id: str, row: Mapping[str, Any], fallback_index: int) -> BranchTrace:
    branch_id = str(row.get("branch_id") or row.get("id") or f"{problem_id}::branch::{fallback_index:04d}")
    answer = "" if row.get("answer") is None else str(row.get("answer"))
    reasoning = str(row.get("full_reasoning") or row.get("reasoning") or row.get("trace") or "")
    operator_sequence = row.get("operator_sequence")
    if not isinstance(operator_sequence, list):
        operator_sequence = []
    return BranchTrace(
        branch_id=branch_id,
        problem_id=problem_id,
        steps=_parse_steps(row.get("steps")),
        full_reasoning=reasoning,
        answer=answer or None,
        answer_canonical=None,
        symbolic_valid=_parse_bool(row.get("symbolic_valid", False)),
        branch_score=_parse_float(row.get("branch_score", row.get("score", 0.0))),
        verifier_score=_parse_float(row.get("verifier_score", 0.0)),
        tool_consistency=_parse_float(row.get("tool_consistency", 0.0)),
        logical_consistency=_parse_float(row.get("logical_consistency", 0.0)),
        completeness=_parse_float(row.get("completeness", 0.0)),
        repairability=_parse_float(row.get("repairability", 0.0)),
        answer_correctness_likelihood=_parse_float(row.get("answer_correctness_likelihood", 0.0)),
        symbolic_agreement=_parse_float(row.get("symbolic_agreement", 0.0)),
        step_quality=_parse_float(row.get("step_quality", 0.0)),
        prefix_quality=_parse_float(row.get("prefix_quality", 0.0)),
        prm_prefix_quality=_parse_float(row.get("prm_prefix_quality", row.get("prefix_quality", 0.0))),
        prm_step_quality=_parse_float(row.get("prm_step_quality", 0.0)),
        retrieval_compatibility=_parse_float(row.get("retrieval_compatibility", 0.0)),
        retrieval_support=_parse_float(row.get("retrieval_support", 0.0)),
        operator_reliability=_parse_float(row.get("operator_reliability", 0.0)),
        discharge_fraction=_parse_float(row.get("discharge_fraction", 0.0)),
        open_obligation_burden=_parse_float(row.get("open_obligation_burden", 0.0)),
        verifier_decomposition=dict(row.get("verifier_decomposition") or {}),
        self_critiqued=_parse_bool(row.get("self_critiqued", False)),
        repaired=_parse_bool(row.get("repaired", False)),
        repair_count=int(_parse_float(row.get("repair_count", 0), 0.0)),
        operator_sequence=[str(v) for v in operator_sequence],
        archetype_used=str(row.get("archetype_used")) if row.get("archetype_used") is not None else None,
        retrieval_used=_parse_bool(row.get("retrieval_used", False)),
        generation_time_sec=_parse_float(row.get("generation_time_sec", 0.0)),
        metadata={
            **dict(row.get("metadata") or {}),
            "route_compute_signals": dict(row.get("route_compute_signals") or row.get("compute_signals") or {}),
        },
    )


def _canonical_text(value: str, answer_type: str) -> str:
    canon = canonicalize_answer(value, answer_type=answer_type, allow_heuristic=True)
    key = canon.cluster_key(require_strong=False)
    return key or canon.canonical_text or canon.normalized_text or str(value).strip()


def _group_candidate_answers(branches: Sequence[BranchTrace], answer_type: str) -> list[CandidateAnswer]:
    grouped: dict[str, list[BranchTrace]] = {}
    for branch in branches:
        if not branch.answer:
            continue
        key = _canonical_text(branch.answer, answer_type)
        grouped.setdefault(key, []).append(branch)
    total = sum(len(g) for g in grouped.values()) or 1
    out: list[CandidateAnswer] = []
    for key, members in sorted(grouped.items()):
        verifier_score = mean(float(b.verifier_score or 0.0) for b in members)
        tool_consistency = mean(float(b.tool_consistency or 0.0) for b in members)
        symbolic_check = mean(1.0 if b.symbolic_valid else 0.0 for b in members)
        novelty = mean((len(set(b.operator_sequence)) / max(1, len(b.operator_sequence))) if b.operator_sequence else 0.0 for b in members)
        out.append(
            CandidateAnswer(
                answer=str(members[0].answer or key),
                answer_canonical=key,
                branch_ids=[b.branch_id for b in members],
                verifier_score=verifier_score,
                tool_consistency=tool_consistency,
                answer_agreement=len(members) / float(total),
                branch_novelty=novelty,
                symbolic_check=symbolic_check,
                composite_score=0.0,
                cluster_size=len(members),
                entropy_penalty=0.0,
            )
        )
    return out


def _coerce_examples(rows: Sequence[Mapping[str, Any]]) -> list[CalibrationExample]:
    grouped_examples: list[CalibrationExample] = []
    if rows and any(isinstance(row.get("branches"), list) for row in rows if isinstance(row, Mapping)):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            problem_id = str(row.get("problem_id") or row.get("id") or "").strip()
            if not problem_id:
                continue
            branches_payload = row.get("branches")
            if not isinstance(branches_payload, list):
                branches_payload = []
            branches = []
            for idx, item in enumerate(branches_payload):
                if not isinstance(item, Mapping):
                    continue
                branch = CalibrationBranchRecord(
                    branch_id=str(item.get("branch_id") or item.get("id") or f"{problem_id}::branch::{idx:04d}"),
                    answer=str(item.get("answer") or ""),
                    verifier_score=_parse_float(item.get("verifier_score", 0.0)),
                    tool_consistency=_parse_float(item.get("tool_consistency", 0.0)),
                    branch_score=_parse_float(item.get("branch_score", item.get("score", 0.0))),
                    symbolic_valid=_parse_bool(item.get("symbolic_valid", False)),
                    full_reasoning=str(item.get("full_reasoning") or item.get("reasoning") or ""),
                    repaired=_parse_bool(item.get("repaired", False)),
                    self_critiqued=_parse_bool(item.get("self_critiqued", False)),
                    repair_count=int(_parse_float(item.get("repair_count", 0.0))),
                    retrieval_used=_parse_bool(item.get("retrieval_used", False)),
                    operator_sequence=list(item.get("operator_sequence") or []),
                    archetype_used=item.get("archetype_used"),
                    generation_time_sec=_parse_float(item.get("generation_time_sec", 0.0)),
                    logical_consistency=_parse_float(item.get("logical_consistency", 0.0)),
                    completeness=_parse_float(item.get("completeness", 0.0)),
                    repairability=_parse_float(item.get("repairability", 0.0)),
                    answer_correctness_likelihood=_parse_float(item.get("answer_correctness_likelihood", 0.0)),
                    symbolic_agreement=_parse_float(item.get("symbolic_agreement", 0.0)),
                    step_quality=_parse_float(item.get("step_quality", 0.0)),
                    prefix_quality=_parse_float(item.get("prefix_quality", 0.0)),
                    prm_prefix_quality=_parse_float(item.get("prm_prefix_quality", 0.0)),
                    prm_step_quality=_parse_float(item.get("prm_step_quality", 0.0)),
                    retrieval_compatibility=_parse_float(item.get("retrieval_compatibility", 0.0)),
                    retrieval_support=_parse_float(item.get("retrieval_support", 0.0)),
                    operator_reliability=_parse_float(item.get("operator_reliability", 0.0)),
                    discharge_fraction=_parse_float(item.get("discharge_fraction", 0.0)),
                    open_obligation_burden=_parse_float(item.get("open_obligation_burden", 0.0)),
                    verifier_decomposition=dict(item.get("verifier_decomposition") or {}),
                    route_compute_signals=dict(item.get("route_compute_signals") or item.get("compute_signals") or {}),
                    metadata=dict(item.get("metadata") or {}),
                )
                branches.append(branch)
            grouped_examples.append(
                CalibrationExample(
                    problem_id=problem_id,
                    raw_text=str(row.get("raw_text") or row.get("problem") or ""),
                    gold_answer="" if row.get("gold_answer") is None else str(row.get("gold_answer", row.get("answer", ""))),
                    answer_type=str(row.get("answer_type") or "non_negative_integer"),
                    domain=str(row.get("domain") or "mixed"),
                    branches=branches,
                    source_metadata=dict(row.get("source_metadata") or {"row_keys": sorted(str(key) for key in row.keys())}),
                )
            )
        return grouped_examples

    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        problem_id = str(row.get("problem_id") or row.get("id") or row.get("example_id") or "").strip()
        if not problem_id:
            continue
        bucket = grouped.setdefault(problem_id, {"problem_id": problem_id, "raw_text": str(row.get("raw_text") or row.get("problem") or ""), "gold_answer": "" if row.get("gold_answer") is None else str(row.get("gold_answer", row.get("answer_gold", row.get("target_answer", "")))), "answer_type": str(row.get("answer_type") or "non_negative_integer"), "domain": str(row.get("domain") or "mixed"), "branches": [], "source_metadata": dict(row.get("source_metadata") or {})})
        bucket["branches"].append(_coerce_branch_trace(problem_id, row, index))
    for value in grouped.values():
        grouped_examples.append(
            CalibrationExample(
                problem_id=value["problem_id"],
                raw_text=value["raw_text"],
                gold_answer=value["gold_answer"],
                answer_type=value["answer_type"],
                domain=value["domain"],
                branches=[
                    CalibrationBranchRecord(
                        branch_id=branch.branch_id,
                        answer=branch.answer or "",
                        verifier_score=float(branch.verifier_score or 0.0),
                        tool_consistency=float(branch.tool_consistency or 0.0),
                        branch_score=float(branch.branch_score or 0.0),
                        symbolic_valid=bool(branch.symbolic_valid),
                        full_reasoning=branch.full_reasoning,
                        repaired=bool(branch.repaired),
                        self_critiqued=bool(branch.self_critiqued),
                        repair_count=int(branch.repair_count),
                        retrieval_used=bool(branch.retrieval_used),
                        operator_sequence=list(branch.operator_sequence),
                        archetype_used=branch.archetype_used,
                        generation_time_sec=float(branch.generation_time_sec or 0.0),
                        logical_consistency=float(branch.logical_consistency or 0.0),
                        completeness=float(branch.completeness or 0.0),
                        repairability=float(branch.repairability or 0.0),
                        answer_correctness_likelihood=float(branch.answer_correctness_likelihood or 0.0),
                        symbolic_agreement=float(branch.symbolic_agreement or 0.0),
                        step_quality=float(branch.step_quality or 0.0),
                        prefix_quality=float(branch.prefix_quality or 0.0),
                        prm_prefix_quality=float(branch.prm_prefix_quality or 0.0),
                        prm_step_quality=float(getattr(branch, "prm_step_quality", 0.0) or 0.0),
                        retrieval_compatibility=float(branch.retrieval_compatibility or 0.0),
                        retrieval_support=float(branch.retrieval_support or 0.0),
                        operator_reliability=float(branch.operator_reliability or 0.0),
                        discharge_fraction=float(branch.discharge_fraction or 0.0),
                        open_obligation_burden=float(branch.open_obligation_burden or 0.0),
                        verifier_decomposition=dict(branch.verifier_decomposition),
                        route_compute_signals=dict((branch.metadata or {}).get("route_compute_signals", {})),
                        metadata=dict(branch.metadata or {}),
                    )
                    for branch in sorted(value["branches"], key=lambda item: item.branch_id)
                ],
                source_metadata=value["source_metadata"],
            )
        )
    return sorted(grouped_examples, key=lambda item: item.problem_id)


def _validate_examples(examples: Sequence[CalibrationExample]) -> InputValidationReport:
    duplicates, seen, missing_gold, empty_branch = [], set(), [], []
    total_branches = 0
    for example in examples:
        if example.problem_id in seen:
            duplicates.append(example.problem_id)
        seen.add(example.problem_id)
        if not str(example.gold_answer).strip():
            missing_gold.append(example.problem_id)
        if not example.branches:
            empty_branch.append(example.problem_id)
        total_branches += len(example.branches)
    notes = []
    if duplicates:
        notes.append("Duplicate problem IDs detected; later entries were retained only at normalization time.")
    if missing_gold:
        notes.append("Some calibration examples do not include gold answers; they will be excluded from scoring.")
    if empty_branch:
        notes.append("Some calibration examples have no branches; they will always fall back to default selection.")
    return InputValidationReport(
        valid=bool(examples),
        num_examples=len(examples),
        num_problem_ids=len(seen),
        num_branches=total_branches,
        duplicate_problem_ids=sorted(set(duplicates)),
        missing_gold_problem_ids=sorted(set(missing_gold)),
        empty_branch_problem_ids=sorted(set(empty_branch)),
        notes=notes,
    )


def _resolve_examples(paths: Sequence[str]) -> tuple[list[CalibrationExample], list[ResolvedInputSource]]:
    all_examples: list[CalibrationExample] = []
    sources: list[ResolvedInputSource] = []
    for requested in paths:
        resolved_path = _normalize_path(requested)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {requested}")
        rows, source_kind = _resolve_input_rows(resolved_path)
        examples = _coerce_examples(rows)
        all_examples.extend(examples)
        sources.append(ResolvedInputSource(requested_path=requested, resolved_path=str(resolved_path), source_kind=source_kind, row_count=len(rows)))
    merged: dict[str, CalibrationExample] = {}
    for example in sorted(all_examples, key=lambda item: item.problem_id):
        merged[example.problem_id] = example
    return list(merged.values()), sources


def _iter_parameter_grid(config: CalibrationScriptConfig) -> list[CalibrationParameterSet]:
    parameter_sets = []
    for vt, bb, eb, mbs, mode, rcf, rbias, orf, pbp in product(
        sorted(set(float(v) for v in config.verifier_thresholds)),
        sorted(set(int(v) for v in config.branch_budgets)),
        sorted(set(float(v) for v in config.entropy_betas)),
        sorted(set(float(v) for v in config.min_branch_scores)),
        [BranchSelectionMode(value) for value in sorted({mode.value if isinstance(mode, BranchSelectionMode) else str(mode) for mode in config.selection_modes})],
        sorted(set(float(v) for v in config.retrieval_compatibility_floors)),
        sorted(set(float(v) for v in config.repairability_biases)),
        sorted(set(float(v) for v in config.operator_reliability_floors)),
        sorted(set(float(v) for v in config.proof_burden_penalties)),
    ):
        parameter_sets.append(
            CalibrationParameterSet(
                verifier_threshold=vt,
                branch_budget=bb,
                entropy_beta=eb,
                min_branch_score=mbs,
                selection_mode=mode,
                retrieval_compatibility_floor=rcf,
                repairability_bias=rbias,
                operator_reliability_floor=orf,
                proof_burden_penalty=pbp,
            )
        )
    parameter_sets.sort(key=lambda item: item.stable_id())
    return parameter_sets


def _filter_branches(branches: Sequence[BranchTrace], params: CalibrationParameterSet) -> list[BranchTrace]:
    ranked = sorted(branches, key=lambda branch: (-(1.0 if branch.symbolic_valid else 0.0), -float(branch.verifier_score or 0.0), -float(branch.branch_score or 0.0), branch.branch_id))
    filtered: list[BranchTrace] = []
    for branch in ranked:
        if float(branch.verifier_score or 0.0) < params.verifier_threshold:
            continue
        if float(branch.branch_score or 0.0) < params.min_branch_score:
            continue
        if params.selection_mode == BranchSelectionMode.REQUIRE_SYMBOLIC and not branch.symbolic_valid:
            continue
        if float(getattr(branch, "retrieval_compatibility", 0.0) or 0.0) < params.retrieval_compatibility_floor:
            continue
        if float(getattr(branch, "operator_reliability", 0.0) or 0.0) < params.operator_reliability_floor:
            continue
        if params.repairability_bias > 0.0 and float(getattr(branch, "repairability", 0.0) or 0.0) + params.repairability_bias < 0.25:
            continue
        filtered.append(branch)
    if params.selection_mode == BranchSelectionMode.PREFER_SYMBOLIC:
        symbolic = [branch for branch in filtered if branch.symbolic_valid]
        nonsymbolic = [branch for branch in filtered if not branch.symbolic_valid]
        filtered = symbolic + nonsymbolic
    return filtered[: params.branch_budget]


def _build_problem_stub(example: CalibrationExample) -> ParsedProblem:
    return ParsedProblem(
        problem_id=example.problem_id,
        raw_text=example.raw_text,
        latex_spans=[],
        knowns=[],
        unknowns=[],
        constraints=[],
        domain=str(example.domain),
        target="",
        answer_type=example.answer_type or "non_negative_integer",
        difficulty_seed=0.5,
        likely_archetypes=[],
        symmetries=[],
        parity_cues=[],
        integrality_constraints=[],
        symbol_table={},
    )
# scripts/calibrate_search.py
# Replace the weighting/selection part inside _evaluate_one(...) with this version.

def _evaluate_one(example: CalibrationExample, params: CalibrationParameterSet) -> PerProblemResult:
    original_branches = [
        BranchTrace(
            branch_id=record.branch_id,
            problem_id=example.problem_id,
            steps=[],
            full_reasoning=record.full_reasoning,
            answer=record.answer or None,
            answer_canonical=None,
            symbolic_valid=record.symbolic_valid,
            branch_score=record.branch_score,
            verifier_score=record.verifier_score,
            tool_consistency=record.tool_consistency,
            logical_consistency=record.logical_consistency,
            completeness=record.completeness,
            repairability=record.repairability,
            answer_correctness_likelihood=record.answer_correctness_likelihood,
            symbolic_agreement=record.symbolic_agreement,
            step_quality=record.step_quality,
            prefix_quality=record.prefix_quality,
            prm_prefix_quality=record.prm_prefix_quality or record.prefix_quality,
            prm_step_quality=record.prm_step_quality,
            retrieval_compatibility=record.retrieval_compatibility,
            retrieval_support=record.retrieval_support,
            operator_reliability=record.operator_reliability,
            discharge_fraction=record.discharge_fraction,
            open_obligation_burden=record.open_obligation_burden,
            verifier_decomposition=dict(record.verifier_decomposition),
            self_critiqued=record.self_critiqued,
            repaired=record.repaired,
            repair_count=record.repair_count,
            operator_sequence=list(record.operator_sequence),
            archetype_used=record.archetype_used,
            retrieval_used=record.retrieval_used,
            generation_time_sec=record.generation_time_sec,
            metadata={**dict(record.metadata), "route_compute_signals": dict(record.route_compute_signals)},
        )
        for record in example.branches
    ]

    filtered = _filter_branches(original_branches, params)
    candidates = _group_candidate_answers(filtered, example.answer_type)

    weighting_config = EntropyWeightingConfig(
        uncertainty_penalty_weight=float(params.entropy_beta),
    )
    weighting = score_answer_clusters(filtered, config=weighting_config)

    selection = select_final_answer(
        problem_id=example.problem_id,
        weighted_clusters=weighting,
        num_branches_generated=len(original_branches),
        num_branches_survived=len(filtered),
        solve_time_sec=sum(float(b.generation_time_sec or 0.0) for b in filtered),
        method_used="calibrate_search_selector",
        config=FinalSelectorConfig(),
    )
    predicted = str(selection.prediction.final_answer)
    gold_key = _canonical_text(example.gold_answer, example.answer_type)
    pred_key = _canonical_text(predicted, example.answer_type)
    correct = bool(gold_key and pred_key and gold_key == pred_key)

    return PerProblemResult(
        problem_id=example.problem_id,
        gold_answer=example.gold_answer,
        predicted_answer=predicted,
        correct=correct,
        candidate_count=len(candidates),
        filtered_branch_count=len(filtered),
        total_branch_count=len(original_branches),
        winner_confidence=float(selection.prediction.confidence),
        method_used=selection.prediction.method_used,
        selection_warning_count=len(selection.warnings),
        metadata={
            "signal_decomposition": dict(selection.prediction.signal_decomposition or {}),
            "calibration_summary": dict(selection.prediction.calibration_summary or {}),
        },
    )


def _summarize_scores(params: CalibrationParameterSet, results: Sequence[PerProblemResult], examples: Sequence[CalibrationExample]) -> CalibrationScoreSummary:
    split_buckets: dict[str, list[PerProblemResult]] = {}
    signal_accumulator: dict[str, list[float]] = defaultdict(list)
    for example, result in zip(examples, results):
        split = example.split_name.lower()
        if split == "validation":
            split = "valid"
        split_buckets.setdefault(split, []).append(result)
        for branch in example.branches:
            for key, value in (branch.route_compute_signals or {}).items():
                signal_accumulator[f"route_{key}"].append(float(value))
            signal_accumulator["logical_consistency"].append(float(branch.logical_consistency))
            signal_accumulator["completeness"].append(float(branch.completeness))
            signal_accumulator["repairability"].append(float(branch.repairability))
            signal_accumulator["answer_correctness_likelihood"].append(float(branch.answer_correctness_likelihood))
            signal_accumulator["symbolic_agreement"].append(float(branch.symbolic_agreement))
            signal_accumulator["step_quality"].append(float(branch.step_quality))
            signal_accumulator["prefix_quality"].append(float(branch.prefix_quality))
            signal_accumulator["prm_prefix_quality"].append(float(branch.prm_prefix_quality))
            signal_accumulator["prm_step_quality"].append(float(branch.prm_step_quality))
            signal_accumulator["retrieval_compatibility"].append(float(branch.retrieval_compatibility))
            signal_accumulator["retrieval_support"].append(float(branch.retrieval_support))
            signal_accumulator["operator_reliability"].append(float(branch.operator_reliability))
            signal_accumulator["discharge_fraction"].append(float(branch.discharge_fraction))
            signal_accumulator["open_obligation_burden"].append(float(branch.open_obligation_burden))
    split_summaries: dict[str, Any] = {}
    split_accuracies: list[float] = []
    for split, split_results in sorted(split_buckets.items()):
        num_correct = sum(1 for item in split_results if item.correct)
        accuracy = num_correct / float(len(split_results)) if split_results else 0.0
        split_accuracies.append(accuracy)
        split_summaries[split] = {
            "num_examples": len(split_results),
            "num_correct": num_correct,
            "accuracy": accuracy,
            "mean_confidence": mean(r.winner_confidence for r in split_results) if split_results else 0.0,
        }
    robust_accuracy = mean(split_accuracies) if split_accuracies else 0.0
    robustness_penalty = (max(split_accuracies) - min(split_accuracies)) if len(split_accuracies) > 1 else 0.0
    solved_nonzero_rate = sum(1 for item in results if item.predicted_answer not in {"", "0"}) / float(len(results)) if results else 0.0
    score = max(0.0, robust_accuracy - 0.10 * robustness_penalty)
    signal_summary = {key: mean(values) for key, values in sorted(signal_accumulator.items()) if values}
    return CalibrationScoreSummary(
        parameter_id=params.stable_id(),
        parameters=params.model_dump(mode="json"),
        num_examples=len(results),
        num_correct=sum(1 for item in results if item.correct),
        accuracy=(sum(1 for item in results if item.correct) / float(len(results)) if results else 0.0),
        mean_confidence=(mean(r.winner_confidence for r in results) if results else 0.0),
        mean_filtered_branch_count=(mean(r.filtered_branch_count for r in results) if results else 0.0),
        mean_candidate_count=(mean(r.candidate_count for r in results) if results else 0.0),
        solved_nonzero_rate=solved_nonzero_rate,
        score=score,
        split_summaries=split_summaries,
        robustness_penalty=robustness_penalty,
        signal_summary=signal_summary,
    )


def run_calibration(config: CalibrationScriptConfig) -> tuple[RunStatus, CalibrationRunManifest]:
    _set_seed(config.seed)
    examples, sources = _resolve_examples(config.input_paths)
    if config.max_problems is not None:
        examples = examples[: config.max_problems]
    validation = _validate_examples(examples)
    if not examples:
        manifest = CalibrationRunManifest(run_name=config.run_name, run_id=_stable_hash("run", {"run_name": config.run_name}), status=RunStatus.MISSING_INPUT.value)
        return RunStatus.MISSING_INPUT, manifest
    if not validation.valid:
        manifest = CalibrationRunManifest(run_name=config.run_name, run_id=_stable_hash("run", {"run_name": config.run_name, "invalid": True}), status=RunStatus.INVALID_DATASET.value, validation=validation.model_dump(mode="json"))
        return RunStatus.INVALID_DATASET, manifest

    summaries: list[CalibrationScoreSummary] = []
    per_param_results: dict[str, list[PerProblemResult]] = {}
    for params in _iter_parameter_grid(config):
        results = [_evaluate_one(example, params) for example in examples if str(example.gold_answer).strip()]
        summary = _summarize_scores(params, results, examples)
        summaries.append(summary)
        per_param_results[summary.parameter_id] = results

    summaries.sort(key=lambda item: (-item.score, -item.accuracy, item.robustness_penalty, item.parameter_id))
    best_summary = summaries[0]
    run_id = _stable_hash("calibration_run", {"config": config.model_dump(mode="json"), "best_parameter_id": best_summary.parameter_id})

    output_dir = Path(config.output_dir)
    if output_dir.exists() and config.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = CalibrationRunManifest(
        run_name=config.run_name,
        run_id=run_id,
        status=RunStatus.COMPLETED.value,
        input_sources=[source.__dict__ for source in sources],
        validation=validation.model_dump(mode="json"),
        parameter_grid_size=len(summaries),
        best_parameter_id=best_summary.parameter_id,
        best_summary=best_summary.model_dump(mode="json"),
        split_summary=best_summary.split_summaries,
        artifacts={
            "manifest_path": str(output_dir / "manifest.json"),
            "best_summary_path": str(output_dir / "best_summary.json"),
            "summaries_path": str(output_dir / "summaries.jsonl"),
            "per_problem_path": str(output_dir / "per_problem_best.jsonl"),
        },
        notes=["Calibration consumes decomposed runtime signals and reports split-safe robust summaries."],
    )

    _write_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))
    _write_json(output_dir / "best_summary.json", best_summary.model_dump(mode="json"))
    _write_jsonl(output_dir / "summaries.jsonl", [summary.model_dump(mode="json") for summary in summaries])
    _write_jsonl(output_dir / "per_problem_best.jsonl", [result.model_dump(mode="json") for result in per_param_results[best_summary.parameter_id]])

    if config.write_csv:
        csv_path = output_dir / "top_configs.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["parameter_id", "score", "accuracy", "robustness_penalty", "mean_confidence"])
            writer.writeheader()
            for summary in summaries[: config.top_n_configs]:
                writer.writerow({
                    "parameter_id": summary.parameter_id,
                    "score": summary.score,
                    "accuracy": summary.accuracy,
                    "robustness_penalty": summary.robustness_penalty,
                    "mean_confidence": summary.mean_confidence,
                })
        manifest.artifacts["top_configs_csv"] = str(csv_path)
        _write_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))

    return RunStatus.COMPLETED, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="inputs", action="append", default=[])
    parser.add_argument("--run-name", default="search_calibration")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-version", default=OFFLINE_ARTIFACT_VERSION)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--top-n-configs", type=int, default=5)
    parser.add_argument("--verifier-thresholds", nargs="*", default=None)
    parser.add_argument("--branch-budgets", nargs="*", default=None)
    parser.add_argument("--entropy-betas", nargs="*", default=None)
    parser.add_argument("--min-branch-scores", nargs="*", default=None)
    parser.add_argument("--selection-modes", nargs="*", default=None)
    args = parser.parse_args(argv)
    try:
        config = CalibrationScriptConfig(
            run_name=args.run_name,
            input_paths=list(args.inputs),
            output_dir=args.output_dir,
            artifact_version=args.artifact_version,
            seed=args.seed,
            max_problems=args.max_problems,
            overwrite=args.overwrite,
            top_n_configs=args.top_n_configs,
            verifier_thresholds=[float(v) for v in args.verifier_thresholds] if args.verifier_thresholds else [0.45, 0.55, 0.65],
            branch_budgets=[int(v) for v in args.branch_budgets] if args.branch_budgets else [16, 32, 64],
            entropy_betas=[float(v) for v in args.entropy_betas] if args.entropy_betas else [0.18, 0.24, 0.30],
            min_branch_scores=[float(v) for v in args.min_branch_scores] if args.min_branch_scores else [0.0, 0.20],
            selection_modes=[BranchSelectionMode(v) for v in args.selection_modes] if args.selection_modes else [BranchSelectionMode.ANY, BranchSelectionMode.PREFER_SYMBOLIC],
        )
    except ValidationError as exc:
        LOGGER.error("Invalid calibration config: %s", exc)
        return 2
    status, _ = run_calibration(config)
    return 0 if status is RunStatus.COMPLETED else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

from __future__ import annotations

"""Production CLI entrypoint for operator mining and promotion support.

This script is intentionally stricter than a notebook helper:
- validates upstream trace artifacts before consuming them
- runs the structured cluster/mining pipeline
- applies explicit promotion gates with manifest-backed outputs
- emits downstream-usable promoted operator library rows
- writes provenance and summary artifacts for every run
"""

import argparse
import json
import sys
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    yaml = None  # type: ignore

from src.common.constants import OFFLINE_ARTIFACT_VERSION
from src.common.schemas import BranchTrace
from src.offline.cluster_and_mine import (
    ClusterAndMineArtifact,
    ClusterAndMineOrchestrator,
    ClusterSummary,
    MiningConfig,
    TraceArtifactRecord,
    _stable_hash,
    _stable_json,
)
from src.offline.trace_distillation import DistilledTraceArtifact, DistilledTraceRecord, load_distilled_artifact
from src.operators.operator_library import OperatorLibrary
from src.operators.operator_types import OperatorDescriptor, OperatorMiningCandidate


SCHEMA_VERSION = "mine_operators.script.v1"
ARTIFACT_KIND = "operator_promotion_run"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class PromotionGates(StrictModel):
    min_support_count: int = Field(default=3, ge=1)
    min_symbolic_success_rate: float = Field(default=0.80, ge=0.0, le=1.0)
    min_mean_verifier_gain: float = Field(default=0.02)
    min_cluster_operator_purity: float = Field(default=0.85, ge=0.0, le=1.0)
    min_cluster_symbolic_success_rate: float = Field(default=0.80, ge=0.0, le=1.0)
    min_cluster_mean_verifier_score: float = Field(default=0.55, ge=0.0, le=1.0)
    max_cluster_failure_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    min_distinct_problem_support: int = Field(default=1, ge=1)
    require_cluster_promotion_safe: bool = True
    require_candidate_promotion_safe: bool = True
    require_any_promotions: bool = False


class MineOperatorsScriptConfig(StrictModel):
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    output_root: str = "artifacts/operator_miner"
    write_cluster_artifact: bool = True
    fail_on_source_validation_error: bool = True
    source_paths: tuple[str, ...] = Field(default_factory=tuple)
    mining: MiningConfig = Field(default_factory=MiningConfig)
    promotion: PromotionGates = Field(default_factory=PromotionGates)


class LoadedSource(StrictModel):
    path: str
    source_kind: str
    record_count: int = Field(default=0, ge=0)
    source_digest: str
    manifest_payload: dict[str, Any] = Field(default_factory=dict)


class CandidatePromotionDecision(StrictModel):
    operator_name: str
    promoted: bool
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    descriptor_row: dict[str, Any] = Field(default_factory=dict)


class PromotionRunMetadata(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    run_id: str
    run_fingerprint: str
    config_digest: str
    source_digest: str
    loaded_source_count: int
    total_input_records: int
    cluster_artifact_id: str
    cluster_artifact_dir: str | None = None
    cluster_artifact_manifest_path: str | None = None
    total_candidates: int
    promoted_candidates: int
    rejected_candidates: int


class PromotionRunManifest(StrictModel):
    metadata: PromotionRunMetadata
    config: MineOperatorsScriptConfig
    loaded_sources: tuple[LoadedSource, ...] = Field(default_factory=tuple)
    gate_summary: dict[str, Any] = Field(default_factory=dict)
    promoted_operator_names: tuple[str, ...] = Field(default_factory=tuple)
    rejected_operator_names: tuple[str, ...] = Field(default_factory=tuple)


class PromotionWriteResult(StrictModel):
    artifact_dir: str
    manifest_path: str
    summary_path: str
    promotion_candidates_path: str
    promoted_operator_library_path: str
    cluster_artifact_manifest_path: str | None = None


class SourceValidationError(RuntimeError):
    pass


class PromotionFailure(RuntimeError):
    pass


class _JsonLineSink:
    @staticmethod
    def write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_stable_json(dict(row)) + "\n")


def _load_structured_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = target.read_text(encoding="utf-8")
    suffix = target.suffix.lower()
    if suffix in {".json"}:
        payload = json.loads(text or "{}")
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML config requested but PyYAML is not installed")
        payload = yaml.safe_load(text) or {}
    else:
        try:
            payload = json.loads(text or "{}")
        except json.JSONDecodeError:
            if yaml is None:
                raise RuntimeError(f"Unsupported config extension for {path}")
            payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object-like config payload in {path}")
    return payload


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _sha1_text(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SourceValidationError(f"Invalid JSONL row in {path} line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SourceValidationError(f"Expected object rows in {path} line {line_no}")
            rows.append(payload)
    return rows


def _looks_like_generated_raw_trace_row(row: Mapping[str, Any]) -> bool:
    record_type = str(row.get("record_type") or "").strip().lower()
    if record_type == "raw_branch_trace":
        return True
    return (
        "problem_id" in row
        and isinstance(row.get("steps"), list)
        and ("route_snapshot" in row or "operator_sequence" in row)
    )


def _raw_trace_artifact_file(path: Path) -> Path | None:
    if path.is_dir():
        candidate = path / "raw_traces.jsonl"
    elif path.name == "manifest.json":
        candidate = path.with_name("raw_traces.jsonl")
    else:
        return None
    return candidate if candidate.exists() else None


def _validate_distilled_artifact_path(path: Path) -> LoadedSource:
    artifact = load_distilled_artifact(path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha1_text(_stable_json({"manifest": manifest_payload, "record_count": len(artifact.records)}))
    return LoadedSource(
        path=str(path),
        source_kind="distilled_trace_artifact",
        record_count=len(artifact.records),
        source_digest=digest,
        manifest_payload=manifest_payload,
    )


def _validate_raw_trace_artifact_path(path: Path, raw_path: Path) -> LoadedSource:
    rows = _load_jsonl_records(raw_path)
    validated = 0
    for row in rows:
        if not _looks_like_generated_raw_trace_row(row):
            raise SourceValidationError(
                f"Row in {raw_path} is not a supported generated raw trace record"
            )
        validated += 1
    manifest_payload: dict[str, Any] = {}
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            manifest_payload = payload
    digest = _sha1_text(
        _stable_json(
            {
                "manifest": manifest_payload,
                "raw_trace_path": str(raw_path),
                "record_count": validated,
            }
        )
    )
    return LoadedSource(
        path=str(path),
        source_kind="raw_trace_run",
        record_count=validated,
        source_digest=digest,
        manifest_payload=manifest_payload,
    )


def _validate_jsonl_source(path: Path) -> LoadedSource:
    rows = _load_jsonl_records(path)
    validated = 0
    kinds: set[str] = set()
    for row in rows:
        try:
            DistilledTraceRecord.model_validate(row)
            kinds.add("distilled_trace_record")
            validated += 1
            continue
        except ValidationError:
            pass
        try:
            BranchTrace.model_validate(row)
            kinds.add("branch_trace")
            validated += 1
            continue
        except ValidationError as exc:
            if _looks_like_generated_raw_trace_row(row):
                kinds.add("generated_raw_trace")
                validated += 1
                continue
            raise SourceValidationError(
                f"Row in {path} is neither DistilledTraceRecord, BranchTrace, nor generated raw trace: {exc}"
            ) from exc
    digest = _sha1_text(_stable_json({"path": str(path), "rows": rows}))
    return LoadedSource(
        path=str(path),
        source_kind="+".join(sorted(kinds)) if kinds else "jsonl",
        record_count=validated,
        source_digest=digest,
        manifest_payload={"path": str(path), "record_count": validated, "source_kind": "+".join(sorted(kinds))},
    )


def _load_source_payloads(source_paths: Sequence[str]) -> tuple[list[Any], tuple[LoadedSource, ...]]:
    payloads: list[Any] = []
    sources: list[LoadedSource] = []
    for raw_path in source_paths:
        path = Path(raw_path)
        if not path.exists():
            raise SourceValidationError(f"Source path does not exist: {raw_path}")
        if path.is_dir() or path.name == "manifest.json":
            raw_trace_file = _raw_trace_artifact_file(path)
            if raw_trace_file is not None:
                src = _validate_raw_trace_artifact_path(path, raw_trace_file)
                payloads.extend(_load_jsonl_records(raw_trace_file))
                sources.append(src)
                continue
            src = _validate_distilled_artifact_path(path)
            payloads.append(load_distilled_artifact(path))
            sources.append(src)
            continue
        if path.suffix.lower() == ".jsonl":
            src = _validate_jsonl_source(path)
            rows = _load_jsonl_records(path)
            payloads.extend(rows)
            sources.append(src)
            continue
        raise SourceValidationError(
            f"Unsupported source path {raw_path}. Expected distilled artifact dir/manifest.json or .jsonl"
        )
    return payloads, tuple(sources)


def _dominant_cluster_metrics(
    candidate: OperatorMiningCandidate,
    clusters: Sequence[ClusterSummary],
) -> dict[str, Any]:
    relevant = [cluster for cluster in clusters if cluster.dominant_operator == candidate.operator_name]
    if not relevant:
        return {
            "cluster_count": 0,
            "max_cluster_support": 0,
            "max_cluster_operator_purity": 0.0,
            "max_cluster_symbolic_success_rate": 0.0,
            "max_cluster_mean_verifier_score": 0.0,
            "min_cluster_failure_rate": 1.0,
            "max_distinct_problem_support": 0,
            "any_cluster_promotion_safe": False,
        }
    best_support = max(cluster.member_count for cluster in relevant)
    best_purity = max(cluster.operator_purity for cluster in relevant)
    best_symbolic = max(cluster.symbolic_success_rate for cluster in relevant)
    best_verifier = max(cluster.mean_verifier_score for cluster in relevant)
    min_failure = min(cluster.failure_rate for cluster in relevant)
    best_problem_support = max(cluster.distinct_problem_count for cluster in relevant)
    return {
        "cluster_count": len(relevant),
        "max_cluster_support": best_support,
        "max_cluster_operator_purity": round(best_purity, 6),
        "max_cluster_symbolic_success_rate": round(best_symbolic, 6),
        "max_cluster_mean_verifier_score": round(best_verifier, 6),
        "min_cluster_failure_rate": round(min_failure, 6),
        "max_distinct_problem_support": best_problem_support,
        "any_cluster_promotion_safe": any(cluster.promotion.promotion_safe for cluster in relevant),
        "cluster_ids": [cluster.cluster_id for cluster in relevant],
    }


def _descriptor_row(candidate: OperatorMiningCandidate) -> dict[str, Any]:
    return {
        "operator_name": candidate.operator_name,
        "family": candidate.family.value,
        "description": candidate.description,
        "compatible_domains": list(candidate.compatible_domains),
        "compatible_archetypes": list(candidate.compatible_archetypes),
        "tool_capabilities": [cap.value for cap in candidate.tool_capabilities],
        "preconditions": [item.model_dump(mode="json") for item in candidate.preconditions],
        "postconditions": [item.model_dump(mode="json") for item in candidate.postconditions],
        "failure_modes": [mode.value for mode in candidate.failure_modes],
        "repair_neighbors": [item.model_dump(mode="json") for item in candidate.repair_neighbors],
        "tags": ["mined", "promoted_candidate"],
        "mined": True,
        "metadata": dict(candidate.metadata),
    }


def _evaluate_candidate(
    candidate: OperatorMiningCandidate,
    clusters: Sequence[ClusterSummary],
    gates: PromotionGates,
) -> CandidatePromotionDecision:
    cluster_metrics = _dominant_cluster_metrics(candidate, clusters)
    reasons: list[str] = []

    if gates.require_candidate_promotion_safe and not candidate.promotion_safe:
        reasons.append("candidate_not_miner_promotion_safe")
    if candidate.support_count < gates.min_support_count:
        reasons.append("insufficient_support")
    if candidate.symbolic_success_rate < gates.min_symbolic_success_rate:
        reasons.append("symbolic_success_below_threshold")
    if candidate.mean_verifier_gain < gates.min_mean_verifier_gain:
        reasons.append("verifier_gain_below_threshold")
    if cluster_metrics["max_cluster_operator_purity"] < gates.min_cluster_operator_purity:
        reasons.append("cluster_purity_below_threshold")
    if cluster_metrics["max_cluster_symbolic_success_rate"] < gates.min_cluster_symbolic_success_rate:
        reasons.append("cluster_symbolic_success_below_threshold")
    if cluster_metrics["max_cluster_mean_verifier_score"] < gates.min_cluster_mean_verifier_score:
        reasons.append("cluster_verifier_score_below_threshold")
    if cluster_metrics["min_cluster_failure_rate"] > gates.max_cluster_failure_rate:
        reasons.append("cluster_failure_rate_above_threshold")
    if cluster_metrics["max_distinct_problem_support"] < gates.min_distinct_problem_support:
        reasons.append("insufficient_distinct_problem_support")
    if gates.require_cluster_promotion_safe and not cluster_metrics["any_cluster_promotion_safe"]:
        reasons.append("no_promotion_safe_cluster")

    promoted = not reasons
    metrics = {
        "support_count": candidate.support_count,
        "symbolic_success_rate": round(candidate.symbolic_success_rate, 6),
        "mean_verifier_gain": round(candidate.mean_verifier_gain, 6),
        **cluster_metrics,
    }
    return CandidatePromotionDecision(
        operator_name=candidate.operator_name,
        promoted=promoted,
        reasons=reasons,
        metrics=metrics,
        descriptor_row=_descriptor_row(candidate),
    )


def _write_promotion_outputs(
    *,
    config: MineOperatorsScriptConfig,
    loaded_sources: Sequence[LoadedSource],
    cluster_artifact: ClusterAndMineArtifact,
    cluster_output_dir: str | None,
    decisions: Sequence[CandidatePromotionDecision],
) -> PromotionWriteResult:
    config_payload = config.model_dump(mode="json")
    config_digest = _stable_hash("mine_ops_config", config_payload)
    source_digest = _stable_hash(
        "mine_ops_sources",
        [source.model_dump(mode="json") for source in loaded_sources],
    )
    run_fingerprint = _stable_hash(
        "mine_ops_run",
        {
            "cluster_artifact_id": cluster_artifact.metadata.artifact_id,
            "config_digest": config_digest,
            "source_digest": source_digest,
            "candidate_names": [decision.operator_name for decision in decisions],
        },
    )
    run_id = _stable_hash(
        "operator_promotion_run",
        {
            "artifact_version": config.artifact_version,
            "run_fingerprint": run_fingerprint,
        },
    )

    artifact_dir = Path(config.output_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    promotion_candidates_path = artifact_dir / "promotion_candidates.jsonl"
    promoted_library_path = artifact_dir / "promoted_operator_library.json"
    manifest_path = artifact_dir / "manifest.json"
    summary_path = artifact_dir / "summary.json"

    promoted = [decision for decision in decisions if decision.promoted]
    rejected = [decision for decision in decisions if not decision.promoted]

    _JsonLineSink.write(
        promotion_candidates_path,
        (
            {
                "operator_name": decision.operator_name,
                "promoted": decision.promoted,
                "reasons": list(decision.reasons),
                "metrics": dict(decision.metrics),
                "descriptor": dict(decision.descriptor_row),
            }
            for decision in decisions
        ),
    )
    promoted_library_path.write_text(
        json.dumps([decision.descriptor_row for decision in promoted], indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = PromotionRunMetadata(
        artifact_version=config.artifact_version,
        run_id=run_id,
        run_fingerprint=run_fingerprint,
        config_digest=config_digest,
        source_digest=source_digest,
        loaded_source_count=len(loaded_sources),
        total_input_records=sum(source.record_count for source in loaded_sources),
        cluster_artifact_id=cluster_artifact.metadata.artifact_id,
        cluster_artifact_dir=cluster_output_dir,
        cluster_artifact_manifest_path=str(Path(cluster_output_dir) / "manifest.json") if cluster_output_dir else None,
        total_candidates=len(decisions),
        promoted_candidates=len(promoted),
        rejected_candidates=len(rejected),
    )
    manifest = PromotionRunManifest(
        metadata=metadata,
        config=config,
        loaded_sources=tuple(loaded_sources),
        gate_summary=config.promotion.model_dump(mode="json"),
        promoted_operator_names=tuple(sorted(decision.operator_name for decision in promoted)),
        rejected_operator_names=tuple(sorted(decision.operator_name for decision in rejected)),
    )
    manifest_path.write_text(_stable_json(manifest.model_dump(mode="json")) + "\n", encoding="utf-8")

    summary_payload = {
        "run_id": run_id,
        "cluster_artifact_id": cluster_artifact.metadata.artifact_id,
        "total_candidates": len(decisions),
        "promoted_candidates": len(promoted),
        "rejected_candidates": len(rejected),
        "promoted_operator_names": [decision.operator_name for decision in promoted],
        "rejected_operator_names": [decision.operator_name for decision in rejected],
        "gate_summary": config.promotion.model_dump(mode="json"),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    return PromotionWriteResult(
        artifact_dir=str(artifact_dir),
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
        promotion_candidates_path=str(promotion_candidates_path),
        promoted_operator_library_path=str(promoted_library_path),
        cluster_artifact_manifest_path=(str(Path(cluster_output_dir) / "manifest.json") if cluster_output_dir else None),
    )


def _print_summary(
    *,
    write_result: PromotionWriteResult,
    decisions: Sequence[CandidatePromotionDecision],
    cluster_artifact: ClusterAndMineArtifact,
) -> None:
    promoted = [decision for decision in decisions if decision.promoted]
    rejected = [decision for decision in decisions if not decision.promoted]
    print(f"[mine_operators] cluster artifact: {cluster_artifact.metadata.artifact_id}")
    print(f"[mine_operators] total records: {cluster_artifact.metadata.record_count}")
    print(f"[mine_operators] clusters: {cluster_artifact.metadata.cluster_count}")
    print(f"[mine_operators] candidates emitted: {len(decisions)}")
    print(f"[mine_operators] promoted: {len(promoted)}")
    print(f"[mine_operators] rejected: {len(rejected)}")
    print(f"[mine_operators] promotion artifact: {write_result.artifact_dir}")
    if promoted:
        print("[mine_operators] promoted operator names:")
        for decision in promoted:
            print(f"  - {decision.operator_name}")
    if rejected:
        print("[mine_operators] rejected operator names:")
        for decision in rejected:
            joined = ", ".join(decision.reasons) if decision.reasons else "unknown"
            print(f"  - {decision.operator_name}: {joined}")


def run_mining(config: MineOperatorsScriptConfig) -> PromotionWriteResult:
    if not config.source_paths:
        raise SourceValidationError("No source paths provided. Use --source or config.source_paths")

    payloads, loaded_sources = _load_source_payloads(config.source_paths)
    orchestrator = ClusterAndMineOrchestrator(config=config.mining)
    cluster_artifact = orchestrator.run(payloads)

    cluster_output_dir: str | None = None
    if config.write_cluster_artifact:
        cluster_write = orchestrator.save(cluster_artifact, config.output_root)
        cluster_output_dir = cluster_write.artifact_dir

    decisions = [
        _evaluate_candidate(candidate, cluster_artifact.clusters, config.promotion)
        for candidate in cluster_artifact.operator_candidates
    ]
    promoted_count = sum(1 for decision in decisions if decision.promoted)
    if config.promotion.require_any_promotions and promoted_count <= 0:
        raise PromotionFailure("Promotion gates rejected all candidates")

    write_result = _write_promotion_outputs(
        config=config,
        loaded_sources=loaded_sources,
        cluster_artifact=cluster_artifact,
        cluster_output_dir=cluster_output_dir,
        decisions=decisions,
    )
    _print_summary(write_result=write_result, decisions=decisions, cluster_artifact=cluster_artifact)
    return write_result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine operators and emit manifest-backed promotion candidates")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config for this script")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Input source artifact path. Repeat for multiple sources. Accepts distilled artifact dir/manifest.json or JSONL.",
    )
    parser.add_argument("--output-root", type=str, default=None, help="Artifact root for mining outputs")
    parser.add_argument("--artifact-version", type=str, default=None, help="Artifact version tag")

    parser.add_argument("--min-support", type=int, default=None)
    parser.add_argument("--min-problem-support", type=int, default=None)
    parser.add_argument("--min-symbolic-success-rate", type=float, default=None)
    parser.add_argument("--min-mean-verifier-score", type=float, default=None)
    parser.add_argument("--min-operator-purity", type=float, default=None)
    parser.add_argument("--max-failure-rate", type=float, default=None)
    parser.add_argument("--miner-min-support", type=int, default=None)
    parser.add_argument("--miner-min-symbolic-success-rate", type=float, default=None)
    parser.add_argument("--miner-min-mean-verifier-gain", type=float, default=None)

    parser.add_argument("--promotion-min-support", type=int, default=None)
    parser.add_argument("--promotion-min-symbolic-success-rate", type=float, default=None)
    parser.add_argument("--promotion-min-mean-verifier-gain", type=float, default=None)
    parser.add_argument("--promotion-min-cluster-purity", type=float, default=None)
    parser.add_argument("--promotion-min-cluster-symbolic-success-rate", type=float, default=None)
    parser.add_argument("--promotion-min-cluster-mean-verifier-score", type=float, default=None)
    parser.add_argument("--promotion-max-cluster-failure-rate", type=float, default=None)
    parser.add_argument("--promotion-min-distinct-problem-support", type=int, default=None)

    parser.add_argument("--require-promotions", action="store_true", help="Fail if no operator passes promotion gates")
    parser.add_argument(
        "--no-write-cluster-artifact",
        action="store_true",
        help="Skip writing the intermediate cluster-and-mine artifact",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> MineOperatorsScriptConfig:
    base_payload = _load_structured_config(args.config)

    overrides: dict[str, Any] = {}
    if args.source:
        overrides["source_paths"] = list(args.source)
    if args.output_root is not None:
        overrides["output_root"] = args.output_root
    if args.artifact_version is not None:
        overrides["artifact_version"] = args.artifact_version
    if args.no_write_cluster_artifact:
        overrides["write_cluster_artifact"] = False

    mining_overrides: dict[str, Any] = {}
    for arg_name, field_name in [
        ("min_support", "min_cluster_support"),
        ("min_problem_support", "min_distinct_problem_support"),
        ("min_symbolic_success_rate", "min_symbolic_success_rate"),
        ("min_mean_verifier_score", "min_mean_verifier_score"),
        ("min_operator_purity", "min_operator_purity"),
        ("max_failure_rate", "max_failure_rate"),
        ("miner_min_support", "miner_min_support"),
        ("miner_min_symbolic_success_rate", "miner_min_symbolic_success_rate"),
        ("miner_min_mean_verifier_gain", "miner_min_mean_verifier_gain"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            mining_overrides[field_name] = value
    if mining_overrides:
        overrides["mining"] = mining_overrides

    promotion_overrides: dict[str, Any] = {}
    for arg_name, field_name in [
        ("promotion_min_support", "min_support_count"),
        ("promotion_min_symbolic_success_rate", "min_symbolic_success_rate"),
        ("promotion_min_mean_verifier_gain", "min_mean_verifier_gain"),
        ("promotion_min_cluster_purity", "min_cluster_operator_purity"),
        ("promotion_min_cluster_symbolic_success_rate", "min_cluster_symbolic_success_rate"),
        ("promotion_min_cluster_mean_verifier_score", "min_cluster_mean_verifier_score"),
        ("promotion_max_cluster_failure_rate", "max_cluster_failure_rate"),
        ("promotion_min_distinct_problem_support", "min_distinct_problem_support"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            promotion_overrides[field_name] = value
    if args.require_promotions:
        promotion_overrides["require_any_promotions"] = True
    if promotion_overrides:
        overrides["promotion"] = promotion_overrides

    payload = _deep_merge(base_payload, overrides)
    return MineOperatorsScriptConfig.model_validate(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = _config_from_args(args)
        run_mining(config)
        return 0
    except (FileNotFoundError, SourceValidationError, PromotionFailure, ValidationError, ValueError, TypeError) as exc:
        print(f"[mine_operators] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

from __future__ import annotations

"""
Production CLI entrypoint for verifier training support.

This script is intentionally not a fake launcher. It performs:

- typed config loading
- verifier dataset artifact resolution
- split / manifest validation
- explicit backend availability checks
- real linear-json runtime artifact training when the numeric backend is present
- deterministic output layout and checkpoint manifests
- calibration / export manifest emission
- structured summary metrics

The produced runtime artifact is compatible with the current distilled verifier
runtime, which expects a JSON manifest with backend="linear-json" and per-head
linear weights consumed through a sigmoid link.
"""

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel, ConfigDict, Field

from src.common.constants import ANSWER_MAX, ANSWER_MIN
from src.offline.verifier_distillation import (
    SplitName,
    VerifierTrainingArtifact,
    VerifierTrainingExample,
    load_verifier_training_artifact,
    runtime_verifier_rows,
    stronger_verifier_rows,
)


SCHEMA_VERSION = "train_verifier.v1"
DEFAULT_BACKEND = "numpy-linear-json"
SUPPORTED_BACKENDS = {DEFAULT_BACKEND}
TARGET_HEADS = (
    "probability",
    "logical_consistency",
    "symbolic_consistency",
    "completeness",
    "repairability",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class BackendConfig(StrictModel):
    backend_name: str = DEFAULT_BACKEND
    ridge_lambda: float = Field(default=0.25, ge=0.0)
    target_epsilon: float = Field(default=1e-4, gt=0.0, lt=0.5)
    probability_threshold_grid: tuple[float, ...] = (
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    )


class ExportConfig(StrictModel):
    write_runtime_rows: bool = True
    write_stronger_rows: bool = True
    write_split_row_exports: bool = True
    publish_runtime_aliases: bool = True


class TrainingScriptConfig(StrictModel):
    dataset_path: str
    output_dir: str
    run_name: str = "train_verifier"
    seed: int = 0
    expected_artifact_kind: str | None = None
    backend: BackendConfig = Field(default_factory=BackendConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    require_validation_split: bool = True
    require_calibration_split: bool = True
    overwrite: bool = False


class DatasetValidationReport(StrictModel):
    dataset_path: str
    artifact_id: str
    artifact_kind: str | None = None
    schema_version: str | None = None
    example_count: int
    split_counts_declared: dict[str, int] = Field(default_factory=dict)
    split_counts_actual: dict[str, int] = Field(default_factory=dict)
    retention_counts_actual: dict[str, int] = Field(default_factory=dict)
    scope_counts_actual: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class BackendStatus(StrictModel):
    backend_name: str
    available: bool
    reason: str = ""
    dependency_versions: dict[str, str] = Field(default_factory=dict)


class TrainingMetrics(StrictModel):
    split: str
    example_count: int
    verdict_accuracy: float = 0.0
    answer_accuracy_at_threshold: float = 0.0
    selected_probability_threshold: float = 0.5
    probability_mae: float = 0.0
    probability_rmse: float = 0.0
    probability_brier: float = 0.0
    logical_consistency_mae: float = 0.0
    symbolic_consistency_mae: float = 0.0
    completeness_mae: float = 0.0
    repairability_mae: float = 0.0


class CalibrationManifest(StrictModel):
    selected_probability_threshold: float
    threshold_search_grid: tuple[float, ...]
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    calibration_split_metrics: dict[str, float] = Field(default_factory=dict)


class TrainingSummary(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    backend: BackendStatus
    dataset: DatasetValidationReport
    output_dir: str
    runtime_manifest_path: str | None = None
    distilled_manifest_path: str | None = None
    metrics_by_split: dict[str, TrainingMetrics] = Field(default_factory=dict)
    calibration_manifest_path: str | None = None


@dataclass(frozen=True)
class DerivedExample:
    example: VerifierTrainingExample
    features: dict[str, float]
    targets: dict[str, float]
    binary_answer_target: int
    binary_verdict_target: int


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _stable_hash(prefix: str, payload: Any) -> str:
    return f"{prefix}_{sha1(_stable_json(payload).encode('utf-8')).hexdigest()[:16]}"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clamp01(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        value_f = 0.0
    return max(0.0, min(1.0, value_f))


def _sigmoid_scalar(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_or_yaml_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        loaded = json.loads(text) if text.strip() else {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must decode to an object: {path}")
        return loaded
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"YAML config requested for {path}, but PyYAML is unavailable: {exc}"
            ) from exc
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML config must decode to a mapping: {path}")
        return loaded
    raise ValueError(f"Unsupported config file extension: {path.suffix}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train verifier support artifacts from a verifier dataset artifact.")
    parser.add_argument("--dataset-path", required=True, help="Path to verifier dataset artifact dir or manifest.json.")
    parser.add_argument("--output-dir", required=True, help="Explicit output root directory.")
    parser.add_argument("--config", default=None, help="Optional JSON/YAML config file.")
    parser.add_argument("--run-name", default="train_verifier", help="Human-readable run prefix.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed included in artifact naming.")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Training backend. Supported: numpy-linear-json.")
    parser.add_argument("--ridge-lambda", type=float, default=None, help="Optional ridge regularization override.")
    parser.add_argument(
        "--expected-artifact-kind",
        default=None,
        help="Optional expected dataset artifact kind; mismatches fail validation.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing deterministic run directory.")
    parser.add_argument(
        "--allow-missing-validation-split",
        action="store_true",
        help="Do not fail if the valid split is empty.",
    )
    parser.add_argument(
        "--allow-missing-calibration-split",
        action="store_true",
        help="Do not fail if the calibration split is empty.",
    )
    parser.add_argument(
        "--no-export-runtime-rows",
        action="store_true",
        help="Skip runtime-row JSONL export.",
    )
    parser.add_argument(
        "--no-export-stronger-rows",
        action="store_true",
        help="Skip stronger-row JSONL export.",
    )
    return parser.parse_args(argv)


def _resolve_config(args: argparse.Namespace) -> TrainingScriptConfig:
    payload: dict[str, Any] = {}
    if args.config:
        payload.update(_parse_json_or_yaml_file(Path(args.config).resolve()))

    payload["dataset_path"] = str(Path(args.dataset_path).resolve())
    payload["output_dir"] = str(Path(args.output_dir).resolve())
    payload["run_name"] = args.run_name
    payload["seed"] = int(args.seed)
    payload["overwrite"] = bool(args.overwrite)

    payload.setdefault("backend", {})
    payload["backend"]["backend_name"] = args.backend
    if args.ridge_lambda is not None:
        payload["backend"]["ridge_lambda"] = float(args.ridge_lambda)

    if args.expected_artifact_kind is not None:
        payload["expected_artifact_kind"] = args.expected_artifact_kind

    payload.setdefault("export", {})
    if args.no_export_runtime_rows:
        payload["export"]["write_runtime_rows"] = False
    if args.no_export_stronger_rows:
        payload["export"]["write_stronger_rows"] = False

    payload["require_validation_split"] = not bool(args.allow_missing_validation_split)
    payload["require_calibration_split"] = not bool(args.allow_missing_calibration_split)
    return TrainingScriptConfig.model_validate(payload)


def _artifact_input_root(dataset_path: Path) -> Path:
    if dataset_path.is_dir():
        return dataset_path
    if dataset_path.name == "manifest.json":
        return dataset_path.parent
    raise ValueError(
        f"Unsupported dataset path `{dataset_path}`. Expected an artifact directory or manifest.json."
    )


def _validate_dataset_artifact(
    config: TrainingScriptConfig,
    artifact: VerifierTrainingArtifact,
    dataset_root: Path,
) -> DatasetValidationReport:
    metadata = artifact.metadata
    if config.expected_artifact_kind is not None:
        actual_kind = getattr(metadata, "artifact_kind", None)
        if actual_kind != config.expected_artifact_kind:
            raise ValueError(
                f"Dataset artifact_kind mismatch: expected `{config.expected_artifact_kind}`, got `{actual_kind}`."
            )

    if int(metadata.example_count) != len(artifact.examples):
        raise ValueError(
            f"Dataset manifest example_count={metadata.example_count} does not match loaded examples={len(artifact.examples)}."
        )

    actual_split_counts: dict[str, int] = {}
    actual_retention_counts: dict[str, int] = {}
    actual_scope_counts: dict[str, int] = {}

    for example in artifact.examples:
        split = example.split.value
        actual_split_counts[split] = actual_split_counts.get(split, 0) + 1
        actual_retention_counts[example.retention_tag.value] = actual_retention_counts.get(example.retention_tag.value, 0) + 1
        actual_scope_counts[example.scope.value] = actual_scope_counts.get(example.scope.value, 0) + 1

    declared_split_counts = dict(getattr(metadata, "split_counts", {}) or {})
    if declared_split_counts != actual_split_counts:
        raise ValueError(
            "Dataset split count mismatch between manifest and loaded examples: "
            f"declared={declared_split_counts}, actual={actual_split_counts}."
        )

    for split_name in (SplitName.TRAIN.value, SplitName.VALID.value, SplitName.CALIBRATION.value):
        split_file = dataset_root / f"{split_name}.jsonl"
        declared = actual_split_counts.get(split_name, 0)
        if declared > 0 and not split_file.exists():
            raise ValueError(f"Declared non-empty split `{split_name}` is missing expected file: {split_file}")

    if actual_split_counts.get(SplitName.TRAIN.value, 0) <= 0:
        raise ValueError("Verifier dataset has no training examples.")
    if config.require_validation_split and actual_split_counts.get(SplitName.VALID.value, 0) <= 0:
        raise ValueError("Verifier dataset has no validation examples, and validation is required.")
    if config.require_calibration_split and actual_split_counts.get(SplitName.CALIBRATION.value, 0) <= 0:
        raise ValueError("Verifier dataset has no calibration examples, and calibration is required.")

    warnings: list[str] = []
    if actual_scope_counts.get("prefix", 0) == 0:
        warnings.append("No prefix-scope verifier examples were found.")
    if actual_retention_counts.get("plausible_wrong", 0) == 0:
        warnings.append("No plausible-wrong examples were found; verifier robustness may be weak.")

    return DatasetValidationReport(
        dataset_path=str(dataset_root),
        artifact_id=metadata.artifact_id,
        artifact_kind=getattr(metadata, "artifact_kind", None),
        schema_version=getattr(metadata, "schema_version", None),
        example_count=len(artifact.examples),
        split_counts_declared=declared_split_counts,
        split_counts_actual=actual_split_counts,
        retention_counts_actual=actual_retention_counts,
        scope_counts_actual=actual_scope_counts,
        warnings=warnings,
    )


def _backend_status(config: TrainingScriptConfig) -> BackendStatus:
    backend_name = config.backend.backend_name.strip().lower()
    if backend_name not in SUPPORTED_BACKENDS:
        return BackendStatus(
            backend_name=backend_name,
            available=False,
            reason=f"Unsupported backend `{backend_name}`. Supported backends: {sorted(SUPPORTED_BACKENDS)}.",
        )
    try:
        import numpy as np  # type: ignore

        version = getattr(np, "__version__", "unknown")
        return BackendStatus(
            backend_name=backend_name,
            available=True,
            reason="ready",
            dependency_versions={"numpy": str(version)},
        )
    except Exception as exc:  # pragma: no cover
        return BackendStatus(
            backend_name=backend_name,
            available=False,
            reason=f"NumPy backend unavailable: {exc}",
        )


def _phase_progress(branch_phase: str | None) -> float:
    mapping = {
        "initialized": 0.05,
        "retrieved": 0.18,
        "reasoning": 0.45,
        "verified": 0.70,
        "critiqued": 0.82,
        "repaired": 0.62,
        "rolled_back": 0.40,
        "solved": 1.00,
        "failed": 0.10,
        "pruned": 0.12,
        "trace": 0.45,
    }
    return _clamp01(mapping.get(_normalize_text(branch_phase).lower(), 0.0))


def _extract_route_certainty(bundle_context: Mapping[str, Any]) -> float:
    labels = bundle_context.get("route_problem_type") or []
    if not isinstance(labels, Sequence):
        return 0.0
    best = 0.0
    for item in labels:
        if isinstance(item, Mapping):
            best = max(best, _clamp01(item.get("weight", 0.0)))
    return best


def _extract_prior_confidence(bundle_context: Mapping[str, Any]) -> float:
    labels = bundle_context.get("route_archetypes") or []
    if not isinstance(labels, Sequence):
        return 0.0
    best = 0.0
    for item in labels:
        if isinstance(item, Mapping):
            best = max(best, _clamp01(item.get("weight", 0.0)))
    return best


def _answer_in_range(canonical_answer: str) -> bool:
    text = _normalize_text(canonical_answer)
    if not text:
        return False
    try:
        value = int(text)
    except Exception:
        return False
    return ANSWER_MIN <= value <= ANSWER_MAX


def _bundle_context(example: VerifierTrainingExample) -> Mapping[str, Any]:
    return (example.labels.bundle or {}).get("context", {})


def _bundle_consistency(example: VerifierTrainingExample) -> Mapping[str, Any]:
    return (example.labels.bundle or {}).get("branch_label", {}).get("consistency", {})


def _derive_features(example: VerifierTrainingExample) -> dict[str, float]:
    ctx = _bundle_context(example)
    consistency = _bundle_consistency(example)
    prefix_steps = max(0, int(example.prefix_step_count))
    total_steps = max(prefix_steps, int(example.total_step_count))
    step_density = 0.0 if total_steps <= 0 else min(1.0, float(prefix_steps) / float(total_steps))

    answer_present = 1.0 if _normalize_text(example.texts.answer_text or example.texts.canonical_answer) else 0.0
    reasoning_present = 1.0 if _normalize_text(example.texts.prefix_text or example.texts.reasoning_text) else 0.0
    answer_in_range = 1.0 if _answer_in_range(example.texts.canonical_answer) else 0.0
    route_certainty = _extract_route_certainty(ctx)
    prior_confidence = _extract_prior_confidence(ctx)
    retrieval_used = 1.0 if bool(ctx.get("retrieval_used", False)) else 0.0
    repair_pressure = _clamp01(_safe_float(ctx.get("repair_count", 0)) / 3.0)
    verifier_pressure = _clamp01(_safe_float(ctx.get("verifier_evidence_count", 0)) / 4.0)
    symbolic_score = _clamp01(example.labels.model_targets.get("symbolic_agreement", 0.0))
    contradiction_flag = 1.0 if bool(consistency.get("contradiction_detected", False)) else 0.0

    exact_symbolic = 1.0 if (symbolic_score >= 0.95 and answer_in_range > 0.0) else 0.0
    retrieval_answer_support = retrieval_used * _clamp01(example.labels.model_targets.get("answer_correct_likelihood", 0.0))
    candidate_confidence = _clamp01(example.labels.model_targets.get("verdict_confidence", 0.0))
    generation_confidence = candidate_confidence

    return {
        "answer_present": answer_present,
        "answer_in_range": answer_in_range,
        "reasoning_present": reasoning_present,
        "step_density": step_density,
        "constraint_coverage": 0.0,
        "invariant_coverage": 0.0,
        "goal_coverage": 0.0,
        "candidate_confidence": candidate_confidence,
        "generation_confidence": generation_confidence,
        "symbolic_score": symbolic_score,
        "symbolic_pass": 1.0 if symbolic_score >= 0.50 else 0.0,
        "symbolic_exact": exact_symbolic,
        "retrieval_support": retrieval_used,
        "retrieval_answer_support": retrieval_answer_support,
        "route_certainty": route_certainty,
        "prior_confidence": prior_confidence,
        "phase_progress": _phase_progress(ctx.get("branch_phase")),
        "repair_pressure": repair_pressure,
        "verifier_pressure": verifier_pressure,
        "critique_pressure": 0.0,
        "existing_verifier_probability": candidate_confidence,
        "contradiction_flag": contradiction_flag,
    }


def _derive_targets(example: VerifierTrainingExample) -> dict[str, float]:
    targets = dict(example.labels.model_targets)
    return {
        "probability": _clamp01(targets.get("answer_correct_likelihood", 0.0)),
        "logical_consistency": _clamp01(targets.get("logical_consistency", 0.0)),
        "symbolic_consistency": _clamp01(targets.get("symbolic_agreement", 0.0)),
        "completeness": _clamp01(targets.get("completeness", 0.0)),
        "repairability": _clamp01(targets.get("repairability", 0.0)),
    }


def _binary_answer_target(example: VerifierTrainingExample) -> int:
    return 1 if _clamp01(example.labels.model_targets.get("answer_correct_likelihood", 0.0)) >= 0.5 else 0


def _binary_verdict_target(example: VerifierTrainingExample) -> int:
    verdict = _normalize_text(example.verdict.value).lower()
    return 1 if verdict in {"accept", "accept_with_reservations"} else 0


def _derive_examples(artifact: VerifierTrainingArtifact) -> list[DerivedExample]:
    derived: list[DerivedExample] = []
    for example in artifact.examples:
        derived.append(
            DerivedExample(
                example=example,
                features=_derive_features(example),
                targets=_derive_targets(example),
                binary_answer_target=_binary_answer_target(example),
                binary_verdict_target=_binary_verdict_target(example),
            )
        )
    return derived


def _group_by_split(items: Sequence[DerivedExample]) -> dict[str, list[DerivedExample]]:
    grouped = {
        SplitName.TRAIN.value: [],
        SplitName.VALID.value: [],
        SplitName.CALIBRATION.value: [],
    }
    for item in items:
        grouped[item.example.split.value].append(item)
    return grouped


def _prepare_run_dir(config: TrainingScriptConfig, dataset_report: DatasetValidationReport) -> Path:
    payload = {
        "run_name": config.run_name,
        "seed": config.seed,
        "dataset_artifact_id": dataset_report.artifact_id,
        "backend_name": config.backend.backend_name,
        "ridge_lambda": config.backend.ridge_lambda,
    }
    run_id = _stable_hash(config.run_name.replace(" ", "_"), payload)
    run_dir = Path(config.output_dir) / run_id
    if run_dir.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"Deterministic verifier output directory already exists: {run_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(dict(row)) + "\n")


def _export_rows(
    run_dir: Path,
    artifact: VerifierTrainingArtifact,
    *,
    config: TrainingScriptConfig,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    runtime_rows = runtime_verifier_rows(artifact)
    stronger_rows = stronger_verifier_rows(artifact)

    if config.export.write_runtime_rows:
        path = run_dir / "exports" / "runtime_rows.jsonl"
        _write_jsonl(path, runtime_rows)
        outputs["runtime_rows"] = str(path)

    if config.export.write_stronger_rows:
        path = run_dir / "exports" / "stronger_rows.jsonl"
        _write_jsonl(path, stronger_rows)
        outputs["stronger_rows"] = str(path)

    if config.export.write_split_row_exports:
        for split_name in (SplitName.TRAIN.value, SplitName.VALID.value, SplitName.CALIBRATION.value):
            if config.export.write_runtime_rows:
                split_runtime = [row for row in runtime_rows if row.get("split") == split_name]
                path = run_dir / "exports" / f"{split_name}_runtime_rows.jsonl"
                _write_jsonl(path, split_runtime)
                outputs[f"{split_name}_runtime_rows"] = str(path)
            if config.export.write_stronger_rows:
                split_stronger = [row for row in stronger_rows if row.get("split") == split_name]
                path = run_dir / "exports" / f"{split_name}_stronger_rows.jsonl"
                _write_jsonl(path, split_stronger)
                outputs[f"{split_name}_stronger_rows"] = str(path)

    return outputs


def _logit_clip(y: float, eps: float) -> float:
    clipped = max(eps, min(1.0 - eps, float(y)))
    return math.log(clipped / (1.0 - clipped))


def _fit_linear_head(
    feature_rows: Sequence[dict[str, float]],
    targets: Sequence[float],
    *,
    feature_order: Sequence[str],
    ridge_lambda: float,
    epsilon: float,
) -> tuple[dict[str, float], float]:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"NumPy backend unavailable during training: {exc}") from exc

    if not feature_rows:
        raise ValueError("Cannot fit a verifier head with zero training examples.")

    x = np.asarray(
        [[float(row.get(feature_name, 0.0)) for feature_name in feature_order] for row in feature_rows],
        dtype=float,
    )
    z = np.asarray([_logit_clip(float(target), epsilon) for target in targets], dtype=float)

    ones = np.ones((x.shape[0], 1), dtype=float)
    design = np.concatenate([ones, x], axis=1)
    reg = np.eye(design.shape[1], dtype=float) * float(ridge_lambda)
    reg[0, 0] = 0.0

    lhs = design.T @ design + reg
    rhs = design.T @ z
    params = np.linalg.solve(lhs, rhs)

    bias = float(params[0])
    weights = {feature_name: float(params[idx + 1]) for idx, feature_name in enumerate(feature_order)}
    return weights, bias


def _predict_head(features: Mapping[str, float], weights: Mapping[str, float], bias: float) -> float:
    linear = float(bias)
    for feature_name, weight in weights.items():
        linear += float(weight) * float(features.get(feature_name, 0.0))
    return _sigmoid_scalar(linear)


def _fit_runtime_manifest(
    config: TrainingScriptConfig,
    train_rows: Sequence[DerivedExample],
    *,
    feature_order: Sequence[str],
) -> dict[str, Any]:
    heads: dict[str, dict[str, Any]] = {}
    for head_name in TARGET_HEADS:
        targets = [row.targets[head_name] for row in train_rows]
        weights, bias = _fit_linear_head(
            [row.features for row in train_rows],
            targets,
            feature_order=feature_order,
            ridge_lambda=config.backend.ridge_lambda,
            epsilon=config.backend.target_epsilon,
        )
        heads[head_name] = {
            "weights": weights,
            "bias": bias,
            "scale": 1.0,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "linear-json",
        "backend_origin": config.backend.backend_name,
        "feature_order": list(feature_order),
        "heads": heads,
    }


def _metrics_for_rows(
    rows: Sequence[DerivedExample],
    runtime_manifest: Mapping[str, Any],
    *,
    probability_threshold: float,
    split_name: str,
) -> TrainingMetrics:
    heads = runtime_manifest.get("heads", {})
    predicted: list[dict[str, float]] = []
    for row in rows:
        pred = {
            head_name: _predict_head(
                row.features,
                (heads.get(head_name) or {}).get("weights", {}),
                float((heads.get(head_name) or {}).get("bias", 0.0)),
            )
            for head_name in TARGET_HEADS
        }
        predicted.append(pred)

    if not rows:
        return TrainingMetrics(split=split_name, example_count=0, selected_probability_threshold=probability_threshold)

    def mae(key: str) -> float:
        return sum(abs(pred[key] - row.targets[key]) for pred, row in zip(predicted, rows)) / len(rows)

    def rmse(key: str) -> float:
        return math.sqrt(sum((pred[key] - row.targets[key]) ** 2 for pred, row in zip(predicted, rows)) / len(rows))

    verdict_correct = 0
    answer_correct = 0
    brier = 0.0
    for pred, row in zip(predicted, rows):
        verdict_hat = 1 if pred["probability"] >= probability_threshold else 0
        verdict_correct += int(verdict_hat == row.binary_verdict_target)
        answer_correct += int(verdict_hat == row.binary_answer_target)
        brier += (pred["probability"] - float(row.binary_answer_target)) ** 2

    return TrainingMetrics(
        split=split_name,
        example_count=len(rows),
        verdict_accuracy=float(verdict_correct) / float(len(rows)),
        answer_accuracy_at_threshold=float(answer_correct) / float(len(rows)),
        selected_probability_threshold=probability_threshold,
        probability_mae=mae("probability"),
        probability_rmse=rmse("probability"),
        probability_brier=brier / float(len(rows)),
        logical_consistency_mae=mae("logical_consistency"),
        symbolic_consistency_mae=mae("symbolic_consistency"),
        completeness_mae=mae("completeness"),
        repairability_mae=mae("repairability"),
    )


def _select_probability_threshold(
    valid_rows: Sequence[DerivedExample],
    runtime_manifest: Mapping[str, Any],
    grid: Sequence[float],
) -> tuple[float, dict[str, float]]:
    if not valid_rows:
        threshold = 0.5
        return threshold, {"answer_accuracy_at_threshold": 0.0, "verdict_accuracy": 0.0}

    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    for threshold in grid:
        metrics = _metrics_for_rows(valid_rows, runtime_manifest, probability_threshold=float(threshold), split_name="valid")
        score = metrics.answer_accuracy_at_threshold
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                "answer_accuracy_at_threshold": metrics.answer_accuracy_at_threshold,
                "verdict_accuracy": metrics.verdict_accuracy,
                "probability_brier": metrics.probability_brier,
            }
    return best_threshold, best_metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _resolve_config(args)

    dataset_input = Path(config.dataset_path).resolve()
    dataset_root = _artifact_input_root(dataset_input)
    artifact = load_verifier_training_artifact(dataset_input)
    dataset_report = _validate_dataset_artifact(config, artifact, dataset_root)

    run_dir = _prepare_run_dir(config, dataset_report)
    _write_json(run_dir / "config_snapshot.json", config.model_dump(mode="json"))
    _write_json(run_dir / "dataset_manifest_snapshot.json", artifact.metadata.model_dump(mode="json"))
    _write_json(run_dir / "dataset_validation_report.json", dataset_report.model_dump(mode="json"))

    backend = _backend_status(config)
    _write_json(run_dir / "backend_status.json", backend.model_dump(mode="json"))
    if not backend.available:
        summary = TrainingSummary(
            run_id=run_dir.name,
            backend=backend,
            dataset=dataset_report,
            output_dir=str(run_dir),
        )
        _write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
        print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False))
        return 2

    row_exports = _export_rows(run_dir, artifact, config=config)
    derived = _derive_examples(artifact)
    by_split = _group_by_split(derived)

    train_rows = by_split[SplitName.TRAIN.value]
    valid_rows = by_split[SplitName.VALID.value]
    calibration_rows = by_split[SplitName.CALIBRATION.value]

    feature_order = tuple(sorted({key for row in train_rows for key in row.features.keys()}))
    runtime_manifest = _fit_runtime_manifest(config, train_rows, feature_order=feature_order)

    selected_threshold, valid_threshold_metrics = _select_probability_threshold(
        valid_rows,
        runtime_manifest,
        config.backend.probability_threshold_grid,
    )

    metrics_by_split = {
        SplitName.TRAIN.value: _metrics_for_rows(
            train_rows,
            runtime_manifest,
            probability_threshold=selected_threshold,
            split_name=SplitName.TRAIN.value,
        ),
        SplitName.VALID.value: _metrics_for_rows(
            valid_rows,
            runtime_manifest,
            probability_threshold=selected_threshold,
            split_name=SplitName.VALID.value,
        ),
        SplitName.CALIBRATION.value: _metrics_for_rows(
            calibration_rows,
            runtime_manifest,
            probability_threshold=selected_threshold,
            split_name=SplitName.CALIBRATION.value,
        ),
    }

    calibration = CalibrationManifest(
        selected_probability_threshold=selected_threshold,
        threshold_search_grid=tuple(float(x) for x in config.backend.probability_threshold_grid),
        validation_metrics=valid_threshold_metrics,
        calibration_split_metrics=metrics_by_split[SplitName.CALIBRATION.value].model_dump(mode="json"),
    )

    runtime_manifest.update(
        {
            "run_id": run_dir.name,
            "dataset_artifact_id": dataset_report.artifact_id,
            "dataset_manifest_path": str(run_dir / "dataset_manifest_snapshot.json"),
            "training_summary_path": str(run_dir / "summary.json"),
            "calibration_manifest_path": str(run_dir / "calibration.json"),
            "selected_probability_threshold": selected_threshold,
            "metrics_by_split": {
                split: metrics.model_dump(mode="json") for split, metrics in metrics_by_split.items()
            },
            "row_exports": row_exports,
        }
    )

    runtime_manifest_path = run_dir / "runtime_model.json"
    distilled_manifest_path = run_dir / "distilled_manifest.json"
    _write_json(runtime_manifest_path, runtime_manifest)
    _write_json(distilled_manifest_path, runtime_manifest)

    if config.export.publish_runtime_aliases:
        _write_json(run_dir / "manifest.json", runtime_manifest)
        _write_json(run_dir / "model.json", runtime_manifest)

    _write_json(run_dir / "calibration.json", calibration.model_dump(mode="json"))
    _write_json(
        run_dir / "training_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "backend": backend.model_dump(mode="json"),
            "dataset": dataset_report.model_dump(mode="json"),
            "runtime_manifest_path": str(runtime_manifest_path),
            "distilled_manifest_path": str(distilled_manifest_path),
            "feature_order": list(feature_order),
            "row_exports": row_exports,
        },
    )

    summary = TrainingSummary(
        run_id=run_dir.name,
        backend=backend,
        dataset=dataset_report,
        output_dir=str(run_dir),
        runtime_manifest_path=str(runtime_manifest_path),
        distilled_manifest_path=str(distilled_manifest_path),
        metrics_by_split=metrics_by_split,
        calibration_manifest_path=str(run_dir / "calibration.json"),
    )
    _write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

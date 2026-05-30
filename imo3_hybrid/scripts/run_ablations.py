from __future__ import annotations

"""Manifest-backed controlled ablation runner.

This script is a production CLI entrypoint for deterministic ablation studies.
It resolves a named experiment from an ablation registry (or inline CLI dimensions),
materializes each variant as an explicit config diff, optionally executes a
runner command for each variant, ingests structured result envelopes, and writes
fully auditable manifests plus deterministic aggregate summaries.

Repair emphasis for this pass:
- real deterministic offline evaluation tool, not notebook-style shell behavior
- explicit split-safe comparisons
- manifest/provenance logging for every variant and the study as a whole
- bounded, reproducible aggregate summaries over meaningful comparison axes
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.constants import OFFLINE_ARTIFACT_VERSION

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


SCHEMA_VERSION = "run_ablations.v2"
ARTIFACT_KIND = "ablation_study"
VARIANT_ARTIFACT_KIND = "ablation_variant_run"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class RunStatus(str):
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SplitMetricSummary(StrictModel):
    split: str
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AblationDimensionSpec(StrictModel):
    name: str
    path: str
    values: tuple[Any, ...] = Field(default_factory=tuple)
    description: str = ""

    @model_validator(mode="after")
    def _validate_values(self) -> "AblationDimensionSpec":
        if not self.values:
            raise ValueError(f"Ablation dimension '{self.name}' must define at least one value")
        return self


class AblationExperimentSpec(StrictModel):
    name: str
    description: str = ""
    include_baseline: bool = True
    tags: tuple[str, ...] = Field(default_factory=tuple)
    base_overrides: dict[str, Any] = Field(default_factory=dict)
    dimensions: tuple[AblationDimensionSpec, ...] = Field(default_factory=tuple)
    fixed_dimensions: dict[str, Any] = Field(default_factory=dict)
    runner_command: str | None = None
    result_filename: str = "result.json"
    stop_on_failure: bool = False


class AblationStudyConfig(StrictModel):
    run_name: str = "ablation_study"
    output_dir: str = "artifacts/ablations"
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    base_config_path: str | None = None
    base_config: dict[str, Any] = Field(default_factory=dict)
    experiments: tuple[AblationExperimentSpec, ...] = Field(default_factory=tuple)
    selected_experiment: str | None = None
    runner_command: str | None = None
    result_filename: str = "result.json"
    plan_only: bool = False
    stop_on_failure: bool = False
    emit_resolved_configs: bool = True
    overwrite: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class VariantDimensionValue(StrictModel):
    name: str
    path: str
    value: Any
    description: str = ""


class ReproducibilityMetadata(StrictModel):
    python_version: str
    platform: str
    cwd: str
    script_path: str
    config_path: str | None = None
    base_config_path: str | None = None
    selected_experiment: str
    timestamp_utc: str
    environment: dict[str, str] = Field(default_factory=dict)


class ResultEnvelope(StrictModel):
    status: str = RunStatus.COMPLETED
    metrics: dict[str, float] = Field(default_factory=dict)
    split_metrics: tuple[SplitMetricSummary, ...] = Field(default_factory=tuple)
    artifacts: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VariantManifest(StrictModel):
    artifact_kind: str = VARIANT_ARTIFACT_KIND
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    study_run_id: str
    experiment_name: str
    variant_id: str
    variant_name: str
    variant_index: int = Field(ge=0)
    baseline: bool = False
    dimensions: tuple[VariantDimensionValue, ...] = Field(default_factory=tuple)
    fixed_dimensions: dict[str, Any] = Field(default_factory=dict)
    comparison_axes: tuple[str, ...] = Field(default_factory=tuple)
    config_digest: str
    base_config_digest: str
    config_diff: dict[str, dict[str, Any]] = Field(default_factory=dict)
    resolved_config_path: str
    result_path: str
    runner_command: str | None = None
    status: str = RunStatus.PLANNED
    failure: dict[str, Any] | None = None
    result: ResultEnvelope | None = None
    reproducibility: ReproducibilityMetadata
    notes: list[str] = Field(default_factory=list)


class StudyManifest(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    run_id: str
    run_name: str
    experiment_name: str
    config_digest: str
    experiment_digest: str
    base_config_digest: str
    variant_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    planned_only: bool = False
    reproducibility: ReproducibilityMetadata
    aggregate_summary_path: str
    results_jsonl_path: str
    registry_snapshot_path: str
    variant_manifest_paths: tuple[str, ...] = Field(default_factory=tuple)
    metric_summary: dict[str, Any] = Field(default_factory=dict)
    split_metric_summary: dict[str, Any] = Field(default_factory=dict)
    dimension_summary: dict[str, Any] = Field(default_factory=dict)
    failure_summary: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PlannedVariant:
    variant_id: str
    variant_name: str
    baseline: bool
    dimensions: tuple[VariantDimensionValue, ...]
    fixed_dimensions: dict[str, Any]
    resolved_config: dict[str, Any]
    config_diff: dict[str, dict[str, Any]]
    config_digest: str
    comparison_axes: tuple[str, ...]


class AblationFailure(RuntimeError):
    pass


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(prefix: str, payload: Any) -> str:
    token = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{hashlib.sha1(token.encode('utf-8')).hexdigest()[:16]}"


def _load_config_payload(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = target.read_text(encoding="utf-8")
    suffix = target.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw or "{}")
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML config requested but PyYAML is not installed")
        payload = yaml.safe_load(raw) or {}
    else:
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            if yaml is None:
                raise RuntimeError(f"Unsupported config extension: {path}")
            payload = yaml.safe_load(raw) or {}
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object-like config payload in {path}")
    return payload


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_scalar(text: str) -> Any:
    candidate = text.strip()
    if candidate == "":
        return ""
    lowered = candidate.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        pass
    try:
        if any(ch in candidate for ch in (".", "e", "E")):
            return float(candidate)
        return int(candidate)
    except Exception:
        return candidate


def _set_dotted_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("Empty dotted path is not allowed")
    cursor = payload
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            cursor[part] = {}
            current = cursor[part]
        if not isinstance(current, dict):
            raise ValueError(f"Cannot descend through non-object path segment '{part}' in '{path}'")
        cursor = current
    cursor[parts[-1]] = deepcopy(value)


def _flatten_dict(payload: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(payload.keys()):
        value = payload[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_dict(dict(value), prefix=path))
        else:
            out[path] = value
    return out


def _compute_config_diff(base_config: Mapping[str, Any], variant_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    base_flat = _flatten_dict(base_config)
    variant_flat = _flatten_dict(variant_config)
    keys = sorted(set(base_flat) | set(variant_flat))
    diff: dict[str, dict[str, Any]] = {}
    for key in keys:
        base_value = base_flat.get(key)
        variant_value = variant_flat.get(key)
        if base_value != variant_value:
            diff[key] = {"base": base_value, "variant": variant_value}
    return diff


def _slugify(text: str) -> str:
    safe = [ch.lower() if ch.isalnum() else "_" for ch in text]
    slug = "".join(safe)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "variant"


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must be path=value, got: {text}")
    path, raw_value = text.split("=", 1)
    path = path.strip()
    if not path:
        raise ValueError(f"Override missing path: {text}")
    return path, _parse_scalar(raw_value)


def _parse_dimension(text: str) -> AblationDimensionSpec:
    if "=" not in text or ":" not in text:
        raise ValueError("Dimension must use format name=path:value1,value2,... ; received: " + text)
    name, rhs = text.split("=", 1)
    path, values_blob = rhs.split(":", 1)
    values = tuple(_parse_scalar(part) for part in values_blob.split(",") if part != "")
    return AblationDimensionSpec(name=name.strip(), path=path.strip(), values=values)


def _environment_snapshot() -> dict[str, str]:
    keys = [
        "PYTHONHASHSEED",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _build_repro_metadata(
    *,
    config_path: str | None,
    base_config_path: str | None,
    selected_experiment: str,
) -> ReproducibilityMetadata:
    return ReproducibilityMetadata(
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        cwd=str(Path.cwd()),
        script_path=str(Path(__file__).resolve()),
        config_path=config_path,
        base_config_path=base_config_path,
        selected_experiment=selected_experiment,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        environment=_environment_snapshot(),
    )


def _resolve_study_config(args: argparse.Namespace) -> tuple[AblationStudyConfig, str | None]:
    loaded_payload = _load_config_payload(args.config)

    if args.base_config is not None:
        loaded_payload["base_config_path"] = args.base_config
    if args.output_dir is not None:
        loaded_payload["output_dir"] = args.output_dir
    if args.run_name is not None:
        loaded_payload["run_name"] = args.run_name
    if args.experiment is not None:
        loaded_payload["selected_experiment"] = args.experiment
    if args.runner_command is not None:
        loaded_payload["runner_command"] = args.runner_command
    if args.result_filename is not None:
        loaded_payload["result_filename"] = args.result_filename
    if args.plan_only:
        loaded_payload["plan_only"] = True
    if args.stop_on_failure:
        loaded_payload["stop_on_failure"] = True
    if args.overwrite:
        loaded_payload["overwrite"] = True

    if args.set:
        base_cfg = dict(loaded_payload.get("base_config") or {})
        for item in args.set:
            path, value = _parse_override(item)
            _set_dotted_path(base_cfg, path, value)
        loaded_payload["base_config"] = base_cfg

    if args.dimension:
        inline_experiment_name = args.experiment or "cli_inline"
        inline_dims = tuple(_parse_dimension(item) for item in args.dimension)
        inline_spec = AblationExperimentSpec(
            name=inline_experiment_name,
            description="CLI-defined ablation experiment",
            dimensions=inline_dims,
            include_baseline=not args.no_baseline,
            runner_command=args.runner_command,
            result_filename=args.result_filename or loaded_payload.get("result_filename", "result.json"),
            stop_on_failure=args.stop_on_failure,
        )
        loaded_payload["experiments"] = [
            item for item in loaded_payload.get("experiments", []) if item.get("name") != inline_experiment_name
        ] + [inline_spec.model_dump(mode="json")]
        loaded_payload["selected_experiment"] = inline_experiment_name

    config = AblationStudyConfig.model_validate(loaded_payload)
    return config, args.config


def _select_experiment(config: AblationStudyConfig) -> AblationExperimentSpec:
    if not config.experiments:
        raise AblationFailure(
            "No experiments are defined. Provide configs/ablations.yaml or use --dimension to define an inline study."
        )
    selected = config.selected_experiment or config.experiments[0].name
    for experiment in config.experiments:
        if experiment.name == selected:
            return experiment
    raise AblationFailure(f"Experiment '{selected}' not found in ablation registry")


def _load_base_config(config: AblationStudyConfig) -> dict[str, Any]:
    base_payload = _load_config_payload(config.base_config_path)
    return _deep_merge(base_payload, config.base_config)


def _variant_name_from_dimensions(dimensions: Sequence[VariantDimensionValue], baseline: bool) -> str:
    if baseline:
        return "baseline"
    pieces = [f"{item.name}={item.value}" for item in dimensions]
    return _slugify("__".join(pieces))


def _comparison_axes(
    dimensions: Sequence[VariantDimensionValue],
    config_diff: Mapping[str, Any],
) -> tuple[str, ...]:
    axes = [dimension.name for dimension in dimensions]
    # Add conservative semantic hints from config paths for easier audit.
    for key in sorted(config_diff.keys()):
        lowered = key.lower()
        if "retrieval" in lowered:
            axes.append("retrieval")
        if "operator" in lowered:
            axes.append("operator")
        if "prm" in lowered or "process" in lowered:
            axes.append("prm")
        if "verifier" in lowered:
            axes.append("verifier")
        if "budget" in lowered or "compute" in lowered or "difficulty" in lowered:
            axes.append("compute_economy")
    dedup: list[str] = []
    seen: set[str] = set()
    for item in axes:
        if item not in seen:
            dedup.append(item)
            seen.add(item)
    return tuple(dedup)


def _plan_variants(
    *,
    experiment: AblationExperimentSpec,
    base_config: dict[str, Any],
) -> list[PlannedVariant]:
    experiment_base = _deep_merge(base_config, experiment.base_overrides)
    dimensions = list(experiment.dimensions)
    variants: list[PlannedVariant] = []

    if experiment.include_baseline:
        config_digest = _stable_hash("ablation_config", experiment_base)
        variants.append(
            PlannedVariant(
                variant_id=_stable_hash(
                    "ablation_variant",
                    {
                        "experiment": experiment.name,
                        "baseline": True,
                        "config_digest": config_digest,
                    },
                ),
                variant_name="baseline",
                baseline=True,
                dimensions=tuple(),
                fixed_dimensions=dict(sorted(experiment.fixed_dimensions.items())),
                resolved_config=experiment_base,
                config_diff={},
                config_digest=config_digest,
                comparison_axes=tuple(),
            )
        )

    if dimensions:
        names = [dimension.name for dimension in dimensions]
        if len(set(names)) != len(names):
            raise AblationFailure(f"Duplicate ablation dimension names in experiment '{experiment.name}'")
        for combo in product(*[dimension.values for dimension in dimensions]):
            resolved = deepcopy(experiment_base)
            dim_values: list[VariantDimensionValue] = []
            for dimension, value in zip(dimensions, combo):
                _set_dotted_path(resolved, dimension.path, value)
                dim_values.append(
                    VariantDimensionValue(
                        name=dimension.name,
                        path=dimension.path,
                        value=deepcopy(value),
                        description=dimension.description,
                    )
                )
            config_diff = _compute_config_diff(experiment_base, resolved)
            config_digest = _stable_hash("ablation_config", resolved)
            variant_name = _variant_name_from_dimensions(dim_values, False)
            variant_id = _stable_hash(
                "ablation_variant",
                {
                    "experiment": experiment.name,
                    "variant_name": variant_name,
                    "config_digest": config_digest,
                    "dimensions": [item.model_dump(mode="json") for item in dim_values],
                },
            )
            variants.append(
                PlannedVariant(
                    variant_id=variant_id,
                    variant_name=variant_name,
                    baseline=False,
                    dimensions=tuple(dim_values),
                    fixed_dimensions=dict(sorted(experiment.fixed_dimensions.items())),
                    resolved_config=resolved,
                    config_diff=config_diff,
                    config_digest=config_digest,
                    comparison_axes=_comparison_axes(dim_values, config_diff),
                )
            )

    variants.sort(key=lambda item: (0 if item.baseline else 1, item.variant_name, item.variant_id))
    dedup: dict[str, PlannedVariant] = {}
    for variant in variants:
        dedup[variant.variant_id] = variant
    return list(dedup.values())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(dict(row)) + "\n")


def _coerce_split_metrics(payload: Any) -> tuple[SplitMetricSummary, ...]:
    if payload is None:
        return tuple()
    rows: list[SplitMetricSummary] = []

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            if isinstance(item, Mapping):
                rows.append(SplitMetricSummary.model_validate(item))
        rows.sort(key=lambda row: row.split)
        return tuple(rows)

    if isinstance(payload, Mapping):
        for split_name in sorted(payload.keys()):
            value = payload[split_name]
            if isinstance(value, Mapping):
                if "metrics" in value:
                    rows.append(
                        SplitMetricSummary(
                            split=str(split_name),
                            metrics={k: float(v) for k, v in dict(value.get("metrics", {})).items()},
                            metadata=dict(value.get("metadata", {})),
                        )
                    )
                else:
                    metric_map: dict[str, float] = {}
                    for key, metric_value in value.items():
                        try:
                            metric_map[str(key)] = float(metric_value)
                        except Exception:
                            continue
                    rows.append(SplitMetricSummary(split=str(split_name), metrics=metric_map))
        return tuple(rows)

    return tuple()


def _coerce_result_envelope(payload: Any) -> ResultEnvelope:
    if isinstance(payload, Mapping):
        if "metrics" in payload or "status" in payload or "artifacts" in payload or "notes" in payload or "split_metrics" in payload:
            raw = dict(payload)
            raw["split_metrics"] = _coerce_split_metrics(raw.get("split_metrics"))
            if "metrics" in raw:
                raw["metrics"] = {str(k): float(v) for k, v in dict(raw["metrics"]).items()}
            return ResultEnvelope.model_validate(raw)

        metrics: dict[str, float] = {}
        for key, value in payload.items():
            try:
                metrics[key] = float(value)
            except Exception:
                continue
        return ResultEnvelope(status=RunStatus.COMPLETED, metrics=metrics, metadata={"raw_payload": dict(payload)})
    raise AblationFailure("Result payload must be a JSON object")


def _runner_env(manifest: VariantManifest) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ABLATION_STUDY_RUN_ID": manifest.study_run_id,
            "ABLATION_EXPERIMENT": manifest.experiment_name,
            "ABLATION_VARIANT_ID": manifest.variant_id,
            "ABLATION_VARIANT_NAME": manifest.variant_name,
            "ABLATION_BASELINE": "1" if manifest.baseline else "0",
            "ABLATION_RESOLVED_CONFIG": manifest.resolved_config_path,
            "ABLATION_RESULT_PATH": manifest.result_path,
        }
    )
    return env


def _render_runner_command(template: str, manifest: VariantManifest) -> str:
    return template.format(
        study_run_id=manifest.study_run_id,
        experiment_name=manifest.experiment_name,
        variant_id=manifest.variant_id,
        variant_name=manifest.variant_name,
        run_dir=str(Path(manifest.resolved_config_path).parent),
        resolved_config=manifest.resolved_config_path,
        result_path=manifest.result_path,
    )


def _execute_variant(
    *,
    manifest: VariantManifest,
    command_template: str,
) -> tuple[str, ResultEnvelope | None, dict[str, Any] | None]:
    rendered = _render_runner_command(command_template, manifest)
    try:
        completed = subprocess.run(
            rendered,
            shell=True,
            env=_runner_env(manifest),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        return (
            RunStatus.FAILED,
            None,
            {
                "code": "runner_start_failure",
                "message": str(exc),
            },
        )

    result_path = Path(manifest.result_path)
    if completed.returncode != 0:
        return (
            RunStatus.FAILED,
            None,
            {
                "code": "runner_nonzero_exit",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "command": rendered,
            },
        )
    if not result_path.exists():
        return (
            RunStatus.FAILED,
            None,
            {
                "code": "missing_result_file",
                "message": f"Runner completed successfully but did not write result file: {result_path}",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "command": rendered,
            },
        )

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        envelope = _coerce_result_envelope(payload)
    except Exception as exc:
        return (
            RunStatus.FAILED,
            None,
            {
                "code": "invalid_result_payload",
                "message": str(exc),
                "command": rendered,
            },
        )
    return RunStatus.COMPLETED, envelope, None


def _aggregate_metric_summary(manifests: Sequence[VariantManifest]) -> dict[str, Any]:
    successful = [manifest for manifest in manifests if manifest.status == RunStatus.COMPLETED and manifest.result is not None]
    baseline = next((manifest for manifest in successful if manifest.baseline), None)
    metric_rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for manifest in successful:
        assert manifest.result is not None
        for key, value in sorted(manifest.result.metrics.items()):
            metric_rows[key].append((manifest.variant_id, float(value)))

    summary: dict[str, Any] = {}
    for metric_name in sorted(metric_rows):
        rows = sorted(metric_rows[metric_name], key=lambda item: item[0])
        values = [value for _, value in rows]
        best_variant_id, best_value = sorted(rows, key=lambda item: (-item[1], item[0]))[0]
        baseline_value = None
        if baseline is not None and baseline.result is not None and metric_name in baseline.result.metrics:
            baseline_value = float(baseline.result.metrics[metric_name])
        summary[metric_name] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "best_variant_id": best_variant_id,
            "best_value": best_value,
            "baseline_value": baseline_value,
            "best_delta_vs_baseline": (best_value - baseline_value) if baseline_value is not None else None,
        }
    return summary


def _aggregate_split_metric_summary(manifests: Sequence[VariantManifest]) -> dict[str, Any]:
    successful = [manifest for manifest in manifests if manifest.status == RunStatus.COMPLETED and manifest.result is not None]
    baseline = next((manifest for manifest in successful if manifest.baseline), None)

    split_rows: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    baseline_lookup: dict[str, dict[str, float]] = defaultdict(dict)

    if baseline is not None and baseline.result is not None:
        for row in baseline.result.split_metrics:
            for metric_name, value in sorted(row.metrics.items()):
                baseline_lookup[row.split][metric_name] = float(value)

    for manifest in successful:
        assert manifest.result is not None
        for row in manifest.result.split_metrics:
            for metric_name, value in sorted(row.metrics.items()):
                split_rows[row.split][metric_name].append((manifest.variant_id, float(value)))

    out: dict[str, Any] = {}
    for split_name in sorted(split_rows):
        out[split_name] = {}
        for metric_name in sorted(split_rows[split_name]):
            rows = sorted(split_rows[split_name][metric_name], key=lambda item: item[0])
            values = [value for _, value in rows]
            best_variant_id, best_value = sorted(rows, key=lambda item: (-item[1], item[0]))[0]
            baseline_value = baseline_lookup.get(split_name, {}).get(metric_name)
            out[split_name][metric_name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "best_variant_id": best_variant_id,
                "best_value": best_value,
                "baseline_value": baseline_value,
                "best_delta_vs_baseline": (best_value - baseline_value) if baseline_value is not None else None,
            }
    return out


def _aggregate_dimension_summary(manifests: Sequence[VariantManifest]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[VariantManifest]]] = defaultdict(lambda: defaultdict(list))
    for manifest in manifests:
        for dimension in manifest.dimensions:
            grouped[dimension.name][json.dumps(dimension.value, sort_keys=True)].append(manifest)

    output: dict[str, Any] = {}
    for dimension_name in sorted(grouped):
        value_summary: dict[str, Any] = {}
        for value_key in sorted(grouped[dimension_name]):
            manifests_for_value = sorted(grouped[dimension_name][value_key], key=lambda item: item.variant_id)
            completed = [item for item in manifests_for_value if item.status == RunStatus.COMPLETED and item.result is not None]
            metrics: dict[str, list[float]] = defaultdict(list)
            split_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for manifest in completed:
                assert manifest.result is not None
                for metric_name, metric_value in sorted(manifest.result.metrics.items()):
                    metrics[metric_name].append(float(metric_value))
                for row in manifest.result.split_metrics:
                    for metric_name, metric_value in sorted(row.metrics.items()):
                        split_metrics[row.split][metric_name].append(float(metric_value))

            value_summary[value_key] = {
                "variant_ids": [item.variant_id for item in manifests_for_value],
                "completed_count": len(completed),
                "failed_count": sum(1 for item in manifests_for_value if item.status == RunStatus.FAILED),
                "metrics": {
                    metric_name: {
                        "mean": (sum(values) / len(values)) if values else None,
                        "min": min(values) if values else None,
                        "max": max(values) if values else None,
                    }
                    for metric_name, values in sorted(metrics.items())
                },
                "split_metrics": {
                    split: {
                        metric_name: {
                            "mean": (sum(values) / len(values)) if values else None,
                            "min": min(values) if values else None,
                            "max": max(values) if values else None,
                        }
                        for metric_name, values in sorted(metric_map.items())
                    }
                    for split, metric_map in sorted(split_metrics.items())
                },
            }
        output[dimension_name] = value_summary
    return output


def _build_variant_manifest(
    *,
    study_run_id: str,
    experiment_name: str,
    variant_index: int,
    variant: PlannedVariant,
    resolved_config_path: str,
    result_path: str,
    runner_command: str | None,
    reproducibility: ReproducibilityMetadata,
    base_config_digest: str,
) -> VariantManifest:
    return VariantManifest(
        artifact_version=OFFLINE_ARTIFACT_VERSION,
        study_run_id=study_run_id,
        experiment_name=experiment_name,
        variant_id=variant.variant_id,
        variant_name=variant.variant_name,
        variant_index=variant_index,
        baseline=variant.baseline,
        dimensions=variant.dimensions,
        fixed_dimensions=variant.fixed_dimensions,
        comparison_axes=variant.comparison_axes,
        config_digest=variant.config_digest,
        base_config_digest=base_config_digest,
        config_diff=variant.config_diff,
        resolved_config_path=resolved_config_path,
        result_path=result_path,
        runner_command=runner_command,
        reproducibility=reproducibility,
    )


def run_ablation_study(config: AblationStudyConfig, *, config_path: str | None = None) -> StudyManifest:
    experiment = _select_experiment(config)
    base_config = _load_base_config(config)
    experiment_base = _deep_merge(base_config, experiment.base_overrides)
    base_config_digest = _stable_hash("ablation_base_config", experiment_base)
    study_config_digest = _stable_hash("ablation_study_config", config.model_dump(mode="json"))
    experiment_digest = _stable_hash("ablation_experiment", experiment.model_dump(mode="json"))

    variants = _plan_variants(experiment=experiment, base_config=base_config)
    if not variants:
        raise AblationFailure(f"Experiment '{experiment.name}' resolved to zero variants")

    repro = _build_repro_metadata(
        config_path=config_path,
        base_config_path=config.base_config_path,
        selected_experiment=experiment.name,
    )
    run_id = _stable_hash(
        "ablation_study",
        {
            "run_name": config.run_name,
            "experiment_name": experiment.name,
            "base_config_digest": base_config_digest,
            "variants": [variant.variant_id for variant in variants],
        },
    )

    base_dir = Path(config.output_dir)
    run_dir = base_dir / run_id
    if run_dir.exists():
        if not config.overwrite:
            raise AblationFailure(
                f"Deterministic ablation output directory already exists: {run_dir}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    variant_dir = run_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=False)

    registry_snapshot_path = run_dir / "registry_snapshot.json"
    _write_json(
        registry_snapshot_path,
        {
            "study_config": config.model_dump(mode="json"),
            "selected_experiment": experiment.model_dump(mode="json"),
            "resolved_base_config": experiment_base,
        },
    )

    runner_command = config.runner_command or experiment.runner_command
    stop_on_failure = bool(config.stop_on_failure or experiment.stop_on_failure)
    plan_only = bool(config.plan_only)
    result_filename = experiment.result_filename or config.result_filename

    manifests: list[VariantManifest] = []
    for index, variant in enumerate(variants):
        this_variant_dir = variant_dir / variant.variant_id
        this_variant_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = this_variant_dir / "resolved_config.json"
        manifest_path = this_variant_dir / "manifest.json"
        result_path = this_variant_dir / result_filename

        if config.emit_resolved_configs:
            _write_json(resolved_config_path, variant.resolved_config)
        else:
            _write_json(resolved_config_path, {"config_digest": variant.config_digest})

        manifest = _build_variant_manifest(
            study_run_id=run_id,
            experiment_name=experiment.name,
            variant_index=index,
            variant=variant,
            resolved_config_path=str(resolved_config_path),
            result_path=str(result_path),
            runner_command=runner_command,
            reproducibility=repro,
            base_config_digest=base_config_digest,
        )

        if plan_only:
            manifest = manifest.model_copy(update={"status": RunStatus.PLANNED})
        else:
            if runner_command is None:
                raise AblationFailure(
                    "A runner command is required unless --plan-only is used. Provide it via config or --runner-command."
                )
            status, result, failure = _execute_variant(manifest=manifest, command_template=runner_command)
            manifest = manifest.model_copy(update={"status": status, "result": result, "failure": failure})
            if status == RunStatus.FAILED and stop_on_failure:
                _write_json(manifest_path, manifest.model_dump(mode="json"))
                manifests.append(manifest)
                break

        _write_json(manifest_path, manifest.model_dump(mode="json"))
        manifests.append(manifest)

    manifests.sort(key=lambda item: (item.variant_index, item.variant_id))
    results_jsonl_path = run_dir / "results.jsonl"
    _write_jsonl(
        results_jsonl_path,
        (
            {
                "variant_id": manifest.variant_id,
                "variant_name": manifest.variant_name,
                "baseline": manifest.baseline,
                "comparison_axes": list(manifest.comparison_axes),
                "status": manifest.status,
                "dimensions": [item.model_dump(mode="json") for item in manifest.dimensions],
                "config_diff": manifest.config_diff,
                "metrics": manifest.result.metrics if manifest.result else {},
                "split_metrics": [row.model_dump(mode="json") for row in manifest.result.split_metrics] if manifest.result else [],
                "failure": manifest.failure,
            }
            for manifest in manifests
        ),
    )

    aggregate_summary = {
        "metric_summary": _aggregate_metric_summary(manifests),
        "split_metric_summary": _aggregate_split_metric_summary(manifests),
        "dimension_summary": _aggregate_dimension_summary(manifests),
    }
    aggregate_summary_path = run_dir / "aggregate_summary.json"
    _write_json(aggregate_summary_path, aggregate_summary)

    failure_summary: dict[str, int] = defaultdict(int)
    for manifest in manifests:
        if manifest.failure and isinstance(manifest.failure, Mapping):
            failure_summary[str(manifest.failure.get("code", "unknown"))] += 1

    study_manifest = StudyManifest(
        artifact_version=config.artifact_version,
        run_id=run_id,
        run_name=config.run_name,
        experiment_name=experiment.name,
        config_digest=study_config_digest,
        experiment_digest=experiment_digest,
        base_config_digest=base_config_digest,
        variant_count=len(manifests),
        completed_count=sum(1 for manifest in manifests if manifest.status == RunStatus.COMPLETED),
        failed_count=sum(1 for manifest in manifests if manifest.status == RunStatus.FAILED),
        planned_only=plan_only,
        reproducibility=repro,
        aggregate_summary_path=str(aggregate_summary_path),
        results_jsonl_path=str(results_jsonl_path),
        registry_snapshot_path=str(registry_snapshot_path),
        variant_manifest_paths=tuple(str(variant_dir / manifest.variant_id / "manifest.json") for manifest in manifests),
        metric_summary=aggregate_summary["metric_summary"],
        split_metric_summary=aggregate_summary["split_metric_summary"],
        dimension_summary=aggregate_summary["dimension_summary"],
        failure_summary=dict(sorted(failure_summary.items())),
        notes=[] if not stop_on_failure else ["stop_on_failure was enabled"],
    )
    _write_json(run_dir / "manifest.json", study_manifest.model_dump(mode="json"))
    return study_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run manifest-backed controlled ablation studies")
    parser.add_argument("--config", type=str, default=None, help="Path to ablation study config (JSON/YAML)")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name in the ablation registry")
    parser.add_argument("--base-config", type=str, default=None, help="Base runtime/training config to ablate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root for ablation artifacts")
    parser.add_argument("--run-name", type=str, default=None, help="Human-readable ablation study name")
    parser.add_argument(
        "--dimension",
        action="append",
        default=[],
        help="Inline ablation dimension in the form name=path:value1,value2,... ; repeatable",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Base-config override in the form path=value ; repeatable",
    )
    parser.add_argument("--runner-command", type=str, default=None, help="Command template used to execute each variant")
    parser.add_argument("--result-filename", type=str, default=None, help="Structured JSON result filename expected per variant")
    parser.add_argument("--plan-only", action="store_true", help="Only materialize manifests and resolved configs")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop the study after the first failed variant")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing deterministic study directory")
    parser.add_argument("--no-baseline", action="store_true", help="Do not emit a baseline variant for inline experiments")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config, config_path = _resolve_study_config(args)
        manifest = run_ablation_study(config, config_path=config_path)
        print(f"[run_ablations] study: {manifest.run_id}")
        print(f"[run_ablations] experiment: {manifest.experiment_name}")
        print(f"[run_ablations] variants: {manifest.variant_count}")
        print(f"[run_ablations] completed: {manifest.completed_count}")
        print(f"[run_ablations] failed: {manifest.failed_count}")
        print(f"[run_ablations] aggregate summary: {manifest.aggregate_summary_path}")
        print(f"[run_ablations] results: {manifest.results_jsonl_path}")
        return 0 if manifest.failed_count == 0 else 2
    except (AblationFailure, FileNotFoundError, ValidationError, ValueError, TypeError) as exc:
        print(f"[run_ablations] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
"""Locked pass inventory validation for the final release gate."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash


class LockedPassRegistryError(ValueError):
    """Raised when the locked pass registry is incomplete or inconsistent."""


REQUIRED_PASS_IDS = (
    "pass_1_metric_submission_truth",
    "pass_2_canonical_proof_spine",
    "pass_2_5_dsl_compatibility",
    "pass_3_small_typed_solver_core",
    "pass_4_split_safe_generation",
    "pass_5_trace_compiler_audit",
    "pass_6_sft_dpo_dataset_builders",
    "pass_7_lora_training_interfaces",
    "pass_8_transfer_promotion_gates",
    "pass_9_packaging_runtime_validator",
)


@dataclass(frozen=True)
class LockedPassRecord:
    pass_id: str
    name: str
    status: str
    required_modules: tuple[str, ...]
    required_tests: tuple[str, ...]
    representative_imports: tuple[str, ...]
    expected_suite_marker: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    record_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("pass_id", "name", "status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LockedPassRegistryError(f"{name} must be non-empty.")
        if self.status != "LOCKED":
            raise LockedPassRegistryError("locked pass status must be LOCKED.")
        for tuple_name in ("required_modules", "required_tests", "representative_imports"):
            value = getattr(self, tuple_name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item.strip() for item in value):
                raise LockedPassRegistryError(f"{tuple_name} must be a tuple of non-empty strings.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "record_hash")
        if not self.record_hash:
            object.__setattr__(self, "record_hash", expected)
        elif self.record_hash != expected:
            raise LockedPassRegistryError("record_hash does not match locked pass record payload.")


@dataclass(frozen=True)
class LockedPassRegistry:
    records: tuple[LockedPassRecord, ...]
    suite_result_summary: str | None
    scoped_suite_command: str
    repository_root: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    registry_hash: str = ""

    def __post_init__(self) -> None:
        records = tuple(sorted(self.records, key=lambda item: item.pass_id))
        if not records:
            raise LockedPassRegistryError("registry requires locked pass records.")
        ids = tuple(record.pass_id for record in records)
        if len(ids) != len(set(ids)):
            raise LockedPassRegistryError("registry contains duplicate pass_id values.")
        missing = tuple(pass_id for pass_id in REQUIRED_PASS_IDS if pass_id not in set(ids))
        if missing:
            raise LockedPassRegistryError(f"registry missing required pass(es): {missing}")
        if any(record.status != "LOCKED" for record in records):
            raise LockedPassRegistryError("all locked pass records must have status LOCKED.")
        if not isinstance(self.scoped_suite_command, str) or not self.scoped_suite_command.strip():
            raise LockedPassRegistryError("scoped_suite_command must be non-empty.")
        if self.suite_result_summary is not None and not isinstance(self.suite_result_summary, str):
            raise LockedPassRegistryError("suite_result_summary must be a string or None.")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = compute_locked_pass_registry_hash(self)
        if not self.registry_hash:
            object.__setattr__(self, "registry_hash", expected)
        elif self.registry_hash != expected:
            raise LockedPassRegistryError("registry_hash does not match locked pass registry payload.")


def build_locked_pass_registry(
    *,
    repository_root: str | Path | None = None,
    suite_result_summary: str | None = None,
    scoped_suite_command: str = "pytest tests -q -p no:cacheprovider",
    metadata: Mapping[str, Any] | None = None,
) -> LockedPassRegistry:
    root = None if repository_root is None else str(Path(repository_root))
    return LockedPassRegistry(
        records=_default_records(),
        suite_result_summary=suite_result_summary,
        scoped_suite_command=scoped_suite_command,
        repository_root=root,
        metadata=dict(metadata or {}),
    )


def validate_locked_pass_registry(registry: LockedPassRegistry) -> LockedPassRegistry:
    if not isinstance(registry, LockedPassRegistry):
        raise LockedPassRegistryError("registry must be a LockedPassRegistry.")
    if registry.registry_hash != compute_locked_pass_registry_hash(registry):
        raise LockedPassRegistryError("registry_hash does not match locked pass registry payload.")
    root = _repository_root(registry.repository_root)
    if not (root / "src").is_dir() or not (root / "tests").is_dir():
        raise LockedPassRegistryError("repository root must contain src/ and tests/.")
    src_root = (root / "src").resolve()
    for record in registry.records:
        expected_record_hash = _payload_hash(record, "record_hash")
        if record.record_hash != expected_record_hash:
            raise LockedPassRegistryError(f"record_hash mismatch: {record.pass_id}")
        for module_name in record.required_modules:
            _require_module_origin_under_src(module_name, src_root, missing_prefix="missing module", outside_prefix="module_origin_outside_repo")
        for test_path in record.required_tests:
            if not (root / test_path).is_file():
                raise LockedPassRegistryError(f"missing required test: {test_path}")
        for import_ref in record.representative_imports:
            _resolve_import_ref(import_ref, src_root)
    return registry


def compute_locked_pass_registry_hash(registry: LockedPassRegistry | Mapping[str, Any]) -> str:
    if isinstance(registry, Mapping):
        payload = dict(registry)
    else:
        payload = {item.name: getattr(registry, item.name) for item in fields(LockedPassRegistry)}
    payload.pop("registry_hash", None)
    return stable_hash(payload)


def _default_records() -> tuple[LockedPassRecord, ...]:
    return (
        LockedPassRecord(
            pass_id="pass_1_metric_submission_truth",
            name="Metric and submission truth",
            status="LOCKED",
            required_modules=("nemotron_engine.scoring.local_scorer", "nemotron_engine.runtime.serving_config", "nemotron_engine.submission.submission_validator"),
            required_tests=("tests/test_local_scorer.py", "tests/test_serving_config.py", "tests/test_submission_validator.py"),
            representative_imports=("nemotron_engine.runtime.serving_config:ServingConfig", "nemotron_engine.submission.submission_validator:validate_submission_zip"),
        ),
        LockedPassRecord(
            pass_id="pass_2_canonical_proof_spine",
            name="Canonical proof spine",
            status="LOCKED",
            required_modules=("nemotron_engine.core.schemas", "nemotron_engine.parsing.canonicalizer"),
            required_tests=("tests/test_schemas.py", "tests/test_canonicalizer.py", "tests/test_pass2_api_compatibility.py"),
            representative_imports=("nemotron_engine.core.schemas:CanonicalProblem", "nemotron_engine.core.schemas:stable_hash"),
        ),
        LockedPassRecord(
            pass_id="pass_2_5_dsl_compatibility",
            name="DSL compatibility patch",
            status="LOCKED",
            required_modules=("nemotron_engine.programs.dsl", "nemotron_engine.programs.executor"),
            required_tests=("tests/test_dsl_executor.py", "tests/test_program_primitives.py"),
            representative_imports=("nemotron_engine.programs.executor:execute_program",),
        ),
        LockedPassRecord(
            pass_id="pass_3_small_typed_solver_core",
            name="Small typed solver core",
            status="LOCKED",
            required_modules=("nemotron_engine.solvers.base", "nemotron_engine.solvers.known_numeric"),
            required_tests=("tests/test_solver_base.py", "tests/test_known_numeric_solver.py", "tests/test_pass3_api_compatibility.py"),
            representative_imports=("nemotron_engine.solvers.known_numeric:solve", "nemotron_engine.programs.type_system:infer_value_domain"),
        ),
        LockedPassRecord(
            pass_id="pass_4_split_safe_generation",
            name="Split-safe generation",
            status="LOCKED",
            required_modules=("nemotron_engine.data.split_builder", "nemotron_engine.data.contamination"),
            required_tests=("tests/test_split_builder.py", "tests/test_contamination.py", "tests/test_pass4_api_compatibility.py"),
            representative_imports=("nemotron_engine.data.split_builder:SplitManifest", "nemotron_engine.data.contamination:ContaminationReport"),
        ),
        LockedPassRecord(
            pass_id="pass_5_trace_compiler_audit",
            name="Trace compiler and audit",
            status="LOCKED",
            required_modules=("nemotron_engine.traces.trace_compiler", "nemotron_engine.traces.trace_auditor"),
            required_tests=("tests/test_trace_compiler.py", "tests/test_trace_auditor.py", "tests/test_pass5_api_compatibility.py"),
            representative_imports=("nemotron_engine.traces.trace_compiler:TraceRecord", "nemotron_engine.traces.trace_compiler:TraceAuditReport"),
        ),
        LockedPassRecord(
            pass_id="pass_6_sft_dpo_dataset_builders",
            name="SFT/DPO dataset builders",
            status="LOCKED",
            required_modules=("nemotron_engine.training.sft_builder", "nemotron_engine.training.dpo_builder"),
            required_tests=("tests/test_sft_builder.py", "tests/test_dpo_builder.py", "tests/test_pass6_api_compatibility.py"),
            representative_imports=("nemotron_engine.training.sft_builder:build_sft_dataset", "nemotron_engine.training.dpo_builder:build_dpo_dataset"),
        ),
        LockedPassRecord(
            pass_id="pass_7_lora_training_interfaces",
            name="LoRA/training interface safety gates",
            status="LOCKED",
            required_modules=("nemotron_engine.training.lora_config", "nemotron_engine.training.train_sft", "nemotron_engine.training.train_dpo", "nemotron_engine.training.train_grpo"),
            required_tests=("tests/test_lora_config.py", "tests/test_train_sft.py", "tests/test_train_dpo.py", "tests/test_train_grpo.py", "tests/test_pass7_api_compatibility.py"),
            representative_imports=("nemotron_engine.training.lora_config:LoRAConfig", "nemotron_engine.training.train_sft:plan_sft_training"),
        ),
        LockedPassRecord(
            pass_id="pass_8_transfer_promotion_gates",
            name="Transfer harness and promotion gates",
            status="LOCKED",
            required_modules=("nemotron_engine.evaluation.transfer_harness", "nemotron_engine.evaluation.promotion_gates"),
            required_tests=("tests/test_transfer_harness.py", "tests/test_promotion_gates.py", "tests/test_pass8_api_compatibility.py"),
            representative_imports=("nemotron_engine.evaluation.transfer_harness:TransferEvaluationReport", "nemotron_engine.evaluation.promotion_gates:PromotionGateReport"),
        ),
        LockedPassRecord(
            pass_id="pass_9_packaging_runtime_validator",
            name="Submission packaging and offline runtime validator",
            status="LOCKED",
            required_modules=("nemotron_engine.packaging.submission_manifest", "nemotron_engine.packaging.preflight_gate"),
            required_tests=("tests/test_submission_manifest.py", "tests/test_preflight_gate.py", "tests/test_pass9_api_compatibility.py"),
            representative_imports=("nemotron_engine.packaging.submission_manifest:SubmissionManifest", "nemotron_engine.packaging.preflight_gate:PreflightGateReport"),
        ),
    )


def _resolve_import_ref(import_ref: str, src_root: Path) -> object:
    module_name, sep, attr = import_ref.partition(":")
    if not sep or not module_name or not attr:
        raise LockedPassRegistryError(f"representative import must be module:attribute: {import_ref}")
    _require_module_origin_under_src(
        module_name,
        src_root,
        missing_prefix="missing representative import",
        outside_prefix="representative_import_origin_outside_repo",
    )
    module = importlib.import_module(module_name)
    current: object = module
    for part in attr.split("."):
        if not hasattr(current, part):
            raise LockedPassRegistryError(f"representative import missing attribute: {import_ref}")
        current = getattr(current, part)
    return current


def _require_module_origin_under_src(module_name: str, src_root: Path, *, missing_prefix: str, outside_prefix: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise LockedPassRegistryError(f"{missing_prefix}: {module_name}")
    if module_name == "nemotron_engine" or module_name.startswith("nemotron_engine."):
        origin = spec.origin
        if not origin or origin in {"built-in", "frozen", "namespace"}:
            raise LockedPassRegistryError(f"{missing_prefix}: {module_name}")
        origin_path = Path(origin).resolve()
        if not origin_path.exists():
            raise LockedPassRegistryError(f"{missing_prefix}: {module_name}")
        try:
            origin_path.relative_to(src_root)
        except ValueError as exc:
            raise LockedPassRegistryError(f"{outside_prefix}: {module_name}: {origin_path}") from exc
        return origin_path
    return Path(spec.origin).resolve() if spec.origin else src_root


def _repository_root(value: str | None) -> Path:
    if value is not None:
        return Path(value)
    return Path(__file__).resolve().parents[3]


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "LockedPassRecord",
    "LockedPassRegistry",
    "LockedPassRegistryError",
    "build_locked_pass_registry",
    "compute_locked_pass_registry_hash",
    "validate_locked_pass_registry",
]

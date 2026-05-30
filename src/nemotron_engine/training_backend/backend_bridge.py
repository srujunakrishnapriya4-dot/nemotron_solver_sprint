"""Bridge Pass 7 dry-run training plans to Pass 11 backend invocations."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.training.reward_audit import RewardAuditReport, grpo_admission_decision
from nemotron_engine.training.training_contracts import TrainingInputManifest, TrainingPlan, validate_training_plan

from .adapter_validator import AdapterValidationReport
from .backend_contracts import BackendInvocation, BackendKind, TrainingStage
from .backend_contracts import validate_backend_result
from .run_manifest import TrainingRunManifest, build_training_run_manifest
from .training_log_validator import TrainingLogValidationReport


class BackendBridgeError(ValueError):
    """Raised when Pass 7 plans cannot be bridged safely."""


def build_invocation_from_sft_plan(
    plan: TrainingPlan,
    manifest: TrainingInputManifest | None = None,
    *,
    invocation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BackendInvocation:
    return _build_invocation(plan, manifest, BackendKind.EXTERNAL_SFT.value, TrainingStage.SFT.value, invocation_id, metadata)


def build_invocation_from_dpo_plan(
    plan: TrainingPlan,
    manifest: TrainingInputManifest | None = None,
    *,
    invocation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BackendInvocation:
    return _build_invocation(plan, manifest, BackendKind.EXTERNAL_DPO.value, TrainingStage.DPO.value, invocation_id, metadata)


def build_invocation_from_grpo_plan(
    plan: TrainingPlan,
    reward_audit: RewardAuditReport,
    manifest: TrainingInputManifest | None = None,
    *,
    invocation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BackendInvocation:
    if not isinstance(reward_audit, RewardAuditReport) or grpo_admission_decision(reward_audit) is not True:
        raise BackendBridgeError("GRPO invocation requires accepted reward audit.")
    _validate_hash_field(reward_audit, "report_hash", "reward audit")
    extra = {**dict(metadata or {}), "reward_audit_hash": reward_audit.report_hash}
    return _build_invocation(plan, manifest, BackendKind.EXTERNAL_GRPO.value, TrainingStage.GRPO.value, invocation_id, extra)


def build_invocation_from_final_sft_refresh(
    plan: TrainingPlan,
    manifest: TrainingInputManifest | None = None,
    *,
    invocation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BackendInvocation:
    return _build_invocation(
        plan,
        manifest,
        BackendKind.EXTERNAL_FINAL_SFT.value,
        TrainingStage.FINAL_SFT_REFRESH.value,
        invocation_id,
        metadata,
    )


def build_run_manifest_from_backend_outputs(
    *,
    invocation: BackendInvocation,
    backend_result: Any,
    adapter_report: AdapterValidationReport | None = None,
    log_report: TrainingLogValidationReport | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingRunManifest:
    invocation = _validate_invocation(invocation)
    result = validate_backend_result(backend_result, invocation=invocation, require_artifact_hashes=True)
    if result.trained:
        if adapter_report is None or adapter_report.passed is not True:
            raise BackendBridgeError("trained backend result requires passed adapter report.")
    if adapter_report is not None:
        _validate_hash_field(adapter_report, "report_hash", "adapter report")
    if log_report is not None:
        _validate_hash_field(log_report, "report_hash", "training log report")
    return build_training_run_manifest(
        invocation_hash=invocation.invocation_hash,
        backend_result=result,
        training_plan_hash=invocation.training_plan_hash,
        lora_config_hash=invocation.lora_config_hash,
        dataset_manifest_hash=invocation.dataset_manifest_hash,
        adapter_validation_report=adapter_report,
        training_log_report=log_report,
        metadata=dict(metadata or {}),
    )


def _build_invocation(
    plan: TrainingPlan,
    manifest: TrainingInputManifest | None,
    backend_kind: str,
    stage: str,
    invocation_id: str | None,
    metadata: Mapping[str, Any] | None,
) -> BackendInvocation:
    if not isinstance(plan, TrainingPlan):
        raise BackendBridgeError("plan must be a Pass 7 TrainingPlan.")
    plan = validate_training_plan(plan)
    _validate_hash_field(plan, "plan_hash", "training plan")
    if plan.stage != stage:
        raise BackendBridgeError("plan stage is incompatible with backend invocation.")
    row_count = 0
    pair_count = 0
    dataset_hash = plan.dataset_manifest_hash
    meta = dict(metadata or {})
    if manifest is not None:
        if not isinstance(manifest, TrainingInputManifest):
            raise BackendBridgeError("manifest must be a TrainingInputManifest.")
        _validate_hash_field(manifest, "manifest_hash", "training input manifest")
        if manifest.manifest_hash != plan.dataset_manifest_hash:
            raise BackendBridgeError("manifest hash does not match training plan.")
        if manifest.dataset_type == "sft":
            row_count = manifest.input_count
        elif manifest.dataset_type == "dpo":
            pair_count = manifest.input_count
        else:
            raise BackendBridgeError("unsupported manifest dataset_type.")
    else:
        meta["input_counts_from_manifest"] = False
    invocation_id = invocation_id or "inv-" + stable_hash(
        {
            "backend_kind": backend_kind,
            "stage": stage,
            "plan_hash": plan.plan_hash,
            "seed": plan.seed,
            "dry_run": plan.dry_run,
        }
    )[:24]
    return BackendInvocation(
        invocation_id=invocation_id,
        backend_kind=backend_kind,
        stage=stage,
        training_plan_hash=plan.plan_hash,
        lora_config_hash=plan.lora_config_hash,
        dataset_manifest_hash=dataset_hash,
        input_row_count=row_count,
        input_pair_count=pair_count,
        seed=plan.seed,
        dry_run=plan.dry_run,
        output_dir=plan.output_dir,
        metadata=meta,
    )


def _validate_hash_field(instance: object, hash_field: str, label: str) -> None:
    try:
        observed = getattr(instance, hash_field)
        expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    except Exception as exc:
        raise BackendBridgeError(f"could not validate {label} hash.") from exc
    if observed != expected:
        raise BackendBridgeError(f"{label} {hash_field} mismatch.")


def _validate_invocation(invocation: BackendInvocation) -> BackendInvocation:
    if not isinstance(invocation, BackendInvocation):
        raise BackendBridgeError("invocation must be a BackendInvocation.")
    return invocation


__all__ = [
    "BackendBridgeError",
    "build_invocation_from_dpo_plan",
    "build_invocation_from_final_sft_refresh",
    "build_invocation_from_grpo_plan",
    "build_invocation_from_sft_plan",
    "build_run_manifest_from_backend_outputs",
]

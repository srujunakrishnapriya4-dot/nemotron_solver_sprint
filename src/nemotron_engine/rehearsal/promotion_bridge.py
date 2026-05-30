"""Promotion-to-package rehearsal bridge for Pass 14."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation import (
    PrivateLikeReport,
    PromotionDecision,
    PromotionGateReport,
    SubmissionExactReport,
    TransferEvaluationReport,
)
from nemotron_engine.inference_eval import InferenceEvaluationManifest, validate_inference_evaluation_manifest

from .adapter_evidence import AdapterEvidenceBundle, validate_adapter_evidence_bundle
from .rehearsal_config import RehearsalConfig, reject_unsafe_metadata_claims, validate_rehearsal_config


class PromotionBridgeError(ValueError):
    """Raised when promotion rehearsal evidence is invalid."""


@dataclass(frozen=True)
class PromotionBridgeConfig:
    require_private_like: bool = True
    allow_warnings: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.require_private_like, bool) or not isinstance(self.allow_warnings, bool):
            raise PromotionBridgeError("promotion bridge flags must be booleans.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=False)
        except Exception as exc:
            raise PromotionBridgeError(str(exc)) from exc
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise PromotionBridgeError("config_hash does not match promotion bridge config payload.")


@dataclass(frozen=True)
class PromotionBridgeReport:
    promotion_decision_hash: str
    submission_exact_hash: str
    transfer_report_hash: str
    private_like_report_hash: str
    inference_evaluation_hash: str
    adapter_evidence_hash: str
    allowed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "promotion_decision_hash",
            "submission_exact_hash",
            "transfer_report_hash",
            "private_like_report_hash",
            "inference_evaluation_hash",
            "adapter_evidence_hash",
        ):
            _require_non_empty(getattr(self, name), name)
        if not isinstance(self.allowed, bool):
            raise PromotionBridgeError("allowed must be boolean.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.allowed and errors:
            raise PromotionBridgeError("allowed=True cannot include errors.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=False)
        except Exception as exc:
            raise PromotionBridgeError(str(exc)) from exc
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise PromotionBridgeError("report_hash does not match promotion bridge payload.")


def build_promotion_rehearsal_report(
    *,
    promotion_gate_report: PromotionGateReport,
    submission_exact_report: SubmissionExactReport,
    transfer_report: TransferEvaluationReport,
    private_like_report: PrivateLikeReport,
    inference_evaluation_manifest: InferenceEvaluationManifest,
    adapter_evidence_bundle: AdapterEvidenceBundle,
    rehearsal_config: RehearsalConfig,
    metadata: Mapping[str, Any] | None = None,
    config: PromotionBridgeConfig | None = None,
) -> PromotionBridgeReport:
    cfg = config or PromotionBridgeConfig(require_private_like=validate_rehearsal_config(rehearsal_config).require_private_like)
    rehearsal_config = validate_rehearsal_config(rehearsal_config)
    evidence = validate_adapter_evidence_bundle(adapter_evidence_bundle)
    inference = validate_inference_evaluation_manifest(inference_evaluation_manifest)
    _validate_report_hash(promotion_gate_report, "decision_hash")
    _validate_report_hash(submission_exact_report, "report_hash")
    _validate_report_hash(transfer_report, "report_hash")
    _validate_report_hash(private_like_report, "report_hash")
    errors: list[str] = []
    warnings: list[str] = []
    if promotion_gate_report.decision != PromotionDecision.ALLOW or promotion_gate_report.passed is not True:
        errors.append("promotion_not_allow")
    if submission_exact_report.passed is not True:
        errors.append("submission_exact_failed")
    if transfer_report.passed is not True:
        errors.append("transfer_failed")
    require_private = rehearsal_config.require_private_like and cfg.require_private_like
    if require_private and private_like_report.passed is not True:
        errors.append("private_like_failed")
    if rehearsal_config.require_completion_evaluation and inference.passed is not True:
        errors.append("inference_evaluation_failed")
    if evidence.evidence_complete is not True:
        errors.append("adapter_evidence_incomplete")
    input_hashes = dict(promotion_gate_report.input_report_hashes)
    if "submission_exact_report" not in input_hashes:
        errors.append("promotion_submission_exact_hash_missing")
    elif input_hashes["submission_exact_report"] != submission_exact_report.report_hash:
        errors.append("promotion_submission_exact_hash_mismatch")
    if "transfer_report" not in input_hashes:
        errors.append("promotion_transfer_hash_missing")
    elif input_hashes["transfer_report"] != transfer_report.report_hash:
        errors.append("promotion_transfer_hash_mismatch")
    if require_private:
        if "private_like_report" not in input_hashes:
            errors.append("promotion_private_like_hash_missing")
        elif input_hashes["private_like_report"] != private_like_report.report_hash:
            errors.append("promotion_private_like_hash_mismatch")
    elif input_hashes.get("private_like_report") not in (None, private_like_report.report_hash):
        errors.append("promotion_private_like_hash_mismatch")
    if promotion_gate_report.required_hashes.get("adapter_hash") not in (None, evidence.adapter_hash):
        errors.append("promotion_adapter_hash_mismatch")
    warnings.extend(evidence.warnings)
    if warnings and not cfg.allow_warnings:
        errors.append("warnings_not_allowed")
    return PromotionBridgeReport(
        promotion_decision_hash=promotion_gate_report.decision_hash,
        submission_exact_hash=submission_exact_report.report_hash,
        transfer_report_hash=transfer_report.report_hash,
        private_like_report_hash=private_like_report.report_hash,
        inference_evaluation_hash=inference.manifest_hash,
        adapter_evidence_hash=evidence.evidence_hash,
        allowed=not errors,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
        metadata=dict(metadata or {}),
    )


def validate_promotion_rehearsal_report(report: PromotionBridgeReport | Mapping[str, Any]) -> PromotionBridgeReport:
    normalized = report if isinstance(report, PromotionBridgeReport) else PromotionBridgeReport(**dict(report))
    if normalized.report_hash != _payload_hash(normalized, "report_hash"):
        raise PromotionBridgeError("report_hash does not match promotion bridge payload.")
    if normalized.allowed and normalized.errors:
        raise PromotionBridgeError("allowed=True cannot include errors.")
    return normalized


def _validate_report_hash(report: object, hash_field: str) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise PromotionBridgeError("could not validate input report hash.") from exc
    if observed != expected:
        raise PromotionBridgeError("input report hash mismatch.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionBridgeError(f"{field_name} must be non-empty.")
    return value


__all__ = [
    "PromotionBridgeConfig",
    "PromotionBridgeError",
    "PromotionBridgeReport",
    "build_promotion_rehearsal_report",
    "validate_promotion_rehearsal_report",
]

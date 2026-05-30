"""Deterministic manifest for Pass 13 completion evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .completion_capture import CompletionCaptureReport
from .inference_contracts import InferenceBackendInvocation, InferenceBackendResult, _reject_forbidden_metadata
from .private_like_bridge import CompletionPrivateLikeReport
from .prompt_batch import PromptBatch
from .transfer_bridge import CompletionTransferReport


class InferenceEvaluationManifestError(ValueError):
    """Raised when an inference evaluation manifest is unsafe."""


@dataclass(frozen=True)
class InferenceEvaluationManifest:
    manifest_id: str
    prompt_batch_hash: str
    serving_config_hash: str
    inference_invocation_hash: str
    inference_result_hash: str
    completion_capture_hash: str
    transfer_report_hash: str | None
    private_like_report_hash: str | None
    non_submission_exact: bool
    backend_kind: str
    dry_run: bool
    evaluated: bool
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "prompt_batch_hash",
            "serving_config_hash",
            "inference_invocation_hash",
            "inference_result_hash",
            "completion_capture_hash",
            "backend_kind",
        ):
            _require_non_empty(getattr(self, name), name)
        for name in ("transfer_report_hash", "private_like_report_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        for name in ("non_submission_exact", "dry_run", "evaluated", "passed"):
            if not isinstance(getattr(self, name), bool):
                raise InferenceEvaluationManifestError(f"{name} must be boolean.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        metadata = _safe_metadata(self.metadata)
        private_like_required = metadata.get("require_private_like") is not False
        expected_id = compute_inference_evaluation_manifest_id(self)
        if self.manifest_id != expected_id:
            raise InferenceEvaluationManifestError("manifest_id does not match inference evaluation identity payload.")
        if self.passed:
            if self.evaluated is not True:
                raise InferenceEvaluationManifestError("passed=True requires evaluated=True.")
            if not self.transfer_report_hash:
                raise InferenceEvaluationManifestError("passed=True requires transfer_report_hash.")
            if private_like_required and not self.private_like_report_hash:
                raise InferenceEvaluationManifestError("passed=True requires private_like_report_hash when private_like is required.")
            if self.non_submission_exact:
                raise InferenceEvaluationManifestError("passed=True blocked for non-submission-exact evaluation.")
            if self.dry_run:
                raise InferenceEvaluationManifestError("dry_run=True cannot pass.")
            if errors:
                raise InferenceEvaluationManifestError("passed=True cannot include errors.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_inference_evaluation_manifest_hash(self)
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected)
        elif self.manifest_hash != expected:
            raise InferenceEvaluationManifestError("manifest_hash does not match inference evaluation manifest payload.")


def build_inference_evaluation_manifest(
    *,
    prompt_batch: PromptBatch,
    invocation: InferenceBackendInvocation,
    result: InferenceBackendResult,
    capture_report: CompletionCaptureReport,
    transfer_report: CompletionTransferReport | None = None,
    private_like_report: CompletionPrivateLikeReport | None = None,
    require_private_like: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> InferenceEvaluationManifest:
    meta = {**dict(metadata or {}), "require_private_like": require_private_like}
    non_exact = bool(transfer_report.non_submission_exact if transfer_report is not None else False)
    evaluated = transfer_report is not None
    passed = bool(
        evaluated
        and transfer_report is not None
        and transfer_report.passed
        and (not require_private_like or (private_like_report is not None and private_like_report.passed))
        and not non_exact
        and not invocation.dry_run
    )
    errors: list[str] = []
    if transfer_report is not None and not transfer_report.passed:
        errors.extend(transfer_report.errors or ("transfer_failed",))
    if require_private_like:
        if private_like_report is None:
            errors.append("private_like_missing")
        elif not private_like_report.passed:
            errors.extend(private_like_report.errors or ("private_like_failed",))
    if non_exact:
        errors.append("non_submission_exact")
    if invocation.dry_run:
        errors.append("dry_run")
    payload = {
        "prompt_batch_hash": prompt_batch.batch_hash,
        "serving_config_hash": prompt_batch.serving_config_hash,
        "inference_invocation_hash": invocation.invocation_hash,
        "inference_result_hash": result.result_hash,
        "completion_capture_hash": capture_report.report_hash,
        "transfer_report_hash": transfer_report.report_hash if transfer_report is not None else None,
        "private_like_report_hash": private_like_report.report_hash if private_like_report is not None else None,
        "non_submission_exact": non_exact,
        "backend_kind": invocation.backend_kind,
        "dry_run": invocation.dry_run,
        "evaluated": evaluated,
        "passed": passed,
    }
    manifest_id = stable_hash(payload)
    return InferenceEvaluationManifest(
        manifest_id=manifest_id,
        prompt_batch_hash=prompt_batch.batch_hash,
        serving_config_hash=prompt_batch.serving_config_hash,
        inference_invocation_hash=invocation.invocation_hash,
        inference_result_hash=result.result_hash,
        completion_capture_hash=capture_report.report_hash,
        transfer_report_hash=transfer_report.report_hash if transfer_report is not None else None,
        private_like_report_hash=private_like_report.report_hash if private_like_report is not None else None,
        non_submission_exact=non_exact,
        backend_kind=invocation.backend_kind,
        dry_run=invocation.dry_run,
        evaluated=evaluated,
        passed=passed and not errors,
        errors=tuple(sorted(set(errors))) if not passed else (),
        warnings=(),
        metadata=meta,
    )


def validate_inference_evaluation_manifest(
    manifest: InferenceEvaluationManifest | Mapping[str, Any],
) -> InferenceEvaluationManifest:
    normalized = manifest if isinstance(manifest, InferenceEvaluationManifest) else InferenceEvaluationManifest(**dict(manifest))
    if normalized.manifest_id != compute_inference_evaluation_manifest_id(normalized):
        raise InferenceEvaluationManifestError("manifest_id does not match inference evaluation identity payload.")
    if normalized.manifest_hash != compute_inference_evaluation_manifest_hash(normalized):
        raise InferenceEvaluationManifestError("manifest_hash does not match inference evaluation manifest payload.")
    return normalized


def compute_inference_evaluation_manifest_hash(manifest: InferenceEvaluationManifest | Mapping[str, Any]) -> str:
    payload = dict(manifest) if isinstance(manifest, Mapping) else {item.name: getattr(manifest, item.name) for item in fields(InferenceEvaluationManifest)}
    payload.pop("manifest_hash", None)
    return stable_hash(payload)


def compute_inference_evaluation_manifest_id(manifest: InferenceEvaluationManifest | Mapping[str, Any]) -> str:
    payload = dict(manifest) if isinstance(manifest, Mapping) else {item.name: getattr(manifest, item.name) for item in fields(InferenceEvaluationManifest)}
    return stable_hash(
        {
            "prompt_batch_hash": payload.get("prompt_batch_hash"),
            "serving_config_hash": payload.get("serving_config_hash"),
            "inference_invocation_hash": payload.get("inference_invocation_hash"),
            "inference_result_hash": payload.get("inference_result_hash"),
            "completion_capture_hash": payload.get("completion_capture_hash"),
            "transfer_report_hash": payload.get("transfer_report_hash"),
            "private_like_report_hash": payload.get("private_like_report_hash"),
            "non_submission_exact": payload.get("non_submission_exact"),
            "backend_kind": payload.get("backend_kind"),
            "dry_run": payload.get("dry_run"),
            "evaluated": payload.get("evaluated"),
            "passed": payload.get("passed"),
        }
    )


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceEvaluationManifestError("metadata must be a mapping.")
    metadata = dict(value)
    try:
        _reject_forbidden_metadata(metadata)
    except Exception as exc:
        raise InferenceEvaluationManifestError(str(exc)) from exc
    return metadata


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceEvaluationManifestError(f"{field_name} must be non-empty.")
    return value


__all__ = [
    "InferenceEvaluationManifest",
    "InferenceEvaluationManifestError",
    "build_inference_evaluation_manifest",
    "compute_inference_evaluation_manifest_hash",
    "validate_inference_evaluation_manifest",
]

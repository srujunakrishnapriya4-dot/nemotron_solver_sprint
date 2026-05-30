"""Private-like evaluation bridge for captured Pass 13 completions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation.private_like import PrivateLikeGateConfig, PrivateLikeReport, evaluate_private_like_gate

from .inference_contracts import _reject_forbidden_metadata
from .transfer_bridge import CompletionTransferReport


class CompletionPrivateLikeError(ValueError):
    """Raised when private-like completion evaluation evidence is invalid."""


@dataclass(frozen=True)
class CompletionPrivateLikeConfig:
    private_like_config: PrivateLikeGateConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.private_like_config is not None and not isinstance(self.private_like_config, PrivateLikeGateConfig):
            raise CompletionPrivateLikeError("private_like_config must be a PrivateLikeGateConfig.")
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise CompletionPrivateLikeError("config_hash does not match private-like config payload.")


@dataclass(frozen=True)
class CompletionPrivateLikeReport:
    completion_transfer_report_hash: str
    private_like_report_hash: str
    private_like_report: PrivateLikeReport
    passed: bool
    non_submission_exact: bool = False
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("completion_transfer_report_hash", "private_like_report_hash"):
            _require_non_empty(getattr(self, name), name)
        if not isinstance(self.private_like_report, PrivateLikeReport):
            raise CompletionPrivateLikeError("private_like_report must be a PrivateLikeReport.")
        if self.private_like_report_hash != self.private_like_report.report_hash:
            raise CompletionPrivateLikeError("private_like_report_hash mismatch.")
        if not isinstance(self.passed, bool) or not isinstance(self.non_submission_exact, bool):
            raise CompletionPrivateLikeError("passed and non_submission_exact must be booleans.")
        errors = tuple(str(item) for item in self.errors)
        if self.passed and self.non_submission_exact:
            raise CompletionPrivateLikeError("passed=True is blocked for non-submission-exact evaluation.")
        if self.passed and self.private_like_report.passed is not True:
            raise CompletionPrivateLikeError("passed=True requires passed Pass 8 private-like report.")
        if self.passed and errors:
            raise CompletionPrivateLikeError("passed=True cannot include errors.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise CompletionPrivateLikeError("report_hash does not match private-like completion report payload.")


def evaluate_private_like_from_completions(
    transfer_report: CompletionTransferReport,
    *,
    config: CompletionPrivateLikeConfig | None = None,
) -> CompletionPrivateLikeReport:
    if not isinstance(transfer_report, CompletionTransferReport):
        raise CompletionPrivateLikeError("transfer_report must be a CompletionTransferReport.")
    private_examples = tuple(item for item in transfer_report.transfer_examples if item.split == "private_like")
    if not private_examples:
        raise CompletionPrivateLikeError("private_like slice required.")
    if not any(item.slice_name == "private_like" for item in transfer_report.transfer_report.slice_results):
        raise CompletionPrivateLikeError("private_like transfer slice report required.")
    cfg = config or CompletionPrivateLikeConfig()
    private_report = evaluate_private_like_gate(transfer_report.transfer_examples, transfer_report.transfer_report, cfg.private_like_config)
    passed = private_report.passed is True and not transfer_report.non_submission_exact
    errors: tuple[str, ...] = ()
    if private_report.passed is not True:
        errors = tuple(private_report.failure_reasons or ("private_like_failed",))
    elif transfer_report.non_submission_exact:
        errors = ("non_submission_exact",)
    return CompletionPrivateLikeReport(
        completion_transfer_report_hash=transfer_report.report_hash,
        private_like_report_hash=private_report.report_hash,
        private_like_report=private_report,
        passed=passed,
        non_submission_exact=transfer_report.non_submission_exact,
        errors=errors,
        metadata=dict(cfg.metadata),
    )


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompletionPrivateLikeError("metadata must be a mapping.")
    metadata = dict(value)
    try:
        _reject_forbidden_metadata(metadata)
    except Exception as exc:
        raise CompletionPrivateLikeError(str(exc)) from exc
    return metadata


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompletionPrivateLikeError(f"{field_name} must be non-empty.")
    return value


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "CompletionPrivateLikeConfig",
    "CompletionPrivateLikeError",
    "CompletionPrivateLikeReport",
    "evaluate_private_like_from_completions",
]

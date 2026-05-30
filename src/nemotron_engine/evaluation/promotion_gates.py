"""Final deterministic promotion gates for Pass 8."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .private_like import PrivateLikeReport
from .public_firewall import PublicFirewallReport
from .regression_report import RegressionReport
from .submission_exact import SubmissionExactReport
from .submission_exact import _validate_submission_exact_critical_fields
from .transfer_harness import TransferEvaluationReport


class PromotionGateError(ValueError):
    """Raised when promotion reports are inconsistent."""


class PromotionDecision(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"


@dataclass(frozen=True)
class PromotionGateConfig:
    required_hash_fields: tuple[str, ...] = (
        "training_plan_hash",
        "dataset_manifest_hash",
        "trace_manifest_hash",
        "lora_config_hash",
        "adapter_hash",
    )
    max_adapter_rank: int = 32
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.required_hash_fields, tuple) or any(not isinstance(item, str) or not item.strip() for item in self.required_hash_fields):
            raise PromotionGateError("required_hash_fields must be a tuple of non-empty strings.")
        if not isinstance(self.max_adapter_rank, int) or isinstance(self.max_adapter_rank, bool) or self.max_adapter_rank < 1:
            raise PromotionGateError("max_adapter_rank must be a positive integer.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise PromotionGateError("config_hash does not match promotion gate config payload.")


@dataclass(frozen=True)
class PromotionGateReport:
    decision: PromotionDecision | str
    passed: bool
    failure_reasons: tuple[str, ...]
    input_report_hashes: Mapping[str, str]
    required_hashes: Mapping[str, str | None]
    adapter_rank: int | None = None
    contamination_flags: tuple[str, ...] = ()
    unresolved_investigations: tuple[str, ...] = ()
    decision_hash: str = ""

    def __post_init__(self) -> None:
        decision = self.decision if isinstance(self.decision, PromotionDecision) else PromotionDecision(str(self.decision))
        failures = tuple(str(item) for item in self.failure_reasons)
        if self.passed and decision is not PromotionDecision.ALLOW:
            raise PromotionGateError("passed=True while decision is not allow.")
        if decision is PromotionDecision.ALLOW and self.passed is not True:
            raise PromotionGateError("decision=allow requires passed=True.")
        if decision is PromotionDecision.ALLOW and failures:
            raise PromotionGateError("decision=allow cannot include failure_reasons.")
        if decision is PromotionDecision.REJECT and not failures:
            raise PromotionGateError("decision=reject requires failure_reasons.")
        if self.adapter_rank is not None and (not isinstance(self.adapter_rank, int) or self.adapter_rank > 32):
            raise PromotionGateError("adapter_rank exceeds maximum allowed rank.")
        input_hashes = {str(key): str(value) for key, value in sorted(self.input_report_hashes.items())}
        required = {str(key): (None if value is None else str(value)) for key, value in sorted(self.required_hashes.items())}
        contamination = tuple(str(item) for item in self.contamination_flags)
        investigations = tuple(str(item) for item in self.unresolved_investigations)
        if decision is PromotionDecision.ALLOW and (contamination or investigations):
            raise PromotionGateError("allow decision cannot include contamination or investigations.")
        if contamination and "contamination" not in failures:
            raise PromotionGateError("contamination_flags require contamination failure reason.")
        if investigations and "unresolved_investigation" not in failures:
            raise PromotionGateError("unresolved investigations require unresolved_investigation failure reason.")
        missing_required = tuple(key for key, value in required.items() if not _non_empty(value))
        if missing_required:
            raise PromotionGateError(f"required_hashes contain empty values: {missing_required}")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "failure_reasons", failures)
        object.__setattr__(self, "input_report_hashes", input_hashes)
        object.__setattr__(self, "required_hashes", required)
        object.__setattr__(self, "contamination_flags", contamination)
        object.__setattr__(self, "unresolved_investigations", investigations)
        expected = _payload_hash(self, "decision_hash")
        if not self.decision_hash:
            object.__setattr__(self, "decision_hash", expected)
        elif self.decision_hash != expected:
            raise PromotionGateError("decision_hash does not match promotion report payload.")


def evaluate_promotion(
    *,
    submission_exact_report: SubmissionExactReport,
    transfer_report: TransferEvaluationReport,
    private_like_report: PrivateLikeReport,
    public_firewall_report: PublicFirewallReport,
    regression_report: RegressionReport,
    training_plan_hash: str | None,
    dataset_manifest_hash: str | None,
    trace_manifest_hash: str | None,
    lora_config_hash: str | None,
    adapter_hash: str | None,
    metadata: Mapping[str, Any] | None = None,
    config: PromotionGateConfig | None = None,
) -> PromotionGateReport:
    """Evaluate all promotion gates and return a deterministic decision report."""

    cfg = config or PromotionGateConfig()
    _validate_input_report_hashes(
        submission_exact_report=submission_exact_report,
        transfer_report=transfer_report,
        private_like_report=private_like_report,
        public_firewall_report=public_firewall_report,
        regression_report=regression_report,
    )
    submission_semantic_valid = _validate_submission_exact_report(submission_exact_report)
    meta = dict(metadata or {})
    required_hashes = {
        "training_plan_hash": training_plan_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "trace_manifest_hash": trace_manifest_hash,
        "lora_config_hash": lora_config_hash,
        "adapter_hash": adapter_hash,
    }
    failures: list[str] = []
    if submission_exact_report.passed is not True:
        failures.append("submission_exact_failed")
    elif not submission_semantic_valid:
        failures.append("submission_exact_failed")
    if transfer_report.passed is not True:
        failures.append("transfer_failed")
    if private_like_report.passed is not True:
        failures.append("private_like_failed")
    if public_firewall_report.decision != "allow":
        failures.append("public_firewall_blocked")
    if regression_report.passed is not True:
        failures.append("regression_failed")
    for field_name in cfg.required_hash_fields:
        if not _non_empty(required_hashes.get(field_name)):
            failures.append(f"missing_hash:{field_name}")
    adapter_rank = _metadata_adapter_rank(meta, fallback=submission_exact_report.adapter_rank)
    if adapter_rank is not None and adapter_rank > cfg.max_adapter_rank:
        failures.append("adapter_rank_exceeds_32")
    contamination_flags = _metadata_tuple(meta, "contamination_flags")
    if contamination_flags or meta.get("contaminated") is True:
        failures.append("contamination")
    investigations = _metadata_tuple(meta, "unresolved_investigations")
    if public_firewall_report.decision == "investigate":
        investigations = tuple(sorted(set(investigations + tuple(public_firewall_report.reasons or ("public_firewall_investigate",)))))
    if investigations:
        failures.append("unresolved_investigation")
    unique_failures = tuple(sorted(set(failures)))
    decision = PromotionDecision.ALLOW if not unique_failures else PromotionDecision.REJECT
    return PromotionGateReport(
        decision=decision,
        passed=decision is PromotionDecision.ALLOW,
        failure_reasons=unique_failures,
        input_report_hashes={
            "submission_exact_report": submission_exact_report.report_hash,
            "transfer_report": transfer_report.report_hash,
            "private_like_report": private_like_report.report_hash,
            "public_firewall_report": public_firewall_report.report_hash,
            "regression_report": regression_report.report_hash,
        },
        required_hashes=required_hashes,
        adapter_rank=adapter_rank if adapter_rank is None or adapter_rank <= cfg.max_adapter_rank else None,
        contamination_flags=contamination_flags,
        unresolved_investigations=investigations,
    )


def _validate_input_report_hashes(**reports: object) -> None:
    for name, report in reports.items():
        hash_field = "decision_hash" if isinstance(report, PromotionGateReport) else "report_hash"
        try:
            expected = _payload_hash(report, hash_field)
            observed = getattr(report, hash_field)
        except Exception as exc:
            raise PromotionGateError(f"input report hash mismatch: {name}") from exc
        if observed != expected:
            raise PromotionGateError(f"input report hash mismatch: {name}")


def _validate_submission_exact_report(report: SubmissionExactReport) -> bool:
    try:
        _validate_submission_exact_critical_fields(
            report.critical_serving_fields,
            adapter_checked=report.adapter_checked,
            adapter_rank=report.adapter_rank,
        )
    except Exception as exc:
        if report.passed is True:
            raise PromotionGateError("submission exact critical fields failed validation.") from exc
        return False
    return True


def _metadata_adapter_rank(metadata: Mapping[str, Any], *, fallback: int | None) -> int | None:
    for key in ("adapter_rank", "rank", "lora_rank"):
        if key in metadata:
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise PromotionGateError("adapter rank metadata must be an integer.")
            return value
    adapter_config = metadata.get("adapter_config")
    if isinstance(adapter_config, Mapping):
        for key in ("r", "rank", "lora_rank"):
            if key in adapter_config:
                value = adapter_config[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise PromotionGateError("adapter config rank must be an integer.")
                return value
        peft = adapter_config.get("peft_config")
        if isinstance(peft, Mapping) and "r" in peft:
            value = peft["r"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise PromotionGateError("adapter config rank must be an integer.")
            return value
    return fallback


def _metadata_tuple(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key, ())
    if value is None or value is False:
        return ()
    if value is True:
        return (key,)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list, set)):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "PromotionDecision",
    "PromotionGateConfig",
    "PromotionGateError",
    "PromotionGateReport",
    "evaluate_promotion",
]

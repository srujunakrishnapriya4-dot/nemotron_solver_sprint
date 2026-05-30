from __future__ import annotations

from dataclasses import fields, replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.evaluation.private_like import PrivateLikeReport
from nemotron_engine.evaluation.promotion_gates import (
    PromotionDecision,
    PromotionGateError,
    PromotionGateReport,
    evaluate_promotion,
)
from nemotron_engine.evaluation.public_firewall import PublicFirewallConfig, evaluate_public_firewall
from nemotron_engine.evaluation.regression_report import REQUIRED_REGRESSION_METRICS, build_regression_report
from nemotron_engine.evaluation.submission_exact import SubmissionExactConfig, validate_submission_exact_config
from nemotron_engine.evaluation.submission_exact import SubmissionExactReport
from nemotron_engine.evaluation.transfer_harness import TransferExample, TransferHarnessConfig, evaluate_transfer_slices
from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.runtime.serving_config import ServingConfig


def serving(**updates: object) -> ServingConfig:
    data = {
        "prompt_template_hash": "prompt",
        "tokenizer_hash": "tok",
        "model_hash": "model",
        "adapter_hash": "adapter",
    }
    data.update(updates)
    return ServingConfig(**data)


def reports(*, public_decision: str = "allow", private_passed: bool = True, transfer_passed: bool = True, submission_passed: bool = True):
    submission = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}))
    if not submission_passed:
        submission = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(temperature=0.1, strict_submission_mode=False), adapter_config={"r": 8}))
    transfer = evaluate_transfer_slices(
        [TransferExample("p1", "f", "p", "fmt", "private_like", "42", r"\boxed{42}" if transfer_passed else r"\boxed{0}", "integer", {})],
        TransferHarnessConfig(required_slices=("private_like",), allow_small_slices=True),
    )
    private = PrivateLikeReport(1, 1 if private_passed else 0, 1.0 if private_passed else 0.0, 1.0, 0.0 if private_passed else 1.0, 0.0, 0.0, 0, private_passed, () if private_passed else ("private_like_regression",), ("p1",))
    if public_decision == "allow":
        public = evaluate_public_firewall(PublicFirewallConfig(0.0, 0.01, 0.01, 0.0))
    elif public_decision == "investigate":
        public = evaluate_public_firewall(PublicFirewallConfig(0.02, 0.0, 0.0, 0.0))
    else:
        public = evaluate_public_firewall(PublicFirewallConfig(0.02, -0.01, 0.0, 0.0))
    metrics = {
        "overall_accuracy": 1.0,
        "private_like_accuracy": 1.0,
        "format_error_rate": 0.0,
        "extraction_error_rate": 0.0,
        "transfer_gap": 0.0,
        "stability_score": 1.0,
        "contamination_rate": 0.0,
    }
    regression = build_regression_report(metrics, metrics, {name: 0.0 for name in REQUIRED_REGRESSION_METRICS})
    return submission, transfer, private, public, regression


def promote(**kwargs):
    submission, transfer, private, public, regression = reports(**{k: v for k, v in kwargs.items() if k in {"public_decision", "private_passed", "transfer_passed", "submission_passed"}})
    metadata = kwargs.get("metadata", {})
    return evaluate_promotion(
        submission_exact_report=submission,
        transfer_report=transfer,
        private_like_report=private,
        public_firewall_report=public,
        regression_report=regression,
        training_plan_hash=kwargs.get("training_plan_hash", "plan"),
        dataset_manifest_hash=kwargs.get("dataset_manifest_hash", "dataset"),
        trace_manifest_hash=kwargs.get("trace_manifest_hash", "trace"),
        lora_config_hash=kwargs.get("lora_config_hash", "lora"),
        adapter_hash=kwargs.get("adapter_hash", "adapter"),
        metadata=metadata,
    )


def test_all_green_reports_allow_promotion_and_hash_deterministic() -> None:
    first = promote()
    second = promote()

    assert first.decision is PromotionDecision.ALLOW
    assert first.passed is True
    assert first.decision_hash == second.decision_hash


def test_individual_gate_failures_reject() -> None:
    assert promote(submission_passed=False).passed is False
    assert promote(transfer_passed=False).passed is False
    assert promote(private_passed=False).passed is False
    assert promote(public_decision="reject").passed is False
    assert promote(public_decision="investigate").passed is False


def test_missing_hashes_contamination_unresolved_and_rank_reject() -> None:
    with pytest.raises(PromotionGateError):
        promote(adapter_hash=None)
    assert "contamination" in promote(metadata={"contamination_flags": ("x",)}).failure_reasons
    assert "unresolved_investigation" in promote(metadata={"unresolved_investigations": ("review",)}).failure_reasons
    assert "adapter_rank_exceeds_32" in promote(metadata={"adapter_rank": 64}).failure_reasons


def test_report_hash_mismatch_rejected() -> None:
    submission, transfer, private, public, regression = reports()
    tampered = object.__new__(type(submission))
    object.__setattr__(tampered, "config_hash", submission.config_hash)
    object.__setattr__(tampered, "serving_config_hash", submission.serving_config_hash)
    object.__setattr__(tampered, "passed", True)
    object.__setattr__(tampered, "dry_run", submission.dry_run)
    object.__setattr__(tampered, "runtime_success", submission.runtime_success)
    object.__setattr__(tampered, "required_hashes_present", submission.required_hashes_present)
    object.__setattr__(tampered, "adapter_checked", submission.adapter_checked)
    object.__setattr__(tampered, "adapter_rank", submission.adapter_rank)
    object.__setattr__(tampered, "critical_serving_fields", submission.critical_serving_fields)
    object.__setattr__(tampered, "errors", submission.errors)
    object.__setattr__(tampered, "warnings", submission.warnings)
    object.__setattr__(tampered, "report_hash", "forged")

    with pytest.raises(PromotionGateError):
        evaluate_promotion(
            submission_exact_report=tampered,
            transfer_report=transfer,
            private_like_report=private,
            public_firewall_report=public,
            regression_report=regression,
            training_plan_hash="plan",
            dataset_manifest_hash="dataset",
            trace_manifest_hash="trace",
            lora_config_hash="lora",
            adapter_hash="adapter",
        )


def test_promotion_report_rejects_forged_and_inconsistent_objects() -> None:
    report = promote()

    with pytest.raises(PromotionGateError):
        replace(report, decision_hash="forged")
    with pytest.raises(PromotionGateError):
        PromotionGateReport("reject", True, (), {}, {})
    with pytest.raises(PromotionGateError):
        PromotionGateReport("allow", True, ("failure",), {}, {})
    with pytest.raises(PromotionGateError):
        PromotionGateReport("reject", False, ("rank",), {}, {}, adapter_rank=64)


def test_promotion_report_rejects_missing_hashes_and_unexplained_flags() -> None:
    with pytest.raises(PromotionGateError):
        PromotionGateReport("reject", False, ("missing_hash:adapter_hash",), {}, {"adapter_hash": None})
    with pytest.raises(PromotionGateError):
        PromotionGateReport("reject", False, ("other",), {}, {"adapter_hash": "a"}, contamination_flags=("x",))
    with pytest.raises(PromotionGateError):
        PromotionGateReport("reject", False, ("other",), {}, {"adapter_hash": "a"}, unresolved_investigations=("x",))
    with pytest.raises(PromotionGateError):
        PromotionGateReport("reject", False, (), {}, {"adapter_hash": "a"})


def test_promotion_rejects_semantically_forged_submission_report() -> None:
    submission, transfer, private, public, regression = reports()
    tampered = object.__new__(SubmissionExactReport)
    for field in fields(SubmissionExactReport):
        if field.name != "report_hash":
            object.__setattr__(tampered, field.name, getattr(submission, field.name))
    critical = dict(submission.critical_serving_fields)
    critical["temperature"] = 0.5
    object.__setattr__(tampered, "critical_serving_fields", critical)
    object.__setattr__(
        tampered,
        "report_hash",
        stable_hash({field.name: getattr(tampered, field.name) for field in fields(SubmissionExactReport) if field.name != "report_hash"}),
    )
    with pytest.raises(PromotionGateError):
        evaluate_promotion(
            submission_exact_report=tampered,
            transfer_report=transfer,
            private_like_report=private,
            public_firewall_report=public,
            regression_report=regression,
            training_plan_hash="plan",
            dataset_manifest_hash="dataset",
            trace_manifest_hash="trace",
            lora_config_hash="lora",
            adapter_hash="adapter",
        )

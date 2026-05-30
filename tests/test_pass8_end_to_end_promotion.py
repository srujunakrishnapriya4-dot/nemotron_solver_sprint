from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.evaluation.private_like import evaluate_private_like_gate
from nemotron_engine.evaluation.promotion_gates import PromotionDecision, evaluate_promotion
from nemotron_engine.evaluation.public_firewall import PublicFirewallConfig, evaluate_public_firewall
from nemotron_engine.evaluation.regression_report import REQUIRED_REGRESSION_METRICS, build_regression_report
from nemotron_engine.evaluation.submission_exact import SubmissionExactConfig, compare_serving_configs, validate_submission_exact_config
from nemotron_engine.evaluation.transfer_harness import TRANSFER_SLICE_NAMES, TransferExample, TransferHarnessConfig, evaluate_transfer_slices
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


def examples(wrong_slice: str | None = None) -> list[TransferExample]:
    result: list[TransferExample] = []
    for name in TRANSFER_SLICE_NAMES:
        for index in range(2):
            completion = r"\boxed{42}"
            if wrong_slice == name:
                completion = r"\boxed{0}"
            result.append(TransferExample(f"{name}-{index}", "fam", "prim", "fmt", name, "42", completion, "integer", {}))
    return result


def promotion(
    *,
    public_config: PublicFirewallConfig | None = None,
    wrong_slice: str | None = None,
    serving_config: ServingConfig | None = None,
    baseline_private: float = 1.0,
):
    exs = examples(wrong_slice)
    transfer = evaluate_transfer_slices(exs, TransferHarnessConfig(baseline_accuracies={"private_like": baseline_private}))
    private_like = evaluate_private_like_gate(exs, transfer)
    public = evaluate_public_firewall(public_config or PublicFirewallConfig(0.0, 0.01, 0.01, 0.0))
    current = {
        "overall_accuracy": transfer.overall_accuracy,
        "private_like_accuracy": private_like.accuracy,
        "format_error_rate": private_like.format_error_rate,
        "extraction_error_rate": private_like.extraction_error_rate,
        "transfer_gap": max(item.transfer_gap for item in transfer.slice_results),
        "stability_score": 1.0,
        "contamination_rate": 0.0,
    }
    baseline = dict(current)
    regression = build_regression_report(current, baseline, {name: 0.0 for name in REQUIRED_REGRESSION_METRICS})
    submission = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving_config or serving(), adapter_config={"r": 8}))
    return evaluate_promotion(
        submission_exact_report=submission,
        transfer_report=transfer,
        private_like_report=private_like,
        public_firewall_report=public,
        regression_report=regression,
        training_plan_hash="plan",
        dataset_manifest_hash="dataset",
        trace_manifest_hash="trace",
        lora_config_hash="lora",
        adapter_hash="adapter",
    )


def test_all_green_path_promotes_and_decision_hash_deterministic() -> None:
    first = promotion()
    second = promotion()

    assert first.decision is PromotionDecision.ALLOW
    assert first.decision_hash == second.decision_hash


def test_public_only_improvement_cannot_promote() -> None:
    report = promotion(public_config=PublicFirewallConfig(0.02, 0.0, 0.0, 0.0))

    assert report.passed is False
    assert "public_firewall_blocked" in report.failure_reasons


def test_private_like_regression_and_transfer_gap_failure_cannot_promote() -> None:
    private = promotion(wrong_slice="private_like")
    transfer = promotion(wrong_slice="hard_known")

    assert "private_like_failed" in private.failure_reasons
    assert "transfer_failed" in transfer.failure_reasons


def test_submission_exact_mismatch_cannot_promote() -> None:
    report = promotion(serving_config=serving(temperature=0.1, strict_submission_mode=False))

    assert "submission_exact_failed" in report.failure_reasons


def test_compare_serving_exact_mismatch_fails() -> None:
    report = compare_serving_configs(serving(), serving(num_samples=2, strict_submission_mode=False))

    assert report.passed is False

from __future__ import annotations

import dataclasses

import pytest

from nemotron_engine.evaluation import PromotionDecision, PromotionGateReport
from nemotron_engine.rehearsal import PromotionBridgeError, PromotionBridgeReport, build_promotion_rehearsal_report
from test_pass14_end_to_end_rehearsal import (
    make_adapter_bundle,
    make_config,
    make_inference_manifest,
    make_private_like_report,
    make_promotion_report,
    make_submission_exact,
    make_transfer_report,
)


def build_bridge(**overrides):
    bundle = overrides.pop("bundle", make_adapter_bundle())
    config = overrides.pop("config", make_config())
    payload = {
        "promotion_gate_report": make_promotion_report(bundle.adapter_hash),
        "submission_exact_report": make_submission_exact(),
        "transfer_report": make_transfer_report(),
        "private_like_report": make_private_like_report(),
        "inference_evaluation_manifest": make_inference_manifest(),
        "adapter_evidence_bundle": bundle,
        "rehearsal_config": config,
    }
    payload.update(overrides)
    return build_promotion_rehearsal_report(**payload)


def test_all_green_promotion_bridge_passes() -> None:
    assert build_bridge().allowed is True


def test_missing_submission_exact_hash_link_rejected() -> None:
    bundle = make_adapter_bundle()
    base = make_promotion_report(bundle.adapter_hash)
    links = dict(base.input_report_hashes)
    links.pop("submission_exact_report")
    report = dataclasses.replace(base, input_report_hashes=links, decision_hash="")
    assert "promotion_submission_exact_hash_missing" in build_bridge(bundle=bundle, promotion_gate_report=report).errors


def test_missing_transfer_hash_link_rejected() -> None:
    bundle = make_adapter_bundle()
    base = make_promotion_report(bundle.adapter_hash)
    links = dict(base.input_report_hashes)
    links.pop("transfer_report")
    report = dataclasses.replace(base, input_report_hashes=links, decision_hash="")
    assert "promotion_transfer_hash_missing" in build_bridge(bundle=bundle, promotion_gate_report=report).errors


def test_missing_private_like_hash_link_rejected_when_required() -> None:
    bundle = make_adapter_bundle()
    base = make_promotion_report(bundle.adapter_hash)
    links = dict(base.input_report_hashes)
    links.pop("private_like_report")
    report = dataclasses.replace(base, input_report_hashes=links, decision_hash="")
    assert "promotion_private_like_hash_missing" in build_bridge(bundle=bundle, promotion_gate_report=report).errors


def test_mismatched_hash_link_rejected() -> None:
    bundle = make_adapter_bundle()
    base = make_promotion_report(bundle.adapter_hash)
    links = dict(base.input_report_hashes)
    links["transfer_report"] = "wrong"
    report = dataclasses.replace(base, input_report_hashes=links, decision_hash="")
    assert "promotion_transfer_hash_mismatch" in build_bridge(bundle=bundle, promotion_gate_report=report).errors


def test_promotion_decision_reject_blocks() -> None:
    bundle = make_adapter_bundle()
    report = make_promotion_report(bundle.adapter_hash, passed=False)
    bridge = build_bridge(bundle=bundle, promotion_gate_report=report)
    assert "promotion_not_allow" in bridge.errors


def test_promotion_passed_false_blocks() -> None:
    bundle = make_adapter_bundle()
    report = PromotionGateReport(
        decision=PromotionDecision.REJECT,
        passed=False,
        failure_reasons=("manual_reject",),
        input_report_hashes={},
        required_hashes={"adapter_hash": bundle.adapter_hash},
    )
    assert "promotion_not_allow" in build_bridge(bundle=bundle, promotion_gate_report=report).errors


def test_submission_exact_failure_blocks() -> None:
    assert "submission_exact_failed" in build_bridge(submission_exact_report=make_submission_exact(False)).errors


def test_transfer_failure_blocks() -> None:
    assert "transfer_failed" in build_bridge(transfer_report=make_transfer_report(False)).errors


def test_private_like_failure_blocks_when_required() -> None:
    assert "private_like_failed" in build_bridge(private_like_report=make_private_like_report(False)).errors


def test_inference_evaluation_failure_blocks() -> None:
    assert "inference_evaluation_failed" in build_bridge(inference_evaluation_manifest=make_inference_manifest(False)).errors


def test_adapter_evidence_incomplete_blocks() -> None:
    bundle = dataclasses.replace(make_adapter_bundle(), evidence_complete=False, evidence_hash="")
    assert "adapter_evidence_incomplete" in build_bridge(bundle=bundle).errors


def test_forged_report_hash_rejected() -> None:
    data = dataclasses.asdict(build_bridge())
    data["report_hash"] = "forged"
    with pytest.raises(PromotionBridgeError):
        PromotionBridgeReport(**data)


def test_fake_metadata_claims_rejected() -> None:
    with pytest.raises(PromotionBridgeError):
        build_bridge(metadata={"leaderboard_ready": True})

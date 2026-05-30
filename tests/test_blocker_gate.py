from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.release import ReleaseCandidateReport
from nemotron_engine.rehearsal.rehearsal_manifest import FinalRehearsalManifest
from nemotron_engine.runbook import (
    BlockerGateConfig,
    BlockerGateError,
    BlockerGateReport,
    RunbookConfig,
    build_default_command_manifest,
    build_operator_checklist,
    evaluate_blocker_gate,
)


def final_manifest(*, ready: bool = True) -> FinalRehearsalManifest:
    payload = {
        "rehearsal_config_hash": "r" * 64,
        "adapter_evidence_hash": "a" * 64,
        "promotion_bridge_hash": "b" * 64,
        "package_rehearsal_hash": "c" * 64,
        "runtime_rehearsal_hash": "d" * 64,
        "submission_rehearsal_hash": "e" * 64,
        "dry_run": True,
    }
    return FinalRehearsalManifest(
        manifest_id=stable_hash(payload),
        rehearsal_config_hash=payload["rehearsal_config_hash"],
        adapter_evidence_hash=payload["adapter_evidence_hash"],
        promotion_bridge_hash=payload["promotion_bridge_hash"],
        package_rehearsal_hash=payload["package_rehearsal_hash"],
        runtime_rehearsal_hash=payload["runtime_rehearsal_hash"],
        submission_rehearsal_hash=payload["submission_rehearsal_hash"],
        inference_evaluation_hash=None,
        smoke_handoff_hash=None,
        release_candidate_hash=None,
        dry_run=True,
        package_built=False,
        runtime_loaded=False,
        submitted=False,
        ready_for_external_submission=ready,
        package_passed=ready,
        runtime_passed=ready,
        submission_passed=ready,
        allow_warnings=True,
        errors=() if ready else ("not_ready",),
    )


def config(manifest_hash: str | None = None) -> RunbookConfig:
    return RunbookConfig(
        runbook_id="pass15-final",
        mode="external_submission_runbook",
        require_pass14_lock=True,
        require_final_rehearsal=True,
        require_human_approval=True,
        allow_external_submission_instructions=True,
        allow_post_submission_record=True,
        accepted_suite_summary="891 passed, 2 skipped",
        checkpoint_zip_path=None,
        final_rehearsal_manifest_hash=manifest_hash or final_manifest().manifest_hash,
        seed=15,
    )


def release(ready: bool) -> ReleaseCandidateReport:
    return ReleaseCandidateReport(
        candidate_id="candidate",
        repository_root=None,
        locked_registry_hash="locked",
        system_audit_hash="audit",
        scoped_suite_result_summary="891 passed, 2 skipped",
        ready=ready,
        release_level="submission_dry_run_candidate",
        blockers=() if ready else ("not_ready",),
        system_audit_passed=ready,
    )


def test_all_green_gate_unblocked() -> None:
    final = final_manifest()
    cfg = config(final.manifest_hash)
    report = evaluate_blocker_gate(cfg, build_default_command_manifest(cfg), build_operator_checklist(cfg, checked=True), final)
    assert not report.blocked


def test_unchecked_checklist_blocks() -> None:
    final = final_manifest()
    cfg = config(final.manifest_hash)
    report = evaluate_blocker_gate(cfg, build_default_command_manifest(cfg), build_operator_checklist(cfg, checked=False), final)
    assert report.blocked
    assert "operator_checklist_incomplete" in report.blockers


def test_final_rehearsal_missing_or_failing_blocks() -> None:
    final = final_manifest()
    cfg = config(final.manifest_hash)
    missing = evaluate_blocker_gate(cfg, build_default_command_manifest(cfg), build_operator_checklist(cfg, checked=True), None)
    assert "final_rehearsal_manifest_missing" in missing.blockers
    failing = final_manifest(ready=False)
    cfg2 = config(failing.manifest_hash)
    report = evaluate_blocker_gate(cfg2, build_default_command_manifest(cfg2), build_operator_checklist(cfg2, checked=True), failing)
    assert "final_rehearsal_not_ready_for_manual_consideration" in report.blockers


def test_release_candidate_not_ready_blocks_when_required() -> None:
    final = final_manifest()
    cfg = config(final.manifest_hash)
    report = evaluate_blocker_gate(
        cfg,
        build_default_command_manifest(cfg),
        build_operator_checklist(cfg, checked=True),
        final,
        release_candidate_report=release(False),
        gate_config=BlockerGateConfig(require_release_candidate=True),
    )
    assert "release_candidate_not_ready" in report.blockers


def test_report_state_invariants_and_forged_hash_rejected() -> None:
    final = final_manifest()
    cfg = config(final.manifest_hash)
    report = evaluate_blocker_gate(cfg, build_default_command_manifest(cfg), build_operator_checklist(cfg, checked=True), final)
    with pytest.raises(BlockerGateError):
        BlockerGateReport(
            runbook_config_hash=cfg.config_hash,
            command_manifest_hash="c",
            checklist_hash="k",
            final_rehearsal_manifest_hash=None,
            release_candidate_hash=None,
            blocked=False,
            blockers=("x",),
        )
    with pytest.raises(BlockerGateError):
        BlockerGateReport(
            runbook_config_hash=cfg.config_hash,
            command_manifest_hash="c",
            checklist_hash="k",
            final_rehearsal_manifest_hash=None,
            release_candidate_hash=None,
            blocked=True,
            blockers=(),
        )
    assert BlockerGateReport(
        runbook_config_hash=cfg.config_hash,
        command_manifest_hash="c",
        checklist_hash="k",
        final_rehearsal_manifest_hash=None,
        release_candidate_hash=None,
        blocked=True,
        blockers=(),
        metadata={"reason": "manual hold"},
    )
    with pytest.raises(BlockerGateError):
        replace(report, report_hash="forged")


def test_fake_metadata_rejected() -> None:
    final = final_manifest()
    cfg = config(final.manifest_hash)
    with pytest.raises(BlockerGateError):
        evaluate_blocker_gate(cfg, build_default_command_manifest(cfg), build_operator_checklist(cfg, checked=True), final, metadata={"kaggle_success": True})

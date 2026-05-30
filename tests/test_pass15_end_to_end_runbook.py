from __future__ import annotations

from pathlib import Path

import pytest

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.rehearsal.rehearsal_manifest import FinalRehearsalManifest
from nemotron_engine.runbook import (
    RunbookConfig,
    RunbookConfigError,
    SubmissionRecord,
    build_default_command_manifest,
    build_default_failure_recovery_plan,
    build_final_operations_manifest,
    build_human_approval_statement,
    build_manual_submission_record,
    build_operator_checklist,
    build_reproducibility_manifest,
    evaluate_blocker_gate,
)
from nemotron_engine.runbook.human_approval import REQUIRED_APPROVAL_PHRASE
from nemotron_engine.runbook.reproducibility_manifest import PASS_LOCK_KEYS


def final_manifest() -> FinalRehearsalManifest:
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
        ready_for_external_submission=True,
        package_passed=True,
        runtime_passed=True,
        submission_passed=True,
        allow_warnings=True,
    )


def config(final_hash: str) -> RunbookConfig:
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
        final_rehearsal_manifest_hash=final_hash,
        seed=15,
    )


def test_end_to_end_runbook_is_deterministic_no_submission_and_no_command_execution(tmp_path: Path) -> None:
    final = final_manifest()
    cfg = config(final.manifest_hash)
    commands = build_default_command_manifest(cfg)
    checklist = build_operator_checklist(cfg, checked=True)
    gate = evaluate_blocker_gate(cfg, commands, checklist, final)
    repro = build_reproducibility_manifest(
        cfg,
        commands,
        checklist,
        gate,
        pass_lock_summaries={key: "LOCKED" for key in PASS_LOCK_KEYS},
        environment_summary={"python": "3.12"},
    )
    approval = build_human_approval_statement(
        approver="operator",
        approval_text=f"Approved. {REQUIRED_APPROVAL_PHRASE}",
        runbook_config=cfg,
        blocker_gate_report=gate,
        reproducibility_manifest=repro,
        acknowledged_no_automation_submission=True,
        acknowledged_dry_run_not_score=True,
        acknowledged_manual_external_submission_only=True,
    )
    no_submission = build_manual_submission_record(
        manual_submission_performed=False,
        human_approval=approval,
        reproducibility_manifest=repro,
    )
    recovery = build_default_failure_recovery_plan(cfg)
    final_ops = build_final_operations_manifest(
        cfg,
        commands,
        checklist,
        gate,
        repro,
        recovery,
        human_approval=approval,
        submission_record=no_submission,
    )
    again = build_final_operations_manifest(
        cfg,
        commands,
        checklist,
        gate,
        repro,
        recovery,
        human_approval=approval,
        submission_record=no_submission,
    )
    marker = tmp_path / "not_executed.txt"
    assert not marker.exists()
    assert gate.blocked is False
    assert approval.approved
    assert no_submission.manual_submission_performed is False
    assert final_ops.ready_for_manual_submission
    assert final_ops.manual_submission_recorded is False
    assert final_ops.operations_hash == again.operations_hash


def test_fake_kaggle_leaderboard_metadata_blocks() -> None:
    final = final_manifest()
    with pytest.raises(RunbookConfigError):
        RunbookConfig(
            runbook_id="pass15-final",
            mode="external_submission_runbook",
            require_pass14_lock=True,
            require_final_rehearsal=True,
            require_human_approval=True,
            allow_external_submission_instructions=True,
            allow_post_submission_record=True,
            accepted_suite_summary="891 passed, 2 skipped",
            checkpoint_zip_path=None,
            final_rehearsal_manifest_hash=final.manifest_hash,
            seed=15,
            metadata={"leaderboard_ready": True},
        )


def test_manual_submission_record_is_recordkeeping_only() -> None:
    final = final_manifest()
    cfg = config(final.manifest_hash)
    commands = build_default_command_manifest(cfg)
    checklist = build_operator_checklist(cfg, checked=True)
    gate = evaluate_blocker_gate(cfg, commands, checklist, final)
    repro = build_reproducibility_manifest(cfg, commands, checklist, gate, pass_lock_summaries={key: "LOCKED" for key in PASS_LOCK_KEYS})
    approval = build_human_approval_statement(
        approver="operator",
        approval_text=f"Approved. {REQUIRED_APPROVAL_PHRASE}",
        runbook_config=cfg,
        blocker_gate_report=gate,
        reproducibility_manifest=repro,
        acknowledged_no_automation_submission=True,
        acknowledged_dry_run_not_score=True,
        acknowledged_manual_external_submission_only=True,
    )
    record = build_manual_submission_record(
        manual_submission_performed=True,
        human_approval=approval,
        reproducibility_manifest=repro,
        submitted_by="operator",
        external_platform="kaggle",
        artifact_hash="a" * 64,
        submission_reference="human-visible-ref",
        metadata={"human_recorded": True},
    )
    assert isinstance(record, SubmissionRecord)
    assert record.manual_submission_performed
    assert record.external_platform == "kaggle"
    assert record.record_hash

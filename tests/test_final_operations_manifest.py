from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.runbook import (
    BlockerGateReport,
    FinalOperationsManifest,
    FinalOperationsManifestError,
    HumanApprovalStatement,
    ReproducibilityManifest,
    RunbookConfig,
    SubmissionRecord,
    build_default_command_manifest,
    build_default_failure_recovery_plan,
    build_final_operations_manifest,
    build_operator_checklist,
    build_reproducibility_manifest,
)
from nemotron_engine.runbook.human_approval import REQUIRED_APPROVAL_PHRASE
from nemotron_engine.runbook.reproducibility_manifest import PASS_LOCK_KEYS


def config() -> RunbookConfig:
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
        final_rehearsal_manifest_hash="f" * 64,
        seed=15,
    )


def gate(blocked: bool = False, *, command_manifest_hash: str | None = None, checklist_hash: str | None = None) -> BlockerGateReport:
    return BlockerGateReport(
        runbook_config_hash=config().config_hash,
        command_manifest_hash=command_manifest_hash or "c" * 64,
        checklist_hash=checklist_hash or "k" * 64,
        final_rehearsal_manifest_hash="f" * 64,
        release_candidate_hash=None,
        blocked=blocked,
        blockers=("blocker",) if blocked else (),
        metadata={"reason": "manual hold"} if blocked else {},
    )


def repro(
    *,
    command_manifest_hash: str | None = None,
    checklist_hash: str | None = None,
    blocker_gate_hash: str | None = None,
) -> ReproducibilityManifest:
    return ReproducibilityManifest(
        runbook_config_hash=config().config_hash,
        checkpoint_zip_path=None,
        checkpoint_zip_hash=None,
        scoped_suite_summary="891 passed, 2 skipped",
        pass_lock_summaries={key: "LOCKED" for key in PASS_LOCK_KEYS},
        final_rehearsal_manifest_hash="f" * 64,
        release_candidate_hash=None,
        environment_summary={},
        command_manifest_hash=command_manifest_hash or build_default_command_manifest(config()).manifest_hash,
        checklist_hash=checklist_hash or build_operator_checklist(config(), checked=True).checklist_hash,
        blocker_gate_hash=blocker_gate_hash or gate().report_hash,
    )


def approval(
    *,
    blocker_gate_hash: str | None = None,
    reproducibility_manifest_hash: str | None = None,
) -> HumanApprovalStatement:
    return HumanApprovalStatement(
        approver="operator",
        approval_text=f"Approved. {REQUIRED_APPROVAL_PHRASE}",
        acknowledged_no_automation_submission=True,
        acknowledged_dry_run_not_score=True,
        acknowledged_manual_external_submission_only=True,
        runbook_config_hash=config().config_hash,
        blocker_gate_hash=blocker_gate_hash or gate().report_hash,
        reproducibility_manifest_hash=reproducibility_manifest_hash or repro().manifest_hash,
        approved=True,
    )


def components():
    cfg = config()
    commands = build_default_command_manifest(cfg)
    checklist = build_operator_checklist(cfg, checked=True)
    blocker = gate(False, command_manifest_hash=commands.manifest_hash, checklist_hash=checklist.checklist_hash)
    reproducibility = build_reproducibility_manifest(
        cfg,
        commands,
        checklist,
        blocker,
        pass_lock_summaries={key: "LOCKED" for key in PASS_LOCK_KEYS},
    )
    recovery = build_default_failure_recovery_plan(cfg)
    human = approval(blocker_gate_hash=blocker.report_hash, reproducibility_manifest_hash=reproducibility.manifest_hash)
    return cfg, commands, checklist, blocker, reproducibility, recovery, human


def built_manifest(**kwargs: object):
    cfg, commands, checklist, blocker, reproducibility, recovery, human = components()
    params = {
        "runbook_config": cfg,
        "command_manifest": commands,
        "operator_checklist": checklist,
        "blocker_gate_report": blocker,
        "reproducibility_manifest": reproducibility,
        "failure_recovery_plan": recovery,
        "human_approval": human,
    }
    params.update(kwargs)
    return build_final_operations_manifest(**params)


def test_all_green_ready_for_manual_submission_passes() -> None:
    manifest = built_manifest()
    assert manifest.ready_for_manual_submission
    assert not manifest.manual_submission_recorded


def test_ready_true_without_human_approval_rejected() -> None:
    payload = {
        "runbook_config_hash": "r",
        "command_manifest_hash": "c",
        "checklist_hash": "k",
        "blocker_gate_hash": "b",
        "reproducibility_manifest_hash": "m",
        "human_approval_hash": None,
        "submission_record_hash": None,
        "failure_recovery_hash": "f",
        "ready_for_manual_submission": True,
        "manual_submission_recorded": False,
        "blocked": False,
        "blockers": (),
    }
    with pytest.raises(FinalOperationsManifestError):
        FinalOperationsManifest(
            operations_id=stable_hash(payload),
            runbook_config_hash=payload["runbook_config_hash"],
            command_manifest_hash=payload["command_manifest_hash"],
            checklist_hash=payload["checklist_hash"],
            blocker_gate_hash=payload["blocker_gate_hash"],
            reproducibility_manifest_hash=payload["reproducibility_manifest_hash"],
            human_approval_hash=None,
            submission_record_hash=None,
            failure_recovery_hash=payload["failure_recovery_hash"],
            ready_for_manual_submission=True,
            manual_submission_recorded=False,
            blocked=False,
            blockers=(),
            warnings=(),
        )


def test_ready_true_with_blocked_gate_or_blockers_rejected() -> None:
    with pytest.raises(FinalOperationsManifestError):
        FinalOperationsManifest(
            operations_id="x",
            runbook_config_hash="r",
            command_manifest_hash="c",
            checklist_hash="k",
            blocker_gate_hash="b",
            reproducibility_manifest_hash="m",
            human_approval_hash="h",
            submission_record_hash=None,
            failure_recovery_hash="f",
            ready_for_manual_submission=True,
            manual_submission_recorded=False,
            blocked=True,
            blockers=("blocker",),
            warnings=(),
        )
    with pytest.raises(FinalOperationsManifestError):
        FinalOperationsManifest(
            operations_id="x",
            runbook_config_hash="r",
            command_manifest_hash="c",
            checklist_hash="k",
            blocker_gate_hash="b",
            reproducibility_manifest_hash="m",
            human_approval_hash="h",
            submission_record_hash=None,
            failure_recovery_hash="f",
            ready_for_manual_submission=True,
            manual_submission_recorded=False,
            blocked=False,
            blockers=("blocker",),
            warnings=(),
        )


def test_manual_submission_recorded_only_with_manual_record() -> None:
    with pytest.raises(FinalOperationsManifestError):
        FinalOperationsManifest(
            operations_id="x",
            runbook_config_hash="r",
            command_manifest_hash="c",
            checklist_hash="k",
            blocker_gate_hash="b",
            reproducibility_manifest_hash="m",
            human_approval_hash="h",
            submission_record_hash=None,
            failure_recovery_hash="f",
            ready_for_manual_submission=False,
            manual_submission_recorded=True,
            blocked=False,
            blockers=(),
            warnings=(),
        )
    rec_id = stable_hash(
        {
            "manual_submission_performed": True,
            "human_approval_hash": components()[-1].approval_hash,
            "reproducibility_manifest_hash": components()[4].manifest_hash,
            "submission_reference": "ref",
            "artifact_hash": "a" * 64,
        }
    )
    cfg, commands, checklist, blocker, reproducibility, recovery, human = components()
    record = SubmissionRecord(
        record_id=rec_id,
        manual_submission_performed=True,
        submitted_by="operator",
        external_platform="kaggle",
        artifact_hash="a" * 64,
        submission_reference="ref",
        submitted_at_text="human time",
        public_score_text=None,
        private_score_text=None,
        notes=None,
        human_approval_hash=human.approval_hash,
        reproducibility_manifest_hash=reproducibility.manifest_hash,
    )
    manifest = build_final_operations_manifest(cfg, commands, checklist, blocker, reproducibility, recovery, human_approval=human, submission_record=record)
    assert manifest.manual_submission_recorded


def test_direct_ready_true_with_arbitrary_approval_hash_rejected() -> None:
    payload = {
        "runbook_config_hash": "r",
        "command_manifest_hash": "c",
        "checklist_hash": "k",
        "blocker_gate_hash": "b",
        "reproducibility_manifest_hash": "m",
        "human_approval_hash": "h",
        "submission_record_hash": None,
        "failure_recovery_hash": "f",
        "ready_for_manual_submission": True,
        "manual_submission_recorded": False,
        "blocked": False,
        "blockers": (),
    }
    with pytest.raises(FinalOperationsManifestError):
        FinalOperationsManifest(
            operations_id=stable_hash(payload),
            runbook_config_hash=payload["runbook_config_hash"],
            command_manifest_hash=payload["command_manifest_hash"],
            checklist_hash=payload["checklist_hash"],
            blocker_gate_hash=payload["blocker_gate_hash"],
            reproducibility_manifest_hash=payload["reproducibility_manifest_hash"],
            human_approval_hash=payload["human_approval_hash"],
            submission_record_hash=None,
            failure_recovery_hash=payload["failure_recovery_hash"],
            ready_for_manual_submission=True,
            manual_submission_recorded=False,
            blocked=False,
            blockers=(),
            warnings=(),
        )


def test_builder_rejects_mismatched_human_approval_hashes() -> None:
    cfg, commands, checklist, blocker, reproducibility, recovery, _ = components()
    bad_blocker_approval = approval(blocker_gate_hash="x" * 64, reproducibility_manifest_hash=reproducibility.manifest_hash)
    with pytest.raises(FinalOperationsManifestError):
        build_final_operations_manifest(cfg, commands, checklist, blocker, reproducibility, recovery, human_approval=bad_blocker_approval)
    bad_repro_approval = approval(blocker_gate_hash=blocker.report_hash, reproducibility_manifest_hash="x" * 64)
    with pytest.raises(FinalOperationsManifestError):
        build_final_operations_manifest(cfg, commands, checklist, blocker, reproducibility, recovery, human_approval=bad_repro_approval)


def test_builder_rejects_mismatched_reproducibility_blocker_hash() -> None:
    cfg, commands, checklist, blocker, _, recovery, human = components()
    bad_repro = repro(
        command_manifest_hash=commands.manifest_hash,
        checklist_hash=checklist.checklist_hash,
        blocker_gate_hash="x" * 64,
    )
    bad_human = approval(blocker_gate_hash=blocker.report_hash, reproducibility_manifest_hash=bad_repro.manifest_hash)
    with pytest.raises(FinalOperationsManifestError):
        build_final_operations_manifest(cfg, commands, checklist, blocker, bad_repro, recovery, human_approval=bad_human)


def test_builder_rejects_submission_record_approval_or_repro_hash_mismatch() -> None:
    cfg, commands, checklist, blocker, reproducibility, recovery, human = components()
    bad_approval_record_id = stable_hash(
        {
            "manual_submission_performed": True,
            "human_approval_hash": "x" * 64,
            "reproducibility_manifest_hash": reproducibility.manifest_hash,
            "submission_reference": "ref",
            "artifact_hash": "a" * 64,
        }
    )
    bad_approval_record = SubmissionRecord(
        record_id=bad_approval_record_id,
        manual_submission_performed=True,
        submitted_by="operator",
        external_platform="kaggle",
        artifact_hash="a" * 64,
        submission_reference="ref",
        submitted_at_text=None,
        public_score_text=None,
        private_score_text=None,
        notes=None,
        human_approval_hash="x" * 64,
        reproducibility_manifest_hash=reproducibility.manifest_hash,
    )
    with pytest.raises(FinalOperationsManifestError):
        build_final_operations_manifest(cfg, commands, checklist, blocker, reproducibility, recovery, human_approval=human, submission_record=bad_approval_record)
    bad_repro_record_id = stable_hash(
        {
            "manual_submission_performed": True,
            "human_approval_hash": human.approval_hash,
            "reproducibility_manifest_hash": "x" * 64,
            "submission_reference": "ref",
            "artifact_hash": "a" * 64,
        }
    )
    bad_repro_record = SubmissionRecord(
        record_id=bad_repro_record_id,
        manual_submission_performed=True,
        submitted_by="operator",
        external_platform="kaggle",
        artifact_hash="a" * 64,
        submission_reference="ref",
        submitted_at_text=None,
        public_score_text=None,
        private_score_text=None,
        notes=None,
        human_approval_hash=human.approval_hash,
        reproducibility_manifest_hash="x" * 64,
    )
    with pytest.raises(FinalOperationsManifestError):
        build_final_operations_manifest(cfg, commands, checklist, blocker, reproducibility, recovery, human_approval=human, submission_record=bad_repro_record)


def test_fake_metadata_and_forged_ids_rejected() -> None:
    with pytest.raises(FinalOperationsManifestError):
        built_manifest(metadata={"leaderboard_ready": True})
    manifest = built_manifest()
    with pytest.raises(FinalOperationsManifestError):
        replace(manifest, operations_id="forged")
    with pytest.raises(FinalOperationsManifestError):
        replace(manifest, operations_hash="forged")


def test_ready_for_manual_submission_does_not_mean_submitted_or_leaderboard_ready() -> None:
    manifest = built_manifest()
    assert manifest.ready_for_manual_submission
    assert manifest.manual_submission_recorded is False
    assert manifest.submission_record_hash is None
    assert "leaderboard" not in str(manifest.metadata).lower()

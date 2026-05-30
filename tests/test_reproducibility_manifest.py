from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.runbook import (
    BlockerGateReport,
    ReproducibilityManifest,
    ReproducibilityManifestError,
    RunbookConfig,
    build_default_command_manifest,
    build_operator_checklist,
    build_reproducibility_manifest,
)
from nemotron_engine.runbook.reproducibility_manifest import PASS_LOCK_KEYS


def pass_locks() -> dict[str, str]:
    return {key: "LOCKED" for key in PASS_LOCK_KEYS}


def manifest(**kwargs: object) -> ReproducibilityManifest:
    payload = {
        "runbook_config_hash": "r" * 64,
        "checkpoint_zip_path": None,
        "checkpoint_zip_hash": None,
        "scoped_suite_summary": "891 passed, 2 skipped",
        "pass_lock_summaries": pass_locks(),
        "final_rehearsal_manifest_hash": "f" * 64,
        "release_candidate_hash": None,
        "environment_summary": {"python": "3.12"},
        "command_manifest_hash": "c" * 64,
        "checklist_hash": "k" * 64,
        "blocker_gate_hash": "b" * 64,
        "metadata": {},
    }
    payload.update(kwargs)
    return ReproducibilityManifest(**payload)


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


def bound_components() -> tuple[RunbookConfig, object, object, BlockerGateReport]:
    cfg = config()
    commands = build_default_command_manifest(cfg)
    checklist = build_operator_checklist(cfg, checked=True)
    gate = BlockerGateReport(
        runbook_config_hash=cfg.config_hash,
        command_manifest_hash=commands.manifest_hash,
        checklist_hash=checklist.checklist_hash,
        final_rehearsal_manifest_hash=cfg.final_rehearsal_manifest_hash,
        release_candidate_hash=None,
        blocked=False,
        blockers=(),
    )
    return cfg, commands, checklist, gate


def test_valid_manifest_passes() -> None:
    assert manifest().manifest_hash


def test_valid_builder_path_passes() -> None:
    cfg, commands, checklist, gate = bound_components()
    built = build_reproducibility_manifest(cfg, commands, checklist, gate, pass_lock_summaries=pass_locks())
    assert built.blocker_gate_hash == gate.report_hash


def test_builder_rejects_blocker_gate_runbook_hash_mismatch() -> None:
    cfg, commands, checklist, gate = bound_components()
    bad_gate = BlockerGateReport(
        runbook_config_hash="x" * 64,
        command_manifest_hash=commands.manifest_hash,
        checklist_hash=checklist.checklist_hash,
        final_rehearsal_manifest_hash=cfg.final_rehearsal_manifest_hash,
        release_candidate_hash=None,
        blocked=False,
        blockers=(),
    )
    with pytest.raises(ReproducibilityManifestError):
        build_reproducibility_manifest(cfg, commands, checklist, bad_gate, pass_lock_summaries=pass_locks())


def test_builder_rejects_blocker_gate_command_hash_mismatch() -> None:
    cfg, commands, checklist, _ = bound_components()
    bad_gate = BlockerGateReport(
        runbook_config_hash=cfg.config_hash,
        command_manifest_hash="x" * 64,
        checklist_hash=checklist.checklist_hash,
        final_rehearsal_manifest_hash=cfg.final_rehearsal_manifest_hash,
        release_candidate_hash=None,
        blocked=False,
        blockers=(),
    )
    with pytest.raises(ReproducibilityManifestError):
        build_reproducibility_manifest(cfg, commands, checklist, bad_gate, pass_lock_summaries=pass_locks())


def test_builder_rejects_blocker_gate_checklist_hash_mismatch() -> None:
    cfg, commands, checklist, _ = bound_components()
    bad_gate = BlockerGateReport(
        runbook_config_hash=cfg.config_hash,
        command_manifest_hash=commands.manifest_hash,
        checklist_hash="x" * 64,
        final_rehearsal_manifest_hash=cfg.final_rehearsal_manifest_hash,
        release_candidate_hash=None,
        blocked=False,
        blockers=(),
    )
    with pytest.raises(ReproducibilityManifestError):
        build_reproducibility_manifest(cfg, commands, checklist, bad_gate, pass_lock_summaries=pass_locks())


def test_forged_manifest_hash_rejected() -> None:
    with pytest.raises(ReproducibilityManifestError):
        replace(manifest(), manifest_hash="forged")


def test_missing_pass_1_to_14_lock_summary_rejected() -> None:
    locks = pass_locks()
    locks.pop("pass_14")
    with pytest.raises(ReproducibilityManifestError):
        manifest(pass_lock_summaries=locks)


def test_missing_suite_summary_rejected() -> None:
    with pytest.raises(ReproducibilityManifestError):
        manifest(scoped_suite_summary="")


def test_checkpoint_path_without_hash_rejected_when_required() -> None:
    with pytest.raises(ReproducibilityManifestError):
        manifest(checkpoint_zip_path="artifacts/final.zip", checkpoint_zip_hash=None)


@pytest.mark.parametrize("metadata", [{"training_success": True}, {"submission_success": True}, {"leaderboard_ready": True}, {"score": "guaranteed"}])
def test_fake_metadata_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(ReproducibilityManifestError):
        manifest(metadata=metadata)

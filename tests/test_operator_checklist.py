from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.runbook import (
    ChecklistItem,
    OperatorChecklist,
    OperatorChecklistError,
    RunbookConfig,
    build_operator_checklist,
)
from nemotron_engine.runbook.operator_checklist import REQUIRED_CHECKLIST_DESCRIPTIONS


def make_config() -> RunbookConfig:
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
        final_rehearsal_manifest_hash="a" * 64,
        seed=15,
    )


def item(**kwargs: object) -> ChecklistItem:
    payload = {
        "item_id": "item",
        "description": "Pass 14 lock audit returned LOCK.",
        "required": True,
        "checked": True,
        "evidence_hash": "a" * 64,
        "blocker_if_unchecked": True,
        "metadata": {},
    }
    payload.update(kwargs)
    return ChecklistItem(**payload)


def test_default_checklist_contains_all_required_items() -> None:
    checklist = build_operator_checklist(make_config(), checked=True)
    descriptions = {entry.description for entry in checklist.items}
    assert set(REQUIRED_CHECKLIST_DESCRIPTIONS).issubset(descriptions)
    assert checklist.all_required_checked


def test_unchecked_required_item_creates_blocker() -> None:
    checklist = build_operator_checklist(make_config(), checked=False)
    assert not checklist.all_required_checked
    assert checklist.blockers


def test_direct_operator_checklist_missing_required_items_rejected() -> None:
    with pytest.raises(OperatorChecklistError):
        OperatorChecklist(items=(item(),), runbook_config_hash=make_config().config_hash, all_required_checked=True, blockers=())


def test_all_required_checked_forged_true_rejected() -> None:
    unchecked = item(checked=False, evidence_hash=None)
    with pytest.raises(OperatorChecklistError):
        OperatorChecklist(items=(unchecked,), runbook_config_hash=make_config().config_hash, all_required_checked=True, blockers=())


def test_checked_required_item_without_evidence_rejected_unless_informational() -> None:
    with pytest.raises(OperatorChecklistError):
        item(evidence_hash=None)
    assert item(evidence_hash=None, metadata={"informational": True})


def test_forged_item_and_checklist_hash_rejected() -> None:
    with pytest.raises(OperatorChecklistError):
        replace(item(), item_hash="forged")
    checklist = build_operator_checklist(make_config(), checked=True)
    with pytest.raises(OperatorChecklistError):
        replace(checklist, checklist_hash="forged")


def test_fake_metadata_rejected() -> None:
    with pytest.raises(OperatorChecklistError):
        item(metadata={"leaderboard_ready": True})

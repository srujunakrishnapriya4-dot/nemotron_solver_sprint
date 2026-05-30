from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.runbook import HumanApprovalError, HumanApprovalStatement
from nemotron_engine.runbook.human_approval import REQUIRED_APPROVAL_PHRASE


def approval(**kwargs: object) -> HumanApprovalStatement:
    payload = {
        "approver": "operator",
        "approval_text": f"Approved. {REQUIRED_APPROVAL_PHRASE}",
        "acknowledged_no_automation_submission": True,
        "acknowledged_dry_run_not_score": True,
        "acknowledged_manual_external_submission_only": True,
        "runbook_config_hash": "r" * 64,
        "blocker_gate_hash": "b" * 64,
        "reproducibility_manifest_hash": "m" * 64,
        "approved": True,
        "metadata": {},
    }
    payload.update(kwargs)
    return HumanApprovalStatement(**payload)


def test_valid_approval_passes() -> None:
    assert approval().approval_hash


def test_missing_acknowledgement_rejected() -> None:
    with pytest.raises(HumanApprovalError):
        approval(acknowledged_dry_run_not_score=False)


def test_missing_exact_required_phrase_rejected() -> None:
    with pytest.raises(HumanApprovalError):
        approval(approval_text="Approved, but without exact phrase.")


def test_approved_true_without_approver_rejected() -> None:
    with pytest.raises(HumanApprovalError):
        approval(approver="")


def test_forged_approval_hash_rejected() -> None:
    with pytest.raises(HumanApprovalError):
        replace(approval(), approval_hash="forged")


@pytest.mark.parametrize("metadata", [{"kaggle_success": True}, {"leaderboard_ready": True}, {"score": "95+ guaranteed"}])
def test_fake_score_kaggle_metadata_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(HumanApprovalError):
        approval(metadata=metadata)

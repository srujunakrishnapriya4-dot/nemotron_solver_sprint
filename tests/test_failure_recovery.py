from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.runbook import (
    FailureRecoveryError,
    FailureRecoveryPlan,
    RecoveryAction,
    RunbookConfig,
    build_default_failure_recovery_plan,
)
from nemotron_engine.runbook.failure_recovery import REQUIRED_RECOVERY_TRIGGERS


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
        final_rehearsal_manifest_hash="a" * 64,
        seed=15,
    )


def action(**kwargs: object) -> RecoveryAction:
    payload = {
        "action_id": "a",
        "trigger": "test suite fails",
        "action": "Stop and inspect logs.",
        "manual_only": True,
        "destructive": False,
        "allowed": True,
        "metadata": {},
    }
    payload.update(kwargs)
    return RecoveryAction(**payload)


def test_default_recovery_plan_contains_required_triggers() -> None:
    plan = build_default_failure_recovery_plan(config())
    assert set(REQUIRED_RECOVERY_TRIGGERS).issubset({item.trigger for item in plan.actions})


def test_destructive_recovery_action_rejected() -> None:
    with pytest.raises(FailureRecoveryError):
        action(destructive=True, allowed=True, action="Manual destructive step.")
    with pytest.raises(FailureRecoveryError):
        action(action="Run rm -rf output.")


@pytest.mark.parametrize(
    "text",
    [
        "Run kaggle competitions submit file.zip",
        "Use kaggle submit for the artifact.",
        "Submit to Kaggle automatically.",
        "Auto submit after recovery.",
    ],
)
def test_kaggle_submit_recovery_action_rejected(text: str) -> None:
    with pytest.raises(FailureRecoveryError):
        action(action=text)


@pytest.mark.parametrize("text", ["Echo token to debug output.", "Display token on screen.", "Cat token file.", "Print credentials for review."])
def test_secret_token_exposure_rejected(text: str) -> None:
    with pytest.raises(FailureRecoveryError):
        action(action=text)


def test_forged_action_and_plan_hash_rejected() -> None:
    with pytest.raises(FailureRecoveryError):
        replace(action(), action_hash="forged")
    plan = build_default_failure_recovery_plan(config())
    with pytest.raises(FailureRecoveryError):
        replace(plan, plan_hash="forged")


def test_fake_metadata_rejected() -> None:
    with pytest.raises(FailureRecoveryError):
        FailureRecoveryPlan(actions=build_default_failure_recovery_plan(config()).actions, runbook_config_hash=config().config_hash, metadata={"leaderboard_ready": True})

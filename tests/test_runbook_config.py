from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.runbook import RunbookConfig, RunbookConfigError, validate_runbook_config


def make_config(**kwargs: object) -> RunbookConfig:
    payload = {
        "runbook_id": "pass15-final",
        "mode": "external_submission_runbook",
        "require_pass14_lock": True,
        "require_final_rehearsal": True,
        "require_human_approval": True,
        "allow_external_submission_instructions": True,
        "allow_post_submission_record": True,
        "accepted_suite_summary": "891 passed, 2 skipped",
        "checkpoint_zip_path": "artifacts/final.zip",
        "final_rehearsal_manifest_hash": "a" * 64,
        "seed": 15,
        "metadata": {},
    }
    payload.update(kwargs)
    return RunbookConfig(**payload)


def test_valid_config_passes() -> None:
    assert validate_runbook_config(make_config()).config_hash


def test_forged_config_hash_rejected() -> None:
    with pytest.raises(RunbookConfigError):
        replace(make_config(), config_hash="forged")


def test_invalid_mode_rejected() -> None:
    with pytest.raises(RunbookConfigError):
        make_config(mode="train_and_submit")


def test_empty_or_unsafe_runbook_id_rejected() -> None:
    with pytest.raises(RunbookConfigError):
        make_config(runbook_id="")
    with pytest.raises(RunbookConfigError):
        make_config(runbook_id="../bad")


def test_bool_seed_rejected() -> None:
    with pytest.raises(RunbookConfigError):
        make_config(seed=True)


def test_missing_and_stale_suite_summary_rejected_unless_override() -> None:
    with pytest.raises(RunbookConfigError):
        make_config(accepted_suite_summary="")
    with pytest.raises(RunbookConfigError):
        make_config(accepted_suite_summary="890 passed, 2 skipped")
    assert make_config(accepted_suite_summary="890 passed, 2 skipped", metadata={"allow_suite_summary_override": True})


def test_final_rehearsal_hash_required_when_configured() -> None:
    with pytest.raises(RunbookConfigError):
        make_config(final_rehearsal_manifest_hash=None)


def test_checkpoint_path_implying_success_rejected() -> None:
    with pytest.raises(RunbookConfigError):
        make_config(checkpoint_zip_path="kaggle-submitted-success/final.zip")


@pytest.mark.parametrize(
    "metadata",
    [
        {"already_submitted": True},
        {"note": "95+ guaranteed"},
        {"public_score": "0.99"},
        {"private_score": "0.99"},
        {"leaderboard_ready": True},
        {"kaggle_success": True},
        {"submission_success": True},
    ],
)
def test_fake_metadata_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(RunbookConfigError):
        make_config(metadata=metadata)

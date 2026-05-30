from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nemotron_engine.runbook import (
    CommandManifest,
    CommandManifestError,
    RunbookCommand,
    RunbookConfig,
    build_default_command_manifest,
    validate_command_manifest,
)


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
        "checkpoint_zip_path": None,
        "final_rehearsal_manifest_hash": "a" * 64,
        "seed": 15,
        "metadata": {},
    }
    payload.update(kwargs)
    return RunbookConfig(**payload)


def command(**kwargs: object) -> RunbookCommand:
    payload = {
        "command_id": "cmd",
        "description": "A static command.",
        "command_text": "python -m pytest tests -q -p no:cacheprovider",
        "phase": "run_tests",
        "manual_only": True,
        "destructive": False,
        "expected_output": "891 passed, 2 skipped",
        "blocker_if_fails": True,
        "metadata": {},
    }
    payload.update(kwargs)
    return RunbookCommand(**payload)


def test_default_command_manifest_builds() -> None:
    manifest = build_default_command_manifest(make_config())
    assert validate_command_manifest(manifest, make_config()).manifest_hash


def test_forged_command_hash_rejected() -> None:
    with pytest.raises(CommandManifestError):
        replace(command(), command_hash="forged")


def test_forged_manifest_hash_rejected() -> None:
    manifest = build_default_command_manifest(make_config())
    with pytest.raises(CommandManifestError):
        replace(manifest, manifest_hash="forged")


@pytest.mark.parametrize("text", ["rm -rf x", "git reset --hard", "git clean -fd", "del /s x", "Remove-Item -Recurse -Force x", "format c:"])
def test_destructive_command_rejected(text: str) -> None:
    with pytest.raises(CommandManifestError):
        command(command_text=text)


@pytest.mark.parametrize("text", ["kaggle competitions submit x", "kaggle.api.competition_submit()", "submit('x')"])
def test_automatic_submit_command_rejected(text: str) -> None:
    with pytest.raises(CommandManifestError):
        command(command_text=text)


def test_manual_external_instruction_allowed_only_with_permission() -> None:
    allowed_cmd = command(
        command_id="manual",
        command_text="Manual only: use the Kaggle web UI to upload the audited artifact.",
        phase="external_submission_manual",
        manual_only=True,
        blocker_if_fails=False,
        metadata={"manual_external_instruction_allowed": True},
    )
    manifest = CommandManifest(commands=(allowed_cmd,), runbook_config_hash=make_config().config_hash)
    assert validate_command_manifest(manifest, make_config())
    denied_config = make_config(allow_external_submission_instructions=False)
    denied_manifest = CommandManifest(commands=(allowed_cmd,), runbook_config_hash=denied_config.config_hash)
    with pytest.raises(CommandManifestError):
        validate_command_manifest(denied_manifest, denied_config)
    with pytest.raises(CommandManifestError):
        command(command_text="Manual only: use the Kaggle web UI.", phase="external_submission_manual", manual_only=False)


def test_validators_do_not_execute_commands(tmp_path: Path) -> None:
    marker = tmp_path / "would_execute.txt"
    inert = command(command_text=f"python -c \"open(r'{marker}', 'w').write('bad')\"")
    manifest = CommandManifest(commands=(inert,), runbook_config_hash=make_config().config_hash)
    validate_command_manifest(manifest, make_config())
    assert not marker.exists()


def test_success_claim_command_rejected() -> None:
    with pytest.raises(CommandManifestError):
        command(command_text="Leaderboard ready with guaranteed score.")

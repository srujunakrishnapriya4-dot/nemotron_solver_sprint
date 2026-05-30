from __future__ import annotations

import pytest

from nemotron_engine.rehearsal import RehearsalConfig, RehearsalConfigError, validate_rehearsal_config
from test_pass14_end_to_end_rehearsal import make_config


def test_valid_dry_run_config_passes() -> None:
    config = make_config()
    assert validate_rehearsal_config(config).config_hash == config.config_hash


def test_forged_config_hash_rejected() -> None:
    payload = make_config().__dict__ | {"config_hash": "forged"}
    with pytest.raises(RehearsalConfigError):
        RehearsalConfig(**payload)


@pytest.mark.parametrize("field", ["dry_run", "allow_zip_build", "allow_runtime_unloaded", "allow_warnings", "require_smoke_evidence", "require_completion_evaluation", "require_private_like", "require_release_candidate"])
def test_non_bool_flags_rejected(field: str) -> None:
    with pytest.raises(RehearsalConfigError):
        make_config(**{field: "yes"})


def test_bool_seed_rejected() -> None:
    with pytest.raises(RehearsalConfigError):
        make_config(seed=True)


def test_output_dir_required_when_zip_enabled() -> None:
    with pytest.raises(RehearsalConfigError):
        make_config(allow_zip_build=True, output_dir=None)


@pytest.mark.parametrize("name", ["../x.zip", "bad/name.zip", "bad\\name.zip", "", "x y.zip"])
def test_unsafe_package_name_rejected(name: str) -> None:
    with pytest.raises(RehearsalConfigError):
        make_config(package_name=name)


def test_submission_zip_name_rejected_unless_safe_output() -> None:
    with pytest.raises(RehearsalConfigError):
        make_config(package_name="submission.zip")
    assert make_config(package_name="submission.zip", allow_zip_build=True, output_dir="C:/tmp/pass14").package_name == "submission.zip"


@pytest.mark.parametrize("metadata", [{"kaggle_success": True}, {"leaderboard_readiness": "ready"}, {"score": "95+ guaranteed"}, {"trained_adapter_available": True}])
def test_fake_success_metadata_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(RehearsalConfigError):
        make_config(metadata=metadata)

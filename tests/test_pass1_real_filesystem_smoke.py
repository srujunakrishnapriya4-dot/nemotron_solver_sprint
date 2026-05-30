from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.runtime.serving_config import (
    ServingConfig,
    ServingConfigError,
    load_serving_config,
    save_serving_config,
    validate_serving_config,
)
from nemotron_engine.submission.submission_validator import (
    validate_adapter_dir,
    validate_submission_zip,
)


_ROOT = Path(__file__).resolve().parent
_FALLBACK_ROOT = _ROOT / ".tmp_pass1_smoke"


@pytest.fixture()
def smoke_dir() -> Path:
    fallback_dir = _FALLBACK_ROOT / f"run_{uuid.uuid4().hex}"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield fallback_dir
    finally:
        _cleanup_path(fallback_dir)
        _cleanup_empty_fallback_root()


def test_validate_adapter_dir_accepts_real_rank_32(smoke_dir: Path) -> None:
    adapter_dir = smoke_dir / "adapter_ok"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")

    result = validate_adapter_dir(adapter_dir)

    assert result.valid is True
    assert result.errors == ()


def test_validate_adapter_dir_rejects_real_rank_33(smoke_dir: Path) -> None:
    adapter_dir = smoke_dir / "adapter_bad_rank"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 33}), encoding="utf-8")

    result = validate_adapter_dir(adapter_dir)

    assert result.valid is False
    assert "r=33 exceeds" in result.errors[0]


def test_validate_adapter_dir_rejects_real_invalid_json(smoke_dir: Path) -> None:
    adapter_dir = smoke_dir / "adapter_bad_json"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{not-json", encoding="utf-8")

    result = validate_adapter_dir(adapter_dir)

    assert result.valid is False
    assert "Invalid JSON" in result.errors[0]


def test_validate_submission_zip_accepts_real_rank_32(smoke_dir: Path) -> None:
    zip_path = smoke_dir / "adapter_ok.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("adapter_config.json", json.dumps({"r": 32}))

    result = validate_submission_zip(zip_path)

    assert result.valid is True
    assert result.errors == ()


def test_validate_submission_zip_rejects_real_rank_33(smoke_dir: Path) -> None:
    zip_path = smoke_dir / "adapter_bad_rank.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("adapter_config.json", json.dumps({"r": 33}))

    result = validate_submission_zip(zip_path)

    assert result.valid is False
    assert "r=33 exceeds" in result.errors[0]


def test_validate_submission_zip_rejects_real_missing_adapter_config(smoke_dir: Path) -> None:
    zip_path = smoke_dir / "adapter_missing_config.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("weights.bin", "placeholder")

    result = validate_submission_zip(zip_path)

    assert result.valid is False
    assert "Missing adapter_config.json" in result.errors[0]


def test_serving_config_real_save_load_and_validation(smoke_dir: Path) -> None:
    path = smoke_dir / "serving_config.json"
    config = ServingConfig(
        model_path="/models/nemotron",
        prompt_template_hash="prompt",
        tokenizer_hash="tokenizer",
        model_hash="model",
        adapter_hash="adapter",
    )

    save_serving_config(config, path)
    loaded = load_serving_config(path)

    assert path.exists()
    assert loaded == config
    assert validate_serving_config(loaded) == loaded


@pytest.mark.parametrize(
    "payload",
    [
        {"num_samples": 2},
        {"majority_vote": True},
    ],
)
def test_strict_validation_rejects_real_loaded_invalid_config(
    smoke_dir: Path,
    payload: dict[str, object],
) -> None:
    path = smoke_dir / "bad_serving_config.json"
    base = ServingConfig(model_path="/models/nemotron").to_dict()
    base.update(payload)
    path.write_text(json.dumps(base, sort_keys=True), encoding="utf-8")

    with pytest.raises(ServingConfigError):
        load_serving_config(path)


def _cleanup_path(path: Path) -> None:
    resolved = path.resolve()
    tests_root = _ROOT.resolve()
    if tests_root not in resolved.parents:
        raise RuntimeError(f"Refusing to clean fallback path outside tests directory: {resolved}")
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _cleanup_empty_fallback_root() -> None:
    try:
        _FALLBACK_ROOT.rmdir()
    except OSError:
        pass

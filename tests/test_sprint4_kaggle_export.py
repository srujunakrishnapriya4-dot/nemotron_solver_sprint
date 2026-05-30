from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.adapter_training_data.kaggle_export import REQUIRED_CURATED_FILES, export_sprint4_kaggle_input  # noqa: E402


def write_curated_source(root: Path) -> None:
    root.mkdir(parents=True)
    for filename in REQUIRED_CURATED_FILES:
        path = root / filename
        if filename.endswith(".jsonl"):
            path.write_text(
                json.dumps({"problem_id": "train-only", "family": "roman_numeral", "messages": []}) + "\n"
                + json.dumps({"problem_id": "test-id", "family": "cipher_text", "messages": []}) + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps({"artifact_hashes": {"train_direct.jsonl": "abc"}}), encoding="utf-8")


def write_test_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt"])
        writer.writeheader()
        writer.writerow({"id": "test-id", "prompt": "hidden"})


def test_sprint4_kaggle_input_zip_created_and_filters_test_rows(tmp_path: Path) -> None:
    source = tmp_path / "adapter_training"
    output = tmp_path / "sprint4_kaggle_input"
    zip_path = tmp_path / "sprint4_kaggle_input.zip"
    test_csv = tmp_path / "test.csv"
    write_curated_source(source)
    write_test_csv(test_csv)

    manifest = export_sprint4_kaggle_input(source, output, zip_path, test_csv_path=test_csv)

    assert zip_path.exists()
    assert manifest["artifact_hashes"]["dataset_manifest.json"]
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "train_direct.jsonl" in names
    assert "sprint4_export_manifest.json" in names
    exported = (output / "train_direct.jsonl").read_text(encoding="utf-8")
    assert "train-only" in exported
    assert "test-id" not in exported


def load_packager():
    script = Path(__file__).resolve().parents[1] / "kaggle_sprint4" / "kaggle_package_adapter.py"
    spec = importlib.util.spec_from_file_location("kaggle_package_adapter", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_validator():
    script = Path(__file__).resolve().parents[1] / "kaggle_sprint4" / "kaggle_validate_adapter.py"
    spec = importlib.util.spec_from_file_location("kaggle_validate_adapter", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_packaging_script_rejects_csv_predictions(tmp_path: Path) -> None:
    packager = load_packager()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
    (adapter / "predictions.csv").write_text("id,answer\nx,1\n", encoding="utf-8")

    with pytest.raises(packager.AdapterPackagingError):
        packager.create_submission_zip(adapter, tmp_path / "submission.zip")


def test_validation_script_rejects_missing_adapter_model(tmp_path: Path) -> None:
    validator = load_validator()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")

    assert validator.validate_adapter_and_zip(adapter, tmp_path / "submission.zip") is False

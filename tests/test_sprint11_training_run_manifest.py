from pathlib import Path

import pytest

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.training_config_schema import load_training_config
from kaggle_anti086.training.training_run_manifest import build_run_manifest


def test_run_manifest_unique_and_records_hashes(tmp_path):
    audit = tmp_path / "audit.json"
    write_json_checked(audit, {"status": "PASS"}, field_name="audit")
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    manifest = build_run_manifest(config, config_path="kaggle_anti086/training/configs/v2a_base_lora.yaml", collator_audit_path=audit, output_root=tmp_path)
    assert manifest["run_id"]
    assert manifest["config_sha256"]
    assert str(tmp_path) in manifest["output_adapter_dir"]
    assert manifest["packaging_allowed"] is False


def test_run_manifest_refuses_overwrite(monkeypatch, tmp_path):
    import kaggle_anti086.training.training_run_manifest as trm

    class FixedDateTime:
        @staticmethod
        def now(tz=None):
            from datetime import datetime, timezone

            return datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(trm, "datetime", FixedDateTime)
    audit = tmp_path / "audit.json"
    write_json_checked(audit, {"status": "PASS"}, field_name="audit")
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    first = build_run_manifest(config, config_path="kaggle_anti086/training/configs/v2a_base_lora.yaml", collator_audit_path=audit, output_root=tmp_path)
    Path(first["output_adapter_dir"]).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        build_run_manifest(config, config_path="kaggle_anti086/training/configs/v2a_base_lora.yaml", collator_audit_path=audit, output_root=tmp_path)

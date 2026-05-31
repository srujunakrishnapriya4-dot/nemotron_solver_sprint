from pathlib import Path

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.train_v2a_lora import build_training_summary, preflight_v2a_training
from kaggle_anti086.training.training_config_schema import load_training_config
from kaggle_anti086.training.training_run_manifest import build_run_manifest


def test_train_refuses_failed_collator_audit(tmp_path):
    audit = tmp_path / "audit.json"
    write_json_checked(audit, {"status": "FAIL"}, field_name="audit")
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    _, failures = preflight_v2a_training(config, config_path="kaggle_anti086/training/configs/v2a_base_lora.yaml", collator_audit_path=audit)
    assert "real_collator_audit_not_pass" in failures


def test_train_refuses_non_v2a_stage(tmp_path):
    audit = tmp_path / "audit.json"
    write_json_checked(audit, {"status": "PASS"}, field_name="audit")
    config = load_training_config("kaggle_anti086/training/configs/v2b_tinker_parent_lora.yaml")
    _, failures = preflight_v2a_training(config, config_path="kaggle_anti086/training/configs/v2b_tinker_parent_lora.yaml", collator_audit_path=audit)
    assert "config_not_runnable_v2a" in failures


def test_train_refuses_unsafe_output_dir(tmp_path):
    audit = tmp_path / "audit.json"
    write_json_checked(audit, {"status": "PASS"}, field_name="audit")
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    config["output_adapter_dir"] = "/kaggle/input/bad"
    _, failures = preflight_v2a_training(config, config_path="x", collator_audit_path=audit)
    assert any(item.startswith("unsafe_output_adapter_dir") for item in failures)


def test_dry_run_summary_does_not_pretend_trained(tmp_path):
    audit = tmp_path / "audit.json"
    write_json_checked(audit, {"status": "PASS"}, field_name="audit")
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    manifest = build_run_manifest(config, config_path="kaggle_anti086/training/configs/v2a_base_lora.yaml", collator_audit_path=audit, output_root=tmp_path)
    summary = build_training_summary(config, manifest, dry_run=True, failures=[], warnings=[])
    assert summary["status"] == "PASS"
    assert summary["trained"] is False
    assert summary["steps_completed"] == 0

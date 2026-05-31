from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.train_v2a_lora import preflight_v2a_training
from kaggle_anti086.training.train_v2a_lora import build_training_summary
from kaggle_anti086.training.training_config_schema import load_training_config


def test_kaggle_mode_missing_deps_fail_honestly(tmp_path):
    audit = tmp_path / "audit.json"
    write_json_checked(audit, {"status": "WARN_LOCAL_TOKENIZER_UNAVAILABLE"}, field_name="audit")
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    _, failures = preflight_v2a_training(config, config_path="kaggle_anti086/training/configs/v2a_base_lora.yaml", collator_audit_path=audit)
    assert "real_collator_audit_not_pass" in failures


def test_forced_backend_failure_string_removed():
    text = open("kaggle_anti086/training/train_v2a_lora.py", encoding="utf-8").read()
    assert "actual_training_backend_not_available_in_this_entrypoint" not in text


def test_training_summary_includes_memory_and_log_paths():
    summary = build_training_summary(
        {"stage": "v2a_base_lora", "rank": 32, "target_modules": ["q_proj", "v_proj", "o_proj"], "num_steps": 5},
        {"run_id": "r", "output_adapter_dir": "out"},
        dry_run=False,
        failures=[],
        warnings=[],
        backend_summary={"status": "PASS", "steps_completed": 5, "memory_snapshots": [{"stage": "x"}], "training_log_history_path": "log.json", "run_provenance_path": "prov.json"},
    )
    assert summary["memory_snapshots"]
    assert summary["training_log_history_path"] == "log.json"
    assert summary["run_provenance_path"] == "prov.json"
    assert summary["packaging_allowed"] is False

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.train_v2a_lora import preflight_v2a_training
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

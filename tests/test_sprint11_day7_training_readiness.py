from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.day7_training_readiness import build_day7_readiness


def _reports(tmp_path, **statuses):
    defaults = {
        "config_validation": {"status": "PASS", "runnable_count": 1},
        "tokenization": {"status": "PASS"},
        "loss_mask": {"status": "PASS"},
        "sampling_policy_consumption": {"status": "PASS"},
        "adapter_constraint": {"status": "PASS"},
        "source_leakage": {"status": "PASS"},
        "day6_sampling_policy": {"status": "PASS"},
    }
    for key, status in statuses.items():
        defaults[key] = {"status": status} if isinstance(status, str) else status
    paths = {}
    for name, payload in defaults.items():
        path = tmp_path / f"{name}.json"
        write_json_checked(path, payload, field_name=name)
        paths[name] = path
    return paths


def test_all_pass_allows_day8_not_today_training(tmp_path):
    report = build_day7_readiness(_reports(tmp_path))
    assert report["decision"] == "ALLOW_DAY8_V2_TRAINING"
    assert report["training_allowed_today"] is False
    assert report["training_allowed_next_stage"] is True
    assert report["packaging_allowed"] is False


def test_failures_block_training(tmp_path):
    assert build_day7_readiness(_reports(tmp_path, config_validation={"status": "PASS", "runnable_count": 0}))["decision"] == "BLOCK_TRAINING"
    assert build_day7_readiness(_reports(tmp_path, tokenization="FAIL"))["decision"] == "BLOCK_TRAINING"
    assert build_day7_readiness(_reports(tmp_path, loss_mask="FAIL"))["decision"] == "BLOCK_TRAINING"
    assert build_day7_readiness(_reports(tmp_path, sampling_policy_consumption="FAIL"))["decision"] == "BLOCK_TRAINING"
    assert build_day7_readiness(_reports(tmp_path, adapter_constraint="FAIL"))["decision"] == "BLOCK_TRAINING"

from kaggle_anti086.data.day6_1_corpus_audit_summary import build_day6_1_summary
from kaggle_anti086.data.v2_corpus_io import write_json_checked


def _reports(tmp_path, **statuses):
    defaults = {
        "source_leakage_audit": "PASS",
        "independence_report": "PASS",
        "prompt_diversity_report": "PASS",
        "sampling_policy": "PASS",
        "hard_negative_taxonomy": "WARN",
    }
    defaults.update(statuses)
    paths = {}
    for name, status in defaults.items():
        path = tmp_path / f"{name}.json"
        write_json_checked(path, {"status": status}, field_name=name)
        paths[name] = path
    return paths


def test_source_leakage_failure_blocks_day7_config(tmp_path):
    summary = build_day6_1_summary(_reports(tmp_path, source_leakage_audit="FAIL"))
    assert summary["decision"] == "BLOCK_DAY7_TRAINING_CONFIG_PREP"


def test_loss_policy_failure_blocks_day7_config(tmp_path):
    summary = build_day6_1_summary(_reports(tmp_path, sampling_policy="FAIL"))
    assert summary["decision"] == "BLOCK_DAY7_TRAINING_CONFIG_PREP"


def test_prompt_diversity_failure_blocks_day7_config(tmp_path):
    summary = build_day6_1_summary(_reports(tmp_path, prompt_diversity_report="FAIL"))
    assert summary["decision"] == "BLOCK_DAY7_TRAINING_CONFIG_PREP"


def test_independence_failure_blocks_day7_config(tmp_path):
    summary = build_day6_1_summary(_reports(tmp_path, independence_report="FAIL"))
    assert summary["decision"] == "BLOCK_DAY7_TRAINING_CONFIG_PREP"


def test_only_hard_negative_warning_allows_day7_config(tmp_path):
    summary = build_day6_1_summary(_reports(tmp_path, hard_negative_taxonomy="WARN"))
    assert summary["decision"] == "ALLOW_DAY7_TRAINING_CONFIG_PREP"
    assert summary["training_allowed"] is False
    assert summary["packaging_allowed"] is False
    assert summary["submission_allowed"] is False
    assert summary["no_leaderboard_evidence"] is True
    assert summary["no_0_95_evidence"] is True

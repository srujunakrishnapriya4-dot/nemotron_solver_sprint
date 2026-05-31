import json

from kaggle_anti086.training.final_sft_leakage_audit import audit_final_sft_leakage
from kaggle_anti086.training.sft_dataset import SFTItem


def _item(prompt="train prompt", rule="train_rule", group="train_group"):
    return SFTItem({"id": "t1", "prompt": prompt, "rule_id": rule, "leakage_group": group, "family": "word_cipher", "verification_status": "verified"}, "direct_answer", 1.0)


def test_clean_split_passes(tmp_path):
    eval_path = tmp_path / "day5_rule_holdout_eval_512.jsonl"
    eval_path.write_text(json.dumps({"prompt": "other", "rule_id": "holdout_x", "leakage_group": "g2"}) + "\n", encoding="utf-8")
    report = audit_final_sft_leakage([_item()], [eval_path])
    assert report["status"] == "PASS"


def test_prompt_and_rule_overlap_fail(tmp_path):
    eval_path = tmp_path / "day5_rule_holdout_eval_512.jsonl"
    eval_path.write_text(json.dumps({"prompt": "train prompt", "rule_id": "train_rule", "leakage_group": "x"}) + "\n", encoding="utf-8")
    report = audit_final_sft_leakage([_item()], [eval_path])
    assert report["status"] == "FAIL"
    assert "train_eval_prompt_overlap" in report["failures"]
    assert "rule_holdout_rule_overlap" in report["failures"]

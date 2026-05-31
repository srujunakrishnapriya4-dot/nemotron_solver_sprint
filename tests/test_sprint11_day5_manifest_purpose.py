import json

from kaggle_anti086.eval.day5_eval_factory import build_answerable_rows, build_eval_rows
from kaggle_anti086.eval.eval_manifest import build_eval_manifest


def _write(tmp_path, name: str, rows: list[dict]):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_manifest_separates_mixed_behavior_from_answer_accuracy(tmp_path) -> None:
    paths = {
        "private_like": _write(tmp_path, "private", build_eval_rows("private_like", 128, 1)),
        "private_like_answerable": _write(tmp_path, "private_answerable", build_answerable_rows("private_like", 64, 11)),
        "rule_holdout": _write(tmp_path, "holdout", build_eval_rows("rule_holdout", 128, 2)),
        "rule_holdout_answerable": _write(tmp_path, "holdout_answerable", build_answerable_rows("rule_holdout", 64, 12)),
        "family_hard": _write(tmp_path, "hard", build_eval_rows("family_hard", 128, 3)),
        "family_hard_answerable": _write(tmp_path, "hard_answerable", build_answerable_rows("family_hard", 64, 13)),
        "anti_leak": _write(tmp_path, "leak", build_eval_rows("anti_leak", 128, 4)),
    }
    manifest = build_eval_manifest(paths)
    assert manifest["status"] == "PASS"
    assert manifest["eval_purpose"]["private_like"] == "mixed_behavior_eval"
    assert manifest["eval_purpose"]["private_like_answerable"] == "answer_accuracy_eval"
    assert manifest["files"]["family_hard_answerable"]["expected_abstain_count"] == 0
    assert "subfamily_counts" in manifest["files"]["rule_holdout_answerable"]
    assert "duplicate_rule_signature_count" in manifest["files"]["anti_leak"]

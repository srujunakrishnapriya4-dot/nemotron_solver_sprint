import json

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.semantic_holdout import build_semantic_holdout_report


def _write(tmp_path, name: str, rows: list[dict]):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_semantic_holdout_passes_clean_day5_rows(tmp_path) -> None:
    report = build_semantic_holdout_report(
        _write(tmp_path, "holdout", build_eval_rows("rule_holdout", 64, 1)),
        {
            "private_like": _write(tmp_path, "private", build_eval_rows("private_like", 64, 2)),
            "family_hard": _write(tmp_path, "hard", build_eval_rows("family_hard", 64, 3)),
            "anti_leak": _write(tmp_path, "leak", build_eval_rows("anti_leak", 64, 4)),
        },
    )
    assert report["status"] == "PASS"
    assert report["semantic_holdout_pass"] is True


def test_semantic_holdout_fails_signature_overlap(tmp_path) -> None:
    holdout = build_eval_rows("rule_holdout", 16, 1)
    private = build_eval_rows("private_like", 16, 2)
    private[0]["metadata"]["rule_signature"] = holdout[0]["metadata"]["rule_signature"]
    report = build_semantic_holdout_report(
        _write(tmp_path, "holdout", holdout),
        {"private_like": _write(tmp_path, "private", private)},
    )
    assert report["status"] == "FAIL"
    assert report["rule_signature_overlap_count"] == 1

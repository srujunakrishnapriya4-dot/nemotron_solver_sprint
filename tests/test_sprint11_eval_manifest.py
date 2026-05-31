import json

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows, write_rows
from kaggle_anti086.eval.eval_manifest import build_eval_manifest


def _write(tmp_path, name: str, rows: list[dict]):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_eval_manifest_passes_clean_files(tmp_path) -> None:
    paths = {
        "private_like": _write(tmp_path, "private", build_eval_rows("private_like", 128, 1)),
        "rule_holdout": _write(tmp_path, "holdout", build_eval_rows("rule_holdout", 128, 2)),
        "family_hard": _write(tmp_path, "hard", build_eval_rows("family_hard", 128, 3)),
        "anti_leak": _write(tmp_path, "leak", build_eval_rows("anti_leak", 128, 4)),
    }
    manifest = build_eval_manifest(paths)
    assert manifest["status"] == "PASS"
    assert manifest["files"]["private_like"]["sha256"]
    assert manifest["files"]["private_like"]["row_count"] == 128


def test_eval_manifest_fails_duplicate_rule_and_leakage(tmp_path) -> None:
    rows = build_eval_rows("private_like", 64, 1)
    rows[1]["rule_id"] = rows[0]["rule_id"]
    rows[2]["leakage_group"] = rows[0]["leakage_group"]
    paths = {
        "private_like": _write(tmp_path, "private", rows),
        "rule_holdout": _write(tmp_path, "holdout", build_eval_rows("rule_holdout", 64, 2)),
        "family_hard": _write(tmp_path, "hard", build_eval_rows("family_hard", 64, 3)),
        "anti_leak": _write(tmp_path, "leak", build_eval_rows("anti_leak", 64, 4)),
    }
    assert build_eval_manifest(paths)["status"] == "FAIL"


def test_eval_manifest_fails_missing_family(tmp_path) -> None:
    rows = [row for row in build_eval_rows("private_like", 128, 1) if row["family"] != "symbol_mapping"]
    paths = {
        "private_like": _write(tmp_path, "private", rows),
        "rule_holdout": _write(tmp_path, "holdout", build_eval_rows("rule_holdout", 128, 2)),
        "family_hard": _write(tmp_path, "hard", build_eval_rows("family_hard", 128, 3)),
        "anti_leak": _write(tmp_path, "leak", build_eval_rows("anti_leak", 128, 4)),
    }
    assert build_eval_manifest(paths)["status"] == "FAIL"

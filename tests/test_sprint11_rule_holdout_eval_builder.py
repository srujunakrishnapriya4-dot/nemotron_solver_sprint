from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.eval_manifest import check_no_rule_overlap


def test_rule_holdout_metadata_and_prefixes() -> None:
    rows = build_eval_rows("rule_holdout", 512, 1106)
    assert all(row["metadata"]["holdout_only"] is True for row in rows)
    assert all(row["metadata"]["train_allowed"] is False for row in rows)
    assert all(row["metadata"]["rule_holdout_reserved"] is True for row in rows)
    assert all(row["rule_id"].startswith("holdout_") for row in rows)
    assert {"bit_manipulation", "symbol_mapping", "char_cipher", "word_cipher", "unit_conversion", "custom_numeral", "equation_operator"} <= {row["family"] for row in rows}


def test_rule_holdout_no_overlap_with_other_day5_groups() -> None:
    groups = {
        "private_like": build_eval_rows("private_like", 128, 1),
        "rule_holdout": build_eval_rows("rule_holdout", 128, 2),
        "family_hard": build_eval_rows("family_hard", 128, 3),
        "anti_leak": build_eval_rows("anti_leak", 128, 4),
    }
    report = check_no_rule_overlap(groups)
    assert report["rule_id_overlap_count"] == 0
    assert report["leakage_group_overlap_count"] == 0

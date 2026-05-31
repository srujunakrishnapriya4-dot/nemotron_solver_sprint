from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.eval_manifest import check_no_rule_overlap


def test_all_day5_builders_have_no_cross_file_leakage_or_id_overlap() -> None:
    groups = {
        "private_like": build_eval_rows("private_like", 512, 1105),
        "rule_holdout": build_eval_rows("rule_holdout", 512, 1106),
        "family_hard": build_eval_rows("family_hard", 512, 1107),
        "anti_leak": build_eval_rows("anti_leak", 256, 1108),
    }
    all_rows = [row for rows in groups.values() for row in rows]
    assert validate_rows(all_rows)["failure_count"] == 0
    assert len({row["id"] for row in all_rows}) == len(all_rows)
    overlap = check_no_rule_overlap(groups)
    assert overlap["rule_id_overlap_count"] == 0
    assert overlap["leakage_group_overlap_count"] == 0

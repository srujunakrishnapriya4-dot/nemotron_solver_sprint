from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.family_coverage_report import build_family_coverage_report


def test_family_coverage_report_distributions_and_no_warnings_for_clean_day5() -> None:
    report = build_family_coverage_report(
        {
            "private_like": build_eval_rows("private_like", 512, 1105),
            "rule_holdout": build_eval_rows("rule_holdout", 512, 1106),
            "family_hard": build_eval_rows("family_hard", 512, 1107),
            "anti_leak": build_eval_rows("anti_leak", 256, 1108),
        }
    )
    assert report["evals"]["private_like"]["by_family"]["symbol_mapping"] == 64
    assert report["evals"]["family_hard"]["by_family"]["char_cipher"] >= 80
    assert report["evals"]["private_like"]["noise_profile_distribution"]
    assert report["evals"]["private_like"]["difficulty_distribution"]
    assert report["warnings"] == []


def test_family_coverage_report_warns_on_low_weak_family_coverage() -> None:
    rows = [row for row in build_eval_rows("family_hard", 128, 3) if row["family"] != "symbol_mapping"]
    report = build_family_coverage_report({"family_hard": rows, "private_like": [], "rule_holdout": [], "anti_leak": []})
    assert "family_hard_low_symbol_mapping_coverage" in report["warnings"]

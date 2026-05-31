from collections import Counter

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows


def test_family_hard_oversamples_weak_families() -> None:
    rows = build_eval_rows("family_hard", 512, 1107)
    counts = Counter(row["family"] for row in rows)
    assert counts["symbol_mapping"] >= 96
    assert counts["char_cipher"] >= 80
    assert counts["bit_manipulation"] >= 80
    assert counts["custom_numeral"] >= 48
    assert counts["equation_operator"] >= 48
    assert all(row["metadata"]["expected_solver_behavior"] == "abstain" for row in rows if row["family"] == "equation_operator")

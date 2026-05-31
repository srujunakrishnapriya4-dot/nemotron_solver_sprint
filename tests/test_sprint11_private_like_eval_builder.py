from collections import Counter

from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.eval.day5_eval_factory import build_eval_rows


def test_private_like_builder_exact_count_schema_unique_and_ratio() -> None:
    rows = build_eval_rows("private_like", 512, 1105)
    assert len(rows) == 512
    assert validate_rows(rows)["failure_count"] == 0
    assert len({row["id"] for row in rows}) == 512
    assert len({row["rule_id"] for row in rows}) == 512
    assert len({row["leakage_group"] for row in rows}) == 512
    counts = Counter(row["family"] for row in rows)
    for family in ("bit_manipulation", "symbol_mapping", "char_cipher", "word_cipher", "unit_conversion", "numeric_formula", "gravity_numeric", "roman_numeral", "custom_numeral", "equation_operator"):
        assert counts[family] > 0
    behaviors = Counter(row["metadata"]["expected_solver_behavior"] for row in rows)
    assert behaviors["answer"] / len(rows) >= 0.70
    assert behaviors["abstain"] / len(rows) <= 0.30
    assert len({row["prompt"] for row in rows}) == len(rows)

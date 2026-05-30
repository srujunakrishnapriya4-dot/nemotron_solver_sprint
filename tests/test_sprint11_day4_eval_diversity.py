from collections import Counter

from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.eval.build_day4_adversarial_solver_eval import build_rows


def test_day4_1_eval_has_unique_rule_and_leakage_instances() -> None:
    rows = build_rows(variant="day4_1")
    assert len(rows) >= 256
    assert validate_rows(rows)["failure_count"] == 0
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["rule_id"] for row in rows}) == len(rows)
    assert len({row["leakage_group"] for row in rows}) == len(rows)
    assert all(row["metadata"].get("rule_signature") for row in rows)
    assert all(row["metadata"].get("generator_id") for row in rows)
    assert all(row["metadata"].get("noise_profile") for row in rows)


def test_day4_1_eval_distribution_and_behavior_labels() -> None:
    rows = build_rows(variant="day4_1")
    counts = Counter(row["family"] for row in rows)
    assert counts["bit_manipulation"] >= 48
    assert counts["symbol_mapping"] >= 48
    assert counts["char_cipher"] >= 40
    assert counts["word_cipher"] >= 32
    assert counts["unit_conversion"] >= 32
    assert counts["numeric_formula"] + counts["gravity_numeric"] >= 32
    assert counts["roman_numeral"] + counts["custom_numeral"] >= 24
    behaviors = Counter(row["metadata"]["expected_solver_behavior"] for row in rows)
    assert behaviors["answer"] > 0
    assert behaviors["abstain"] > 0


def test_day4_1_eval_prompt_templates_are_not_repeated() -> None:
    rows = build_rows(variant="day4_1")
    normalized_templates = Counter(row["prompt"] for row in rows)
    assert max(normalized_templates.values()) <= 1

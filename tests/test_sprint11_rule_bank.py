from __future__ import annotations

from dataclasses import replace

from kaggle_anti086.data.build_rule_bank import RuleSpec, build_rule_bank, validate_rule_bank


def test_build_rule_bank_nonempty_and_valid() -> None:
    rules = build_rule_bank()
    report = validate_rule_bank(rules)
    assert rules
    assert report["status"] == "PASS", report


def test_rule_ids_unique_and_required_families_present() -> None:
    rules = build_rule_bank()
    rule_ids = [rule.rule_id for rule in rules]
    assert len(rule_ids) == len(set(rule_ids))
    families = {rule.family for rule in rules}
    assert {"roman_numeral", "unit_conversion", "numeric_formula", "word_cipher", "bit_manipulation", "symbol_mapping", "format_only"} <= families
    subfamilies = {rule.subfamily for rule in rules}
    assert {"standard_roman", "roman_with_decimal_prefix_noise", "rotate_then_xor", "boxed_answer"} <= subfamilies


def test_duplicate_rule_id_fails() -> None:
    rules = build_rule_bank()
    report = validate_rule_bank([rules[0], rules[0]])
    assert report["status"] == "FAIL"
    assert any("duplicate rule_id" in failure for failure in report["failures"])


def test_unknown_family_and_missing_generator_fail() -> None:
    bad_family = RuleSpec("bad", "not_real", "x", {}, True, True, 0.1, "gen", "")
    missing_generator = replace(build_rule_bank()[0], generator_id="")
    report = validate_rule_bank([bad_family, missing_generator])
    assert report["status"] == "FAIL"
    assert any("unknown family" in failure for failure in report["failures"])
    assert any("missing generator_id" in failure for failure in report["failures"])

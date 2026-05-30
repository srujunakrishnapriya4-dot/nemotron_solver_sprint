from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from kaggle_anti086.data.schema import ALLOWED_FAMILIES


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    family: str
    subfamily: str
    parameters: dict
    train_allowed: bool
    holdout_allowed: bool
    difficulty: float
    generator_id: str
    notes: str


def build_rule_bank() -> list[RuleSpec]:
    specs = [
        ("roman_standard_roman", "roman_numeral", "standard_roman"),
        ("roman_decimal_prefix_noise", "roman_numeral", "roman_with_decimal_prefix_noise"),
        ("unit_linear_scale_round_2", "unit_conversion", "linear_scale_round_2"),
        ("unit_linear_offset_round_2", "unit_conversion", "linear_offset_round_2"),
        ("unit_ratio_conversion", "unit_conversion", "ratio_conversion"),
        ("numeric_linear_ax", "numeric_formula", "linear_ax"),
        ("numeric_linear_ax_plus_b", "numeric_formula", "linear_ax_plus_b"),
        ("numeric_gravity_half_g_t_squared", "numeric_formula", "gravity_half_g_t_squared"),
        ("numeric_quadratic_ax2_plus_b", "numeric_formula", "quadratic_ax2_plus_b"),
        ("word_sub_full_dict", "word_cipher", "word_substitution_full_dictionary"),
        ("word_sub_partial_dict", "word_cipher", "word_substitution_partial_dictionary"),
        ("bit_xor_mask", "bit_manipulation", "xor_mask"),
        ("bit_rotate_left", "bit_manipulation", "rotate_left"),
        ("bit_rotate_right", "bit_manipulation", "rotate_right"),
        ("bit_reverse_bits", "bit_manipulation", "reverse_bits"),
        ("bit_rotate_then_xor", "bit_manipulation", "rotate_then_xor"),
        ("symbol_substitution", "symbol_mapping", "symbol_to_symbol_substitution"),
        ("symbol_positionwise_mapping", "symbol_mapping", "positionwise_symbol_mapping"),
        ("format_boxed_answer", "format_only", "boxed_answer"),
        ("format_answer_prefix", "format_only", "answer_prefix"),
        ("format_think_tag_suffix", "format_only", "think_tag_suffix"),
        ("format_unit_suffix", "format_only", "unit_suffix"),
    ]
    return [
        RuleSpec(
            rule_id=rule_id,
            family=family,
            subfamily=subfamily,
            parameters={},
            train_allowed=True,
            holdout_allowed=True,
            difficulty=0.5,
            generator_id=f"day2_registry::{family}",
            notes="Registry entry only; generator not implemented in Day 2.",
        )
        for rule_id, family, subfamily in specs
    ]


def validate_rule_bank(rules: list[RuleSpec]) -> dict:
    failures: list[str] = []
    seen: set[str] = set()
    collisions: set[tuple[str, str, str]] = set()
    required_families = {"roman_numeral", "unit_conversion", "numeric_formula", "word_cipher", "bit_manipulation", "symbol_mapping", "format_only"}
    present_families = {rule.family for rule in rules}
    for family in sorted(required_families - present_families):
        failures.append(f"missing required family: {family}")
    for rule in rules:
        if not rule.rule_id or not rule.family or not rule.subfamily:
            failures.append("rule missing rule_id/family/subfamily")
        if rule.rule_id in seen:
            failures.append(f"duplicate rule_id: {rule.rule_id}")
        seen.add(rule.rule_id)
        key = (rule.family, rule.subfamily, rule.rule_id)
        if key in collisions:
            failures.append(f"duplicate family/subfamily/rule_id: {key}")
        collisions.add(key)
        if rule.family not in ALLOWED_FAMILIES:
            failures.append(f"unknown family: {rule.family}")
        if not rule.generator_id:
            failures.append(f"missing generator_id: {rule.rule_id}")
        if not isinstance(rule.train_allowed, bool) or not isinstance(rule.holdout_allowed, bool):
            failures.append(f"train_allowed/holdout_allowed must be bool: {rule.rule_id}")
        if not isinstance(rule.parameters, dict):
            failures.append(f"parameters must be dict: {rule.rule_id}")
    return {"status": "FAIL" if failures else "PASS", "rule_count": len(rules), "families": sorted(present_families), "failures": failures}


def write_rule_bank_json(path: str | Path, rules: list[RuleSpec] | None = None) -> None:
    rules = rules or build_rule_bank()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(rule) for rule in rules], sort_keys=True, indent=2), encoding="utf-8")

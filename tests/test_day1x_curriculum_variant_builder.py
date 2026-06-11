from __future__ import annotations

from kaggle_anti086.training.day1x_curriculum_variant_builder import (
    build_composed_heavy_variant,
    build_direct_variant,
    build_format_heavy_variant,
    build_hard_p2_heavy_variant,
    build_mixed_curriculum_variant,
    build_public_style_heavy_variant,
    build_short_trace_variant,
    validate_variant_rows,
)


def _row(index: int, family: str, source: str, prompt_style: str = "direct", difficulty: str = "medium") -> dict[str, object]:
    answer = str(index + 10)
    prompt = f"{family} prompt {index}"
    target = f"Compute the requested value.\nThe value is {answer}.\n\\boxed{{{answer}}}"
    return {
        "id": f"id-{family}-{source}-{index}",
        "record_id": f"record-{family}-{source}-{index}",
        "family": family,
        "prompt_style": prompt_style,
        "difficulty": difficulty,
        "prompt": prompt,
        "answer": answer,
        "target_text": target,
        "trace": target,
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "metadata": {"mixture_source": source, "output_type": "integer"},
    }


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    originals = ["equation_operator", "sequence_pattern", "gravity_numeric", "numeric_formula_safe", "unit_conversion"]
    composed = ["composed_sequence_operator", "composed_unit_formula", "composed_symbol_equation", "composed_custom_numeral_arithmetic"]
    for index in range(30):
        rows.append(_row(index, originals[index % len(originals)], "original_day1"))
    for index in range(30, 55):
        rows.append(_row(index, composed[index % len(composed)], "composed_day1x"))
    for index in range(55, 75):
        rows.append(_row(index, composed[index % len(composed)], "public_style_day1x", "public_style_minimal"))
    for index in range(75, 90):
        rows.append(_row(index, "numeric_formula_safe", "format_stress", "public_style_noisy_context", "trap"))
    return rows


def test_direct_variant_assistant_is_exactly_boxed_answer() -> None:
    variant = build_direct_variant(_rows(), 10, 1)
    assert all(row["messages"][1]["content"] == f"\\boxed{{{row['answer']}}}" for row in variant)
    assert validate_variant_rows(variant, "direct", 10)["status"] == "PASS"


def test_short_trace_variant_has_trace_plus_final_box() -> None:
    variant = build_short_trace_variant(_rows(), 10, 2)
    assert all("\n" in row["target_text"] and row["target_text"].endswith(f"\\boxed{{{row['answer']}}}") for row in variant)


def test_mixed_curriculum_contains_all_expected_buckets() -> None:
    variant = build_mixed_curriculum_variant(_rows(), 40, 3)
    buckets = {row["metadata"]["curriculum_bucket"] for row in variant}
    assert {"direct", "short_trace", "composed", "public_style", "format_stress"} <= buckets
    assert all([row["messages"][0]["role"] == "user" and row["messages"][1]["role"] == "assistant" for row in variant])
    assert validate_variant_rows(variant, "mixed_curriculum", 40)["status"] == "PASS"


def test_format_heavy_has_majority_direct_rows() -> None:
    variant = build_format_heavy_variant(_rows(), 30, 4)
    direct = sum(1 for row in variant if row["metadata"]["target_style"] == "direct")
    assert direct > len(variant) // 2


def test_composed_and_public_style_heavy_contain_target_rows() -> None:
    composed = build_composed_heavy_variant(_rows(), 30, 5)
    public = build_public_style_heavy_variant(_rows(), 30, 6)
    assert any(row["metadata"]["is_composed"] for row in composed)
    assert any(row["metadata"]["is_public_style"] for row in public)


def test_hard_p2_heavy_contains_only_target_families() -> None:
    variant = build_hard_p2_heavy_variant(_rows(), 20, 7)
    assert variant
    assert all(row["metadata"]["curriculum_bucket"] == "hard_p2" for row in variant)
    assert all(row["family"] in {"equation_operator", "sequence_pattern", "gravity_numeric", "numeric_formula_safe", "composed_sequence_operator", "composed_unit_formula", "composed_symbol_equation", "composed_custom_numeral_arithmetic"} for row in variant)


def test_variant_validation_catches_duplicate_ids_and_malformed_box() -> None:
    variant = build_direct_variant(_rows(), 5, 8)
    duplicate = [dict(row) for row in variant]
    duplicate[1]["id"] = duplicate[0]["id"]
    report = validate_variant_rows(duplicate, "direct", 5)
    assert report["duplicate_id_count"] == 1
    assert report["status"] == "FAIL"

    malformed = [dict(row) for row in variant]
    malformed[0]["target_text"] = "\\boxed{1}\\boxed{2}"
    malformed[0]["messages"] = [malformed[0]["messages"][0], {"role": "assistant", "content": "\\boxed{1}\\boxed{2}"}]
    report = validate_variant_rows(malformed, "direct", 5)
    assert report["format_error_count"] > 0


def test_variant_manifest_flags_and_determinism() -> None:
    first = build_direct_variant(_rows(), 10, 9)
    second = build_direct_variant(_rows(), 10, 9)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    report = validate_variant_rows(first, "direct", 10)
    assert report["package_authorized"] is False
    assert report["submission_authorized"] is False
    assert report["leaderboard_claim"] is False
    assert report["abstain_accepted"] == 0

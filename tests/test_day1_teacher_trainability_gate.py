from __future__ import annotations

from dataclasses import asdict
import random

import pytest

from kaggle_anti086.training.day1_teacher_families import SyntheticExample, available_teachers, get_teacher
from kaggle_anti086.training.day1_teacher_trainability_gate import (
    assert_batch_trainable,
    detect_abstain_placeholder,
    detect_answer_mismatch,
    detect_duplicate_prompt,
    detect_multiple_boxes,
    detect_text_after_final_box,
    detect_unsafe_metadata,
    normalize_answer_for_family,
    verify_bit_manipulation,
    verify_char_cipher,
    verify_custom_numeral,
    verify_equation_operator,
    verify_gravity_numeric,
    verify_numeric_formula_safe,
    verify_permutation_sorting,
    verify_sequence_pattern,
    verify_symbol_mapping,
    verify_trainable_batch,
    verify_trainable_row,
    verify_unit_conversion,
    verify_word_cipher,
)


WRAPPERS = {
    "custom_numeral": verify_custom_numeral,
    "symbol_mapping": verify_symbol_mapping,
    "bit_manipulation": verify_bit_manipulation,
    "char_cipher": verify_char_cipher,
    "word_cipher": verify_word_cipher,
    "permutation_sorting": verify_permutation_sorting,
    "gravity_numeric": verify_gravity_numeric,
    "unit_conversion": verify_unit_conversion,
    "numeric_formula_safe": verify_numeric_formula_safe,
    "equation_operator": verify_equation_operator,
    "sequence_pattern": verify_sequence_pattern,
}


def _example(family: str = "symbol_mapping") -> SyntheticExample:
    return get_teacher(family).generate_synthetic(random.Random(123), "easy", "direct")


def _row(family: str = "symbol_mapping") -> dict[str, object]:
    row = asdict(_example(family))
    row["trainable"] = True
    return row


def test_detectors_catch_abstain_boxes_format_and_metadata_failures() -> None:
    assert detect_abstain_placeholder({"answer": "ABSTAIN", "target_text": "\\boxed{ABSTAIN}"})
    assert detect_abstain_placeholder("\\boxed{ABSTAIN}")
    assert detect_multiple_boxes("\\boxed{1}\\boxed{2}")
    assert detect_text_after_final_box("work\n\\boxed{1} trailing")
    assert detect_answer_mismatch("work\n\\boxed{2}", "1", "unit_conversion")
    seen: set[str] = set()
    assert not detect_duplicate_prompt("A   B", seen)
    assert detect_duplicate_prompt(" A B ", seen)
    assert detect_unsafe_metadata({"source": "model_guess_unverified"})
    assert detect_unsafe_metadata({"local_recovery_claim": True})


def test_family_aware_normalization_is_conservative() -> None:
    assert normalize_answer_for_family(" 0007 ", "unit_conversion", {"output_type": "integer"}) == "7"
    assert normalize_answer_for_family(" 0007 ", "custom_numeral", {"output_type": "symbol"}) == "0007"
    assert normalize_answer_for_family(" AbC ", "char_cipher") == "AbC"
    assert normalize_answer_for_family(" red  blue ", "word_cipher") == "red  blue"
    assert normalize_answer_for_family("2, 5,7", "permutation_sorting") == "2,5,7"
    assert normalize_answer_for_family(" AbC ", "symbol_mapping") == "AbC"
    assert normalize_answer_for_family(" -0 ", "numeric_formula_safe") == "0"


def test_valid_generated_example_from_each_teacher_passes_gate() -> None:
    for family, teacher in available_teachers().items():
        example = teacher.generate_synthetic(random.Random(17), "medium", "table")
        check = verify_trainable_row(example)
        assert check.trainable, (family, check.rejection_reason)
        assert check.verification_status == "PASS"
        assert check.ambiguity_count == 0


def test_valid_dict_row_passes_gate() -> None:
    check = verify_trainable_row(_row("bit_manipulation"))
    assert check.trainable
    assert check.rejection_reason is None


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"family": "unknown_family"}, "unknown_family"),
        ({"prompt": ""}, "missing_prompt"),
        ({"answer": ""}, "missing_answer"),
        ({"target_text": ""}, "missing_target_text"),
        ({"trace": ""}, "missing_trace"),
        ({"target_text": "\\boxed{1}\\boxed{1}", "trace": "\\boxed{1}\\boxed{1}"}, "multiple_boxes"),
        ({"target_text": "\\boxed{1} after", "trace": "\\boxed{1} after"}, "text_after_final_box"),
        ({"answer": "not_the_answer"}, "answer_mismatch"),
        ({"ambiguity_count": 1}, "ambiguity_nonzero"),
        ({"verification_status": "FAIL"}, "verification_not_pass"),
        ({"rejection_reason": "bad"}, "row_rejection_reason_present"),
        ({"trainable": False}, "unsafe_metadata"),
        ({"metadata": {"source": "model_guess_unverified"}}, "unsafe_metadata"),
        ({"answer": "ABSTAIN", "target_text": "\\boxed{ABSTAIN}", "trace": "\\boxed{ABSTAIN}"}, "abstain_placeholder"),
    ],
)
def test_bad_rows_fail_with_specific_reasons(mutation: dict[str, object], reason: str) -> None:
    row = _row("symbol_mapping")
    row.update(mutation)
    check = verify_trainable_row(row)
    assert not check.trainable
    assert check.rejection_reason == reason


def test_duplicate_prompt_fails_inside_batch() -> None:
    first = _row("char_cipher")
    second = dict(first)
    checks, report = verify_trainable_batch([first, second])
    assert checks[0].trainable
    assert not checks[1].trainable
    assert checks[1].rejection_reason == "duplicate_prompt"
    assert report.duplicate_prompt_count == 1


def test_family_wrappers_pass_valid_generated_rows() -> None:
    for family, wrapper in WRAPPERS.items():
        check = wrapper(_example(family))
        assert check.trainable, (family, check.rejection_reason)


def test_batch_report_all_pass_and_assert_raises_on_bad_row() -> None:
    rows = [_example(family) for family in available_teachers()]
    _checks, report = verify_trainable_batch(rows)
    assert report.trainable
    assert report.pass_rows == len(rows)
    assert report.fail_rows == 0
    with pytest.raises(ValueError, match="Batch not trainable"):
        bad = _row("unit_conversion")
        bad["answer"] = "wrong"
        assert_batch_trainable([bad])

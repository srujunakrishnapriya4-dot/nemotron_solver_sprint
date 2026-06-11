from __future__ import annotations

import random

from kaggle_anti086.training.day1_teacher_families import (
    available_teachers,
    generate_family_examples,
    get_teacher,
)
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers, validate_target_text


P0_P1 = {
    "symbol_mapping",
    "bit_manipulation",
    "char_cipher",
    "unit_conversion",
    "numeric_formula_safe",
    "word_cipher",
    "custom_numeral",
    "permutation_sorting",
}


def test_available_teachers_includes_all_p0_and_p1_families() -> None:
    assert P0_P1.issubset(set(available_teachers()))


def test_every_p0_p1_teacher_generates_deterministic_examples_for_fixed_seed() -> None:
    for family in P0_P1:
        first = generate_family_examples(family, 3, 123, ("easy", "medium"), ("direct", "table"))
        second = generate_family_examples(family, 3, 123, ("easy", "medium"), ("direct", "table"))
        assert first == second


def test_generated_examples_have_one_final_box_and_verify() -> None:
    for family, teacher in available_teachers().items():
        example = teacher.generate_synthetic(random.Random(7), "easy", "direct")
        assert count_boxed_answers(example.target_text) == 1
        assert example.target_text.endswith(f"\\boxed{{{example.answer}}}")
        assert validate_target_text(example.target_text, example.answer) == (True, None)
        result = teacher.verify_example(example)
        assert result.verification_status == "PASS"
        assert example.ambiguity_count == 0


def test_hard_negative_differs_and_is_rejected() -> None:
    for teacher in available_teachers().values():
        example = teacher.generate_synthetic(random.Random(8), "easy", "direct")
        negative = teacher.generate_hard_negative(example, random.Random(9))
        assert negative.rejected != negative.chosen
        assert negative.rejected_answer != negative.correct_answer
        assert negative.verifier_status != "PASS" or negative.reason_rejected


def test_symbol_mapping_generated_verifies_unseen_symbol_rejected_and_negative_differs() -> None:
    teacher = get_teacher("symbol_mapping")
    example = teacher.generate_synthetic(random.Random(1), "easy", "examples_query")
    assert teacher.verify_example(example).verification_status == "PASS"
    assert teacher.solve("[symbol_mapping] Examples: A->1; B->2. Query: C").verification_status == "FAIL"
    negative = teacher.generate_hard_negative(example, random.Random(2))
    assert negative.rejected != negative.chosen


def test_bit_manipulation_xor_not_reverse_and_invalid_rejections() -> None:
    teacher = get_teacher("bit_manipulation")
    assert teacher.solve("[bit_manipulation] Use 8-bit XOR. Input: 10101010 XOR 00001111. Answer as decimal.").answer == "165"
    assert teacher.solve("[bit_manipulation] Input: NOT 00001111. Answer as decimal.").verification_status == "FAIL"
    assert teacher.solve("[bit_manipulation] Use 8-bit bit reverse. Input: 00010000. Answer as decimal.").answer == "8"
    assert teacher.solve("[bit_manipulation] Convert binary to decimal. Input: 10102.").verification_status == "FAIL"


def test_char_cipher_caesar_atbash_and_ambiguous_shift() -> None:
    teacher = get_teacher("char_cipher")
    assert teacher.solve("[char_cipher] Caesar shift +3. Encode: cat").answer == "fdw"
    assert teacher.solve("[char_cipher] Atbash encode: abc").answer == "zyx"
    assert teacher.solve("[char_cipher] Examples: ab->bc. Query: cd").verification_status == "FAIL"


def test_unit_conversion_exact_integer_and_rounding_rejection() -> None:
    teacher = get_teacher("unit_conversion")
    assert teacher.solve("[unit_conversion] Convert 7 m to cm. Exact integer only.").answer == "700"
    assert teacher.solve("[unit_conversion] Convert 1 cm to m. Exact integer only.").verification_status == "FAIL"
    assert teacher.solve("[unit_conversion] Convert 1.5 m to cm. Exact integer only.").verification_status == "FAIL"


def test_numeric_formula_linear_and_ambiguous_rejections() -> None:
    teacher = get_teacher("numeric_formula_safe")
    assert teacher.solve("[numeric_formula_safe] Formula class: linear. Examples: 0->2; 1->5; 2->8. Query 5?").answer == "17"
    assert teacher.solve("[numeric_formula_safe] Examples: 1 -> 1; 2 -> 4. Query 3?").verification_status == "FAIL"
    assert teacher.solve("[numeric_formula_safe] Examples: 0->0; 1->1; 2->4. Query 3?").verification_status == "FAIL"


def test_word_cipher_reverse_and_semantic_rejection() -> None:
    teacher = get_teacher("word_cipher")
    assert teacher.solve("[word_cipher] Rule: reverse_words. Input: red blue green").answer == "green blue red"
    assert teacher.solve("[word_cipher] Rule: semantic_synonym. Input: big small").verification_status == "FAIL"


def test_custom_numeral_decode_encode_unseen_and_ambiguous_base_rejection() -> None:
    teacher = get_teacher("custom_numeral")
    assert teacher.solve("[custom_numeral] Base 4. Digits: A=0,B=1,C=2,D=3. Decode CBD to decimal.").answer == "39"
    assert teacher.solve("[custom_numeral] Base 4. Digits: A=0,B=1,C=2,D=3. Encode decimal 39.").answer == "CBD"
    assert teacher.solve("[custom_numeral] Base 4. Digits: A=0,B=1,C=2,D=3. Decode CBE to decimal.").verification_status == "FAIL"
    assert teacher.solve("[custom_numeral] Digits: A=0,B=1. Decode BA to decimal.").verification_status == "FAIL"


def test_permutation_sorting_ascending_ties_rejected_and_normalized_output() -> None:
    teacher = get_teacher("permutation_sorting")
    result = teacher.solve("[permutation_sorting] Sort numbers ascending. Input: 7,2,5")
    assert result.answer == "2,5,7"
    assert " " not in result.answer
    assert teacher.solve("[permutation_sorting] Sort numbers ascending. Input: 7,2,2").verification_status == "FAIL"



def test_equation_operator_safe_rules_and_rejections() -> None:
    teacher = get_teacher("equation_operator")
    assert teacher.solve("[equation_operator] Rule: add. Examples: 2 @ 3 = 5; 4 @ 1 = 5; 7 @ 2 = 9; 0 @ 9 = 9. Query: 6 @ 8?").answer == "14"
    assert teacher.solve("[equation_operator] Rule: sub_ab. Examples: 5 @ 2 = 3; 9 @ 4 = 5; 3 @ 1 = 2; 10 @ 7 = 3. Query: 8 @ 6?").answer == "2"
    assert teacher.solve("[equation_operator] Rule: sub_ba. Examples: 5 @ 2 = -3; 9 @ 4 = -5; 3 @ 1 = -2; 10 @ 7 = -3. Query: 8 @ 6?").answer == "-2"
    assert teacher.solve("[equation_operator] Rule: mul. Examples: 2 @ 3 = 6; 4 @ 1 = 4; 7 @ 2 = 14; 5 @ 5 = 25. Query: 6 @ 8?").answer == "48"
    assert teacher.solve("[equation_operator] Rule: exact_div_ab. Examples: 6 @ 3 = 2; 8 @ 4 = 2; 9 @ 3 = 3; 10 @ 5 = 2. Query: 12 @ 4?").answer == "3"
    assert teacher.solve("[equation_operator] Rule: exact_div_ab. Examples: 3 @ 2 = 1; 8 @ 4 = 2; 9 @ 3 = 3; 10 @ 5 = 2. Query: 12 @ 4?").verification_status == "FAIL"
    assert teacher.solve("[equation_operator] Rule: mod_ab. Examples: 3 @ 0 = 0; 8 @ 4 = 0; 9 @ 3 = 0; 10 @ 5 = 0. Query: 12 @ 4?").verification_status == "FAIL"


def test_equation_operator_ambiguous_affine_and_hard_negative() -> None:
    teacher = get_teacher("equation_operator")
    ambiguous = teacher.solve("[equation_operator] Examples: 1 @ 1 = 2; 2 @ 2 = 4; 3 @ 3 = 6; 4 @ 4 = 8. Query: 5 @ 5?")
    assert ambiguous.verification_status == "FAIL"
    affine = teacher.solve("[equation_operator] Rule: affine_small. Examples: 1 @ 2 = 9; 2 @ 1 = 8; 0 @ 3 = 10; 3 @ 0 = 7. Query: 4 @ 5?")
    assert affine.verification_status == "PASS"
    example = teacher.generate_synthetic(random.Random(44), "hard", "direct")
    assert example.ambiguity_count == 0
    negative = teacher.generate_hard_negative(example, random.Random(45))
    assert negative.rejected != negative.chosen


def test_sequence_pattern_safe_rules_and_rejections() -> None:
    teacher = get_teacher("sequence_pattern")
    assert teacher.solve("[sequence_pattern] Rule: arithmetic. Sequence: 3, 7, 11, 15, 19, ?").answer == "23"
    assert teacher.solve("[sequence_pattern] Rule: geometric_integer. Sequence: 2, 6, 18, 54, 162, ?").answer == "486"
    assert teacher.solve("[sequence_pattern] Rule: second_order_arithmetic. Sequence: 1, 3, 7, 13, 21, 31, ?").answer == "43"
    assert teacher.solve("[sequence_pattern] Rule: fibonacci_like. Sequence: 2, 3, 5, 8, 13, 21, 34, ?").answer == "55"
    assert teacher.solve("[sequence_pattern] Sequence: 1, 2, 3, ?").verification_status == "FAIL"
    assert teacher.solve("[sequence_pattern] Sequence: 1, 2, 3, 4, 5, 6, 7, 8, ?").verification_status == "FAIL"


def test_sequence_pattern_hard_negative_and_generated_zero_ambiguity() -> None:
    teacher = get_teacher("sequence_pattern")
    example = teacher.generate_synthetic(random.Random(46), "hard", "table")
    assert example.ambiguity_count == 0
    assert teacher.verify_example(example).verification_status == "PASS"
    negative = teacher.generate_hard_negative(example, random.Random(47))
    assert negative.rejected != negative.chosen


def test_gravity_numeric_safe_rules_and_rejections() -> None:
    teacher = get_teacher("gravity_numeric")
    assert teacher.solve("[gravity_numeric] Use F = m*g. Let g = 10. If m = 7, what is F?").answer == "70"
    assert teacher.solve("[gravity_numeric] Use distance = speed*time. If speed = 6 and time = 9, what is distance?").answer == "54"
    assert teacher.solve("[gravity_numeric] Use speed = distance/time. If distance = 54 and time = 9, what is speed?").answer == "6"
    assert teacher.solve("[gravity_numeric] Use F = m*g. If m = 7, what is F?").verification_status == "FAIL"
    assert teacher.solve("[gravity_numeric] Use F = m*g. Let g = 9.8. If m = 7, what is F?").verification_status == "FAIL"


def test_gravity_numeric_hard_negative_and_generated_zero_ambiguity() -> None:
    teacher = get_teacher("gravity_numeric")
    example = teacher.generate_synthetic(random.Random(48), "medium", "story")
    assert example.ambiguity_count == 0
    assert teacher.verify_example(example).verification_status == "PASS"
    negative = teacher.generate_hard_negative(example, random.Random(49))
    assert negative.rejected != negative.chosen

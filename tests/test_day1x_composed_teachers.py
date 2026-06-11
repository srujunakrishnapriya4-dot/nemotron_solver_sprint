from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_row
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers
from kaggle_anti086.training.day1x_composed_teachers import (
    COMPOSED_TEACHER_REGISTRY,
    REQUIRED_COMPOSED_FAMILIES,
    generate_composed_rows,
    validate_composed_row,
)


def _rows() -> list[dict[str, object]]:
    return generate_composed_rows(per_family=5, seed=123)


def test_registry_contains_all_required_composed_families() -> None:
    assert set(REQUIRED_COMPOSED_FAMILIES).issubset(COMPOSED_TEACHER_REGISTRY)


def test_every_family_generates_valid_trainable_rows() -> None:
    rows = _rows()
    assert len(rows) == 5 * len(REQUIRED_COMPOSED_FAMILIES)
    for family in REQUIRED_COMPOSED_FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        assert len(family_rows) == 5
        for row in family_rows:
            assert str(row["family"]).startswith("composed_")
            assert len(row["subfamilies"]) >= 2
            assert len(row["substeps"]) >= 2
            assert all(step["verification_status"] == "PASS" for step in row["substeps"])
            assert all(step["ambiguity_count"] == 0 for step in row["substeps"])
            assert row["verification_status"] == "PASS"
            assert row["ambiguity_count"] == 0
            assert count_boxed_answers(str(row["target_text"])) == 1
            assert str(row["target_text"]).endswith(f"\\boxed{{{row['answer']}}}")
            joined = "\n".join(str(row.get(key, "")) for key in ("prompt", "answer", "trace", "target_text"))
            assert "ABSTAIN" not in joined.upper()
            assert "TODO" not in joined.upper()
            assert validate_composed_row(row) == (True, [])
            assert verify_trainable_row(row).trainable


def test_validation_rejects_substep_failure_ambiguity_multiple_boxes_and_trailing_text() -> None:
    row = _rows()[0]
    failed = deepcopy(row)
    failed["substeps"][0]["verification_status"] = "FAIL"
    assert not validate_composed_row(failed)[0]
    ambiguous = deepcopy(row)
    ambiguous["ambiguity_count"] = 1
    assert not validate_composed_row(ambiguous)[0]
    multibox = deepcopy(row)
    multibox["target_text"] = "\\boxed{1}\\boxed{2}"
    multibox["trace"] = multibox["target_text"]
    assert not validate_composed_row(multibox)[0]
    trailing = deepcopy(row)
    trailing["target_text"] = f"{row['target_text']} after"
    trailing["trace"] = trailing["target_text"]
    assert not validate_composed_row(trailing)[0]


def test_family_specific_rejections_are_strict() -> None:
    rows = {row["family"]: deepcopy(row) for row in generate_composed_rows(per_family=1, seed=99)}

    custom = rows["composed_custom_numeral_arithmetic"]
    custom["substeps"][0]["input"] = f"{custom['substeps'][0]['input']}Z"
    assert not validate_composed_row(custom)[0]

    bit = rows["composed_bit_conversion"]
    bit["substeps"][0]["op"] = "not"
    bit["substeps"][0]["params"].pop("width", None)
    assert not validate_composed_row(bit)[0]

    unit = rows["composed_unit_formula"]
    unit["substeps"][0]["input"] = "1.5 m to cm"
    assert not validate_composed_row(unit)[0]

    sequence = rows["composed_sequence_operator"]
    sequence["substeps"][0]["params"]["sequence"] = [1, 2, 3]
    assert not validate_composed_row(sequence)[0]

    symbol = rows["composed_symbol_equation"]
    symbol["substeps"][0]["params"]["op"] = "__import__('os').system('echo unsafe')"
    assert not validate_composed_row(symbol)[0]


def test_generation_is_seed_deterministic_and_seed_sensitive() -> None:
    first = generate_composed_rows(per_family=2, seed=777)
    second = generate_composed_rows(per_family=2, seed=777)
    third = generate_composed_rows(per_family=2, seed=778)
    assert first == second
    assert [row["id"] for row in first] != [row["id"] for row in third]
    assert [row["prompt"] for row in first] != [row["prompt"] for row in third]


def test_no_python_eval_used_in_composed_teacher_module() -> None:
    source = Path("kaggle_anti086/training/day1x_composed_teachers.py").read_text(encoding="utf-8")
    assert "eval(" not in source

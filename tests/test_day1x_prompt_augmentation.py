from __future__ import annotations

from copy import deepcopy

import pytest

from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_row
from kaggle_anti086.training.day1x_composed_teachers import (
    COMPOSED_TEACHER_REGISTRY,
    REQUIRED_COMPOSED_FAMILIES,
    generate_composed_rows,
    validate_composed_row,
)
from kaggle_anti086.training.day1x_prompt_augmentation import (
    augment_prompt,
    supported_public_styles,
    validate_augmented_row,
)


def _row() -> dict[str, object]:
    return generate_composed_rows(per_family=1, seed=321)[0]


def test_supported_public_styles_has_required_public_coverage() -> None:
    styles = supported_public_styles()
    public = [style for style in styles if style.startswith("public_style_")]
    assert len(public) >= 8


def test_augment_prompt_preserves_answer_changes_prompt_and_sets_metadata() -> None:
    row = _row()
    augmented = augment_prompt(row, "public_style_verbose", seed=22)
    assert augmented["answer"] == row["answer"]
    assert augmented["target_text"] == row["target_text"]
    assert augmented["prompt"] != row["prompt"]
    assert augmented["parent_id"] == row["id"]
    assert augmented["augmentation_style"] == "public_style_verbose"
    assert validate_composed_row(augmented) == (True, [])
    assert verify_trainable_row(augmented).trainable
    assert validate_augmented_row(row, augmented) == (True, [])


def test_augmentation_is_deterministic_and_rejects_unsupported_style() -> None:
    row = _row()
    first = augment_prompt(row, "public_style_table", seed=7)
    second = augment_prompt(row, "public_style_table", seed=7)
    assert first == second
    with pytest.raises(ValueError, match="unsupported_public_style"):
        augment_prompt(row, "unsupported", seed=7)


def test_validate_augmented_row_rejects_prompt_unchanged_and_answer_changes() -> None:
    row = _row()
    augmented = augment_prompt(row, "public_style_minimal", seed=8)
    unchanged = deepcopy(augmented)
    unchanged["prompt"] = row["prompt"]
    assert not validate_augmented_row(row, unchanged)[0]
    changed_answer = deepcopy(augmented)
    changed_answer["answer"] = str(int(str(changed_answer["answer"])) + 1)
    assert not validate_augmented_row(row, changed_answer)[0]


def test_every_required_composed_family_supports_at_least_four_prompt_styles() -> None:
    for family in REQUIRED_COMPOSED_FAMILIES:
        assert len(COMPOSED_TEACHER_REGISTRY[family].supported_prompt_styles()) >= 4

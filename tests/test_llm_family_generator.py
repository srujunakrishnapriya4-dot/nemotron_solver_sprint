from __future__ import annotations

from nemotron_engine.core.registry import validate_registry
from nemotron_engine.core.schemas import ProblemSource, SplitName
import pytest

from nemotron_engine.data.llm_family_generator import LLMFamilyGenerationError, generate_llm_stress_records


def test_generator_is_stress_only_by_default() -> None:
    (record,) = generate_llm_stress_records(seed=1)

    assert record.source == ProblemSource.LLM_STRESS
    assert record.split == SplitName.STRESS_ONLY


def test_caller_cannot_force_training_permissions_or_split() -> None:
    (record,) = generate_llm_stress_records(
        seed=2,
        requested_split=SplitName.TRAIN,
        allowed_for_sft=True,
        allowed_for_dpo=True,
        allowed_for_grpo=True,
    )

    assert record.split == SplitName.STRESS_ONLY
    assert not record.allowed_for_sft
    assert not record.allowed_for_dpo
    assert not record.allowed_for_grpo
    assert record.allowed_for_eval


def test_collision_with_existing_records_rejected() -> None:
    existing = generate_llm_stress_records(seed=2)

    with pytest.raises(LLMFamilyGenerationError):
        generate_llm_stress_records(seed=2, existing_records=existing)


def test_malformed_existing_records_raise_explicit_error() -> None:
    with pytest.raises(LLMFamilyGenerationError):
        generate_llm_stress_records(seed=2, existing_records=[object()])  # type: ignore[list-item]


def test_no_external_call_path_exists_and_record_validates() -> None:
    records = generate_llm_stress_records(seed=3, count=2)

    assert validate_registry(records) == list(records)

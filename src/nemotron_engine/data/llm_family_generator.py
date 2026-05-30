"""Stress-only LLM family generator stub for Pass 4.

This module intentionally performs no external calls.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from nemotron_engine.core.registry import validate_registry
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName, stable_hash


class LLMFamilyGenerationError(ValueError):
    """Raised when LLM stress record generation is invalid."""


def generate_llm_stress_records(
    *,
    seed: int,
    count: int = 1,
    family_id: str = "llm_stress_family",
    primitive_family_id: str = "primitive:llm_stress",
    format_family_id: str = "format:stress",
    prompt_wrapper_id: str = "llm_stress_stub",
    parent_ids: Iterable[str] = (),
    existing_records: Iterable[ImmutableProblemRecord] = (),
    requested_split: SplitName | str | None = None,
    generator_params: Mapping[str, object] | None = None,
    **ignored_permissions: object,
) -> tuple[ImmutableProblemRecord, ...]:
    del requested_split, ignored_permissions
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise LLMFamilyGenerationError("seed must be an integer.")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise LLMFamilyGenerationError("count must be a positive integer.")
    parents = tuple(str(item) for item in parent_ids)
    existing_ids = _existing_ids(tuple(existing_records))
    records: list[ImmutableProblemRecord] = []
    for index in range(count):
        params = {
            "source": ProblemSource.LLM_STRESS.value,
            "split": SplitName.STRESS_ONLY.value,
            "family_id": family_id,
            "primitive_family_id": primitive_family_id,
            "format_family_id": format_family_id,
            "prompt_wrapper_id": prompt_wrapper_id,
            "parent_ids": parents,
            "seed": seed,
            "index": index,
            "generator_params": dict(generator_params or {}),
        }
        problem_id = f"llm-stress-{stable_hash(params)[:20]}"
        if problem_id in existing_ids or any(record.problem_id == problem_id for record in records):
            raise LLMFamilyGenerationError(f"generated problem_id collision: {problem_id}")
        records.append(
            ImmutableProblemRecord(
                problem_id=problem_id,
                source=ProblemSource.LLM_STRESS,
                source_hash=stable_hash({"llm_stress_stub": params}),
                family_id=family_id,
                primitive_family_id=primitive_family_id,
                format_family_id=format_family_id,
                prompt_wrapper_id=prompt_wrapper_id,
                split=SplitName.STRESS_ONLY,
                created_from="llm_family_generator_stub",
                parent_ids=parents,
                contamination_flags=(),
                allowed_for_sft=False,
                allowed_for_dpo=False,
                allowed_for_grpo=False,
                allowed_for_eval=True,
            )
        )
    validate_registry(records)
    return tuple(records)


def _existing_ids(existing_records: tuple[ImmutableProblemRecord, ...]) -> set[str]:
    ids: set[str] = set()
    for record in existing_records:
        problem_id = getattr(record, "problem_id", None)
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise LLMFamilyGenerationError("existing_records must contain records with valid problem_id values.")
        ids.add(problem_id)
    return ids


__all__ = ["LLMFamilyGenerationError", "generate_llm_stress_records"]

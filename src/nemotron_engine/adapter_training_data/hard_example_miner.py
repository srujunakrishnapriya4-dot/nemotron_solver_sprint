"""Hard-example mining and deterministic oversampling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import random

from nemotron_engine.competition_sprint import CompetitionProblem
from nemotron_engine.core.schemas import stable_hash

from .sft_formatter import SFTExample, format_family_tagged_example
from .solver_distillation import SolverDistillationRecord


class HardExampleMiningError(ValueError):
    """Raised when hard-example mining inputs are inconsistent."""


@dataclass(frozen=True)
class HardMiningConfig:
    family_caps: Mapping[str, int]
    seed: int = 1337
    default_cap: int = 500
    config_hash: str = ""

    def __post_init__(self) -> None:
        caps = {str(key): int(value) for key, value in sorted(self.family_caps.items())}
        if any(value <= 0 for value in caps.values()):
            raise HardExampleMiningError("family caps must be positive.")
        default_cap = int(self.default_cap)
        if default_cap <= 0:
            raise HardExampleMiningError("default_cap must be positive.")
        object.__setattr__(self, "family_caps", caps)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "default_cap", default_cap)
        _set_or_check_hash(self, "config_hash")


def mine_hard_examples(problems: Sequence[CompetitionProblem], records: Sequence[SolverDistillationRecord]) -> tuple[CompetitionProblem, ...]:
    """Select solver abstentions and high-value wrong families as hard examples."""

    by_id = {record.problem_id: record for record in records}
    hard: list[CompetitionProblem] = []
    for problem in problems:
        record = by_id.get(problem.problem_id)
        if record is None:
            continue
        if record.solver_prediction is None:
            hard.append(problem)
            continue
        if not record.correct and problem.family in {"unit_conversion", "gravity_numeric"}:
            hard.append(problem)
            continue
        if record.solver_prediction is None and problem.family in {"bit_manipulation", "equation_symbolic"}:
            hard.append(problem)
    return tuple(hard)


def oversample_hard_examples(problems: Sequence[CompetitionProblem], *, config: HardMiningConfig) -> tuple[SFTExample, ...]:
    """Oversample hard families deterministically up to configurable caps."""

    grouped: dict[str, list[CompetitionProblem]] = defaultdict(list)
    for problem in problems:
        grouped[problem.family].append(problem)

    rng = random.Random(config.seed)
    output: list[SFTExample] = []
    for family in sorted(grouped):
        rows = sorted(grouped[family], key=lambda item: item.problem_id)
        cap = config.family_caps.get(family, config.default_cap)
        if not rows:
            continue
        selected: list[CompetitionProblem] = list(rows[: min(len(rows), cap)])
        while len(selected) < cap:
            selected.append(rows[rng.randrange(len(rows))])
        for index, problem in enumerate(selected):
            example = format_family_tagged_example(problem, source="hard_oversampled")
            metadata = dict(example.metadata)
            metadata["hard_family_index"] = index
            metadata["hard_source_count"] = len(rows)
            output.append(
                SFTExample(
                    problem_id=example.problem_id,
                    family=example.family,
                    answer_kind=example.answer_kind,
                    source=example.source,
                    messages=example.messages,
                    metadata=metadata,
                )
            )
    return tuple(output)


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise HardExampleMiningError(f"{hash_field} does not match payload.")


__all__ = ["HardExampleMiningError", "HardMiningConfig", "mine_hard_examples", "oversample_hard_examples"]

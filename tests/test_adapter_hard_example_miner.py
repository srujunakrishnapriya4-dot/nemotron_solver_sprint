from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.adapter_training_data.hard_example_miner import (  # noqa: E402
    HardExampleMiningError,
    HardMiningConfig,
    mine_hard_examples,
    oversample_hard_examples,
)
from nemotron_engine.adapter_training_data.solver_distillation import SolverDistillationRecord  # noqa: E402
from nemotron_engine.competition_sprint import parse_competition_prompt  # noqa: E402


UNIT_PROMPT = """In Alice's Wonderland, a secret unit conversion is applied to measurements.
1.00 m becomes 3.00
2.00 m becomes 5.00
Now, convert the following measurement: 4.00 m"""

BIT_PROMPT = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
00000000 -> 11111111
11111111 -> 00000000
Now, determine the output for: 00110100"""


def test_mines_abstentions_and_wrong_numeric_cases() -> None:
    bit = parse_competition_prompt("bit", BIT_PROMPT, "11001011")
    unit = parse_competition_prompt("unit", UNIT_PROMPT, "9.00")
    records = (
        SolverDistillationRecord("bit", "bit_manipulation", "11001011", None, "abstain", False, {}),
        SolverDistillationRecord("unit", "unit_conversion", "9.00", "8.00", "solved", False, {}),
    )

    hard = mine_hard_examples((bit, unit), records)

    assert [problem.problem_id for problem in hard] == ["bit", "unit"]


def test_oversamples_by_family_cap_deterministically() -> None:
    bit = parse_competition_prompt("bit", BIT_PROMPT, "11001011")
    config = HardMiningConfig({"bit_manipulation": 3}, seed=7)

    first = oversample_hard_examples((bit,), config=config)
    second = oversample_hard_examples((bit,), config=config)

    assert len(first) == 3
    assert [row.example_hash for row in first] == [row.example_hash for row in second]


def test_hard_config_hash_forgery_rejected() -> None:
    config = HardMiningConfig({"bit_manipulation": 3})

    with pytest.raises(HardExampleMiningError):
        replace(config, config_hash="forged")

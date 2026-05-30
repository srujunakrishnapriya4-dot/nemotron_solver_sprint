from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.adapter_training_data.family_labeler import AdapterFamilyLabelerError, label_competition_family  # noqa: E402
from nemotron_engine.competition_sprint import parse_competition_prompt  # noqa: E402


BIT_PROMPT = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
00000000 -> 11111111
11111111 -> 00000000
Now, determine the output for: 00110100"""


def test_labels_parsed_competition_family() -> None:
    problem = parse_competition_prompt("p1", BIT_PROMPT, "11001011")
    label = label_competition_family(problem)

    assert label.problem_id == "p1"
    assert label.family == "bit_manipulation"
    assert label.answer_kind == "bitstring"
    assert label.example_count == 2


def test_labels_raw_prompt() -> None:
    label = label_competition_family(BIT_PROMPT, problem_id="raw", answer="11001011")

    assert label.problem_id == "raw"
    assert label.family == "bit_manipulation"


def test_family_label_hash_forgery_rejected() -> None:
    label = label_competition_family(BIT_PROMPT, problem_id="raw", answer="11001011")

    with pytest.raises(AdapterFamilyLabelerError):
        replace(label, label_hash="forged")

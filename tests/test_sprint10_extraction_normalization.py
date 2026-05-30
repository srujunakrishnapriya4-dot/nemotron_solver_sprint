from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_eval_anti086_vllm import answers_match, extract_answer, normalize_answer  # noqa: E402


def test_extractor_prefers_boxed_and_phrase_patterns() -> None:
    assert extract_answer("work\n\\boxed{101001}") == "101001"
    assert extract_answer("The answer is 42.") == "42"
    assert extract_answer("decrypted text is hello_world.") == "hello_world"


def test_extractor_finds_binary_and_numeric_answers() -> None:
    assert extract_answer("final bits: 010101") == "010101"
    assert extract_answer("final value: -3.50") == "-3.50"


def test_normalization_preserves_binary_and_roman_policy() -> None:
    assert normalize_answer("00101") == "00101"
    assert normalize_answer("xiv", "roman_numeral") == "XIV"
    assert answers_match("1.0000004", "1.0", "unit_conversion")
    assert not answers_match("00101", "101", "bit_manipulation")

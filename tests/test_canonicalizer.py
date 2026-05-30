from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import CanonicalizationError, DomainKind  # noqa: E402
from nemotron_engine.parsing.canonicalizer import canonicalize_prompt, infer_domain  # noqa: E402


def test_parses_simple_arrow_examples_and_target() -> None:
    problem = canonicalize_prompt("p", "Examples:\n1 -> 2\n3 -> 4\nTarget:\n5 -> ?")

    assert len(problem.examples) == 2
    assert problem.target.input_value == "5"
    assert problem.parse_confidence >= 0.92


def test_parses_input_output_style_and_bitstring_domain() -> None:
    problem = canonicalize_prompt("p", "Input: 001 -> Output: 010\nInput: 111 -> Output: 000\nTarget: 101 -> ?")

    assert problem.input_domain.kind is DomainKind.BITSTRING
    assert problem.output_domain.kind is DomainKind.BITSTRING


def test_infers_integer_and_bitstring_domains() -> None:
    assert infer_domain("12").kind is DomainKind.INTEGER
    assert infer_domain("101").kind is DomainKind.BITSTRING


def test_malformed_prompt_strict_raises_and_permissive_lowers_confidence() -> None:
    with pytest.raises(CanonicalizationError):
        canonicalize_prompt("p", "1 -> 2\nno target here")

    permissive = canonicalize_prompt("p", "1 -> 2\nno target here", permissive=True)
    assert permissive.parse_confidence < 0.92
    assert permissive.target.certain is False


def test_arbitrary_target_word_is_not_target_marker() -> None:
    problem = canonicalize_prompt("p", "target practice -> 2\n3 -> ?")

    assert len(problem.examples) == 1
    assert problem.examples[0].input_value == "target practice"
    assert problem.target.input_value == "3"


def test_target_requires_explicit_marker_or_rhs_question() -> None:
    with pytest.raises(CanonicalizationError):
        canonicalize_prompt("p", "1 -> 2\nTarget value -> 3")

    explicit = canonicalize_prompt("p", "1 -> 2\nTarget: 3 -> ?")
    assert explicit.target.input_value == "3"


def test_permissive_recoverable_unclear_prompt_remains_low_confidence() -> None:
    permissive = canonicalize_prompt("p", "1 -> 2\nunclear words\n3 -> ?", permissive=True)

    assert permissive.parse_confidence < 0.92
    assert permissive.parse_permission.value == "stress_eval_only"


def test_parse_hash_deterministic() -> None:
    raw = "1 -> 2\n3 -> ?"
    assert canonicalize_prompt("p", raw).parse_hash == canonicalize_prompt("p", raw).parse_hash

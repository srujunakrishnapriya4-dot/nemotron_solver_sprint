from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import DomainKind, SchemaValidationError  # noqa: E402
from nemotron_engine.programs.type_system import (  # noqa: E402
    domains_compatible,
    infer_value_domain,
    is_bitstring,
    is_cipher_sequence,
    is_decimal_string,
    is_digit_sequence,
    is_fraction_string,
    is_integer_string,
    is_symbol_sequence,
    require_domain,
)


def test_domain_detection_helpers() -> None:
    assert is_integer_string("-12")
    assert is_decimal_string("3.14")
    assert is_fraction_string("1/2")
    assert is_bitstring("1010")
    assert is_digit_sequence("123")
    assert is_symbol_sequence("ABC")
    assert is_cipher_sequence("@")


def test_infer_value_domain() -> None:
    assert infer_value_domain("-12").kind is DomainKind.INTEGER
    assert infer_value_domain("3.1").kind is DomainKind.DECIMAL
    assert infer_value_domain("1/2").kind is DomainKind.FRACTION
    assert infer_value_domain("101").kind is DomainKind.BITSTRING
    assert infer_value_domain("123").kind is DomainKind.DIGIT_SEQUENCE
    assert infer_value_domain("1 @ 2").kind is DomainKind.EQUATION


def test_require_domain_rejects_mismatch() -> None:
    assert require_domain("101", DomainKind.BITSTRING).kind is DomainKind.BITSTRING
    with pytest.raises(SchemaValidationError):
        require_domain("3.0", DomainKind.INTEGER)


def test_domains_compatible_is_conservative() -> None:
    assert domains_compatible(DomainKind.INTEGER, DomainKind.INTEGER)
    assert not domains_compatible(DomainKind.INTEGER, DomainKind.DECIMAL)
    assert not domains_compatible(DomainKind.CIPHER_SEQUENCE, DomainKind.SYMBOL_SEQUENCE)
    assert not domains_compatible(DomainKind.TOKEN_SEQUENCE, DomainKind.SYMBOL_SEQUENCE)
    assert infer_value_domain("alpha 123").kind is DomainKind.TOKEN_SEQUENCE

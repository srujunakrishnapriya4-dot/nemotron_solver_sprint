from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RuleHypothesis:
    solver_name: str
    rule_id: str
    prediction: str | None
    verified: bool
    reason: str
    metadata: dict


def route_hard_family(family: str) -> tuple[str, ...]:
    family = normalize_family_name(family)
    mapping = {
        "equation_symbolic": ("equation_symbolic_solver",),
        "bit_manipulation": ("bit_composition_solver",),
        "gravity_numeric": ("gravity_unit_solver",),
        "unit_conversion": ("gravity_unit_solver",),
        "cipher_text": ("cipher_symbol_solver",),
        "roman_numeral": (),
    }
    return mapping.get(family, ())


def normalize_family_name(family: str) -> str:
    aliases = {
        "binary": "bit_manipulation",
        "bit": "bit_manipulation",
        "cipher": "cipher_text",
        "cipher_digit": "cipher_text",
        "symbol_digit": "equation_symbolic",
        "numeral": "roman_numeral",
        "unit_conv": "unit_conversion",
        "gravity": "gravity_numeric",
        "equation": "equation_symbolic",
    }
    return aliases.get(str(family).strip().lower(), str(family).strip())

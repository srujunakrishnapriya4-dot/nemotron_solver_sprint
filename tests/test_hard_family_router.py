from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint.hard_family_router import normalize_family_name, route_hard_family  # noqa: E402


def test_hard_family_router_routes_hard_families() -> None:
    assert route_hard_family("bit_manipulation") == ("bit_composition_solver",)
    assert route_hard_family("unknown") == ()


def test_hard_family_router_normalizes_competition_aliases() -> None:
    assert normalize_family_name("binary") == "bit_manipulation"
    assert normalize_family_name("cipher_digit") == "cipher_text"
    assert normalize_family_name("symbol_digit") == "equation_symbolic"

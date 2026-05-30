from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.family_error_taxonomy import build_family_error_taxonomy  # noqa: E402


def test_family_taxonomy_downweights_roman_and_boosts_hard_families(tmp_path: Path) -> None:
    payload = build_family_error_taxonomy(tmp_path / "taxonomy.json", tmp_path / "taxonomy.md")

    assert payload["families"]["roman_numeral"]["weight"] == "downweight"
    assert payload["families"]["bit_manipulation"]["weight"] == "boost"
    assert payload["families"]["equation_symbolic"]["weight"] == "boost"


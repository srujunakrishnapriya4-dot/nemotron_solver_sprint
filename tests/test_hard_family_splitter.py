from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.hard_family_splitter import build_hard_family_split  # noqa: E402


def test_hard_family_splitter_prioritizes_hard() -> None:
    rows = [{"family": "bit_manipulation", "private_like_score": 0.9}, {"family": "roman_numeral", "private_like_score": 0.8}]
    assert len(build_hard_family_split(rows, limit_per_family=1)) == 2

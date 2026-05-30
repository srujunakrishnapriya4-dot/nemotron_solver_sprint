from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.composition_shift_builder import build_composition_shift  # noqa: E402


def test_composition_shift_builder_keeps_deep_rows() -> None:
    rows = [{"parameters": {"depth": 1}}, {"parameters": {"depth": 4}}]
    assert build_composition_shift(rows) == [rows[1]]

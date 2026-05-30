from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.surface_shift_builder import build_surface_shift  # noqa: E402


def test_surface_shift_builder_changes_prompt() -> None:
    row = {"id": "1", "prompt": "original"}
    shifted = build_surface_shift([row])[0]
    assert shifted["prompt"] != row["prompt"]
    assert shifted["surface_shift"] is True

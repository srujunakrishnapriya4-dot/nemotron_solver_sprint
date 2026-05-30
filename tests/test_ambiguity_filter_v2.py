from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.ambiguity_filter_v2 import filter_ambiguity_v2  # noqa: E402


def test_ambiguity_filter_v2_rejects_ambiguous() -> None:
    rows = [{"ambiguity_score": 0.1, "verifier_status": "verified"}, {"ambiguity_score": 0.9, "verifier_status": "verified"}]
    assert filter_ambiguity_v2(rows) == [rows[0]]

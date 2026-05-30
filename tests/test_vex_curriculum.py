from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.vex_curriculum import build_curriculum  # noqa: E402


def test_family_curriculum_balanced_and_excludes_unknown_trace() -> None:
    rows = [
        {"family": "roman", "corpus_type": "direct_raw"},
        {"family": "roman", "corpus_type": "direct_raw"},
        {"family": "bit", "corpus_type": "direct_raw"},
        {"family": "unknown", "corpus_type": "short_rule_trace", "metadata": {"verified_status": "verified_correct"}},
    ]

    selected = build_curriculum(rows, "micro_sanity", max_rows=3)

    assert {row["family"] for row in selected} >= {"roman", "bit"}
    assert all(row["corpus_type"] == "direct_raw" for row in selected)


from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.distribution_gap_analyzer import analyze_distribution_gap  # noqa: E402


def test_distribution_gap_report_contains_ceiling_conclusion(tmp_path: Path) -> None:
    report = analyze_distribution_gap(output_path=tmp_path / "gap.json")

    assert "bit_manipulation" in report["families"]
    assert "Do not blindly add 20K easy binary examples" in report["expected_conclusion"]
    assert report["families_needing_adversarial_rule_generation"]


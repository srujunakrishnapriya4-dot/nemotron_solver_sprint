from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.emergency_score_recovery.adapter_scorebook import default_scorebook_entries, load_scorebook, write_scorebook_template  # noqa: E402


def test_scorebook_records_baseline_and_failed_child() -> None:
    entries = default_scorebook_entries()

    assert entries[0].public_score == 0.85
    assert entries[0].status == "best_known"
    assert entries[1].public_score == 0.49
    assert entries[1].status == "rejected"


def test_scorebook_template_round_trips(tmp_path: Path) -> None:
    path = write_scorebook_template(tmp_path / "adapter_scorebook.json")
    entries = load_scorebook(path)

    assert len(entries) == 2
    assert {entry.status for entry in entries} == {"best_known", "rejected"}


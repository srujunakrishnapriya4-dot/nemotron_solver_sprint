from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.emergency_score_recovery.failed_training_audit import audit_failed_training_paths, known_failed_training_paths  # noqa: E402


def test_failed_training_audit_encodes_known_bad_paths() -> None:
    paths = known_failed_training_paths()

    assert paths[0].verdict == "DO_NOT_USE"
    assert "labels=input_ids" in paths[0].banned_going_forward
    assert paths[0].public_score == 0.49
    assert paths[1].verdict == "DO_NOT_SUBMIT_WITHOUT_STRONG_EVIDENCE"
    assert "huge_custom_child_without_known_good_structure" in paths[1].banned_going_forward


def test_failed_training_audit_reports_global_bans() -> None:
    report = audit_failed_training_paths()

    assert "blind_30b_continuation" in report["global_bans"]
    assert "fast_probe_sane_outputs" in report["required_proof"]


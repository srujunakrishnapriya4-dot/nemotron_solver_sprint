from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

import kaggle_final_candidate_selector as selector  # noqa: E402


def test_sprint9_selector_never_marks_micro_child_submittable() -> None:
    payload = selector.select_candidate()
    assert payload["allowed_submission_decision"] == "KEEP_PARENT_BASELINE"
    assert payload["required_for_child_submission"] == "NEED_V1_V2_EVIDENCE"
    assert payload["not_submission_ready"] is True


def test_sprint9_selector_rejects_v2_v3_in_this_sprint() -> None:
    _score, reasons = selector.score_stage("v2")
    assert "v2_v3_forbidden_in_sprint10" in reasons

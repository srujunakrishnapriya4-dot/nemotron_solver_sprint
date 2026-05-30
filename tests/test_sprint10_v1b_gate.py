from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_eval_anti086_vllm import compute_eval_metrics  # noqa: E402
from kaggle_final_candidate_selector import score_stage  # noqa: E402


def _gate_status(child_exact: float, parent_exact: float | None) -> str:
    summary = {"child_exact_match": child_exact, "parent_exact_match": parent_exact, "child_minus_parent_delta": None if parent_exact is None else child_exact - parent_exact}
    if parent_exact is None:
        return "INCONCLUSIVE_PARENT_MISSING"
    if (summary.get("parent_exact_match") or 0.0) == 0:
        return "INVALID_EVAL_PARENT_ZERO"
    if summary.get("child_exact_match", 0.0) == 0:
        return "FAIL_CHILD_ZERO"
    if summary["child_minus_parent_delta"] is not None and summary["child_minus_parent_delta"] < 0:
        return "FAIL_CHILD_REGRESSED"
    return "V1_EVAL_COMPLETE_NOT_PACKAGEABLE"


def test_v1b_gate_invalid_when_parent_exact_zero() -> None:
    assert _gate_status(0.5, 0.0) == "INVALID_EVAL_PARENT_ZERO"


def test_v1b_gate_fails_when_child_exact_zero() -> None:
    assert _gate_status(0.0, 0.5) == "FAIL_CHILD_ZERO"


def test_eval_metrics_parent_child_delta_is_meaningful() -> None:
    parent = [{"id": "a", "exact_match": True}, {"id": "b", "exact_match": False}]
    child = [{"id": "a", "prompt": "p", "prediction": "x", "exact_match": True}, {"id": "b", "prompt": "p", "prediction": "y", "exact_match": True}]
    metrics = compute_eval_metrics(child, mode="parent_calibrated_64", adapter_path="/child", decoding_params={}, parent_records=parent)
    assert metrics["parent_exact_match"] == 0.5
    assert metrics["child_exact_match"] == 1.0
    assert metrics["child_minus_parent_delta"] == 0.5


def test_final_selector_does_not_make_child_packageable_without_evidence() -> None:
    _score, reasons = score_stage("v1b")
    assert reasons

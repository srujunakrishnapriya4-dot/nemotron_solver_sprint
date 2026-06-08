from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day2_final_decision_report import build_day2_final_decision_report


TMP = Path("artifacts/test_tmp/day2_final_decision_report")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _reports(root: Path) -> dict[str, Path]:
    return {
        "hardening_audit": _write_json(root / "hardening.json", {"schema_version": 1, "status": "WARN"}),
        "final_solver_score_report": _write_json(
            root / "final_solver.json",
            {
                "schema_version": 1,
                "overall": {
                    "rows": 1792,
                    "raw_exact_match": 1383 / 1792,
                    "answerable_accuracy": 1.0,
                    "abstain_rows": 409,
                },
            },
        ),
        "abstain_policy_gate": _write_json(root / "policy_gate.json", {"schema_version": 1, "status": "WARN"}),
        "abstain_resolution_report": _write_json(
            root / "resolution.json",
            {"schema_version": 1, "policy_decision": {"decision": "ABSTAIN_NOT_ALLOWED_NEEDS_FALLBACK"}},
        ),
        "abstain_fallback_mining": _write_json(
            root / "mining.json",
            {
                "schema_version": 1,
                "recoverability_classes": {
                    "maybe_recoverable_requires_new_solver": {"rows": 150},
                },
            },
        ),
        "unsafe_abstain_forensic_audit": _write_json(
            root / "forensic.json",
            {
                "schema_version": 1,
                "forensic_classes": {
                    "possibly_recoverable_with_new_verified_solver": {"rows": 112},
                    "possibly_recoverable_with_better_parser": {"rows": 0},
                    "possibly_recoverable_with_relaxed_but_verified_rule": {"rows": 0},
                },
            },
        ),
    }


def _report(root: Path) -> dict:
    return build_day2_final_decision_report(input_paths=_reports(root))


def test_score_gap_math_is_correct():
    report = _report(_clean())

    assert report["score_state"]["total_rows"] == 1792
    assert report["score_state"]["current_correct"] == 1383
    assert report["score_state"]["target_correct_needed"] == 1667
    assert report["score_state"]["additional_correct_needed"] == 284


def test_projected_max_is_1645_over_1792():
    report = _report(_clean())

    assert report["recovery_state"]["projected_correct_if_all_plausible_solved"] == 1645
    assert report["recovery_state"]["projected_accuracy_if_all_plausible_solved"] == pytest.approx(1645 / 1792)
    assert report["recovery_state"]["projected_accuracy_if_all_plausible_solved"] == pytest.approx(0.91796875)


def test_remaining_gap_to_0_93_is_22_rows():
    report = _report(_clean())

    assert report["recovery_state"]["remaining_gap_to_0_93"] == 22
    assert report["recovery_state"]["can_reach_0_93_from_current_authorized_recovery"] is False


def test_no_0_93_evidence_remains_true():
    report = _report(_clean())

    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True
    assert report["leaderboard_claim"] is False


def test_submission_package_and_v2a_150_remain_blocked():
    report = _report(_clean())

    assert report["blocked_actions"]["v2a_150"] == "blocked"
    assert report["blocked_actions"]["package"] == "blocked"
    assert report["blocked_actions"]["submission"] == "blocked"
    assert report["blocked_actions"]["leaderboard_claim"] == "blocked"


def test_day3_plan_includes_required_families():
    report = _report(_clean())
    families = {item["family"] for item in report["day3_recovery_plan"]}

    assert {"equation_operator", "sequence_pattern", "custom_numeral"}.issubset(families)
    assert report["decision"]["authorized_day3_solver_families"] == ["equation_operator", "sequence_pattern", "custom_numeral"]


def test_extra_gap_reaudit_is_required():
    report = _report(_clean())

    assert report["decision"]["extra_gap_reaudit_required"] is True
    p1 = [item for item in report["day3_recovery_plan"] if item["family"] == "low_risk_extra_gap_recovery"][0]
    assert p1["rows_needed"] == 22


def test_missing_input_report_raises_loudly():
    root = _clean()
    paths = _reports(root)
    paths["abstain_fallback_mining"] = root / "missing.json"

    with pytest.raises(FileNotFoundError):
        build_day2_final_decision_report(input_paths=paths)


def test_output_is_deterministic():
    root = _clean()
    report_a = _report(root)
    report_b = _report(root)

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)

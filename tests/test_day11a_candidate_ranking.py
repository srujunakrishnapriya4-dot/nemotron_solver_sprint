from __future__ import annotations

from kaggle_anti086.training.day11a_candidate_ranking import build_candidate_ranking


def _report(score: float, *, rule: float | None = None, invalid: float = 0.0, no_family_zero: bool = True) -> dict:
    return {
        "status": "PASS",
        "private_like_512": score,
        "family_hard_512": score,
        "rule_holdout_512": score if rule is None else rule,
        "anti_leak_256": score,
        "invalid_answer_rate": invalid,
        "format_error_rate": 0.0,
        "no_family_zero": no_family_zero,
    }


def test_combined_beating_solver_and_adapter_allows_scale_to_150():
    ranking = build_candidate_ranking(
        base_report=_report(0.70),
        solver_report=_report(0.88),
        adapter_report=_report(0.82),
        combined_report=_report(0.91),
    )

    assert ranking["status"] == "PASS"
    assert ranking["decision"]["lora_path_alive"] is True
    assert ranking["decision"]["scale_to_150_allowed"] is True
    assert ranking["decision"]["scale_to_300_allowed"] is False
    assert ranking["decision"]["submit_recommended"] is False


def test_adapter_worse_than_base_blocks_scale_and_marks_lora_dead():
    ranking = build_candidate_ranking(
        base_report=_report(0.80),
        solver_report=_report(0.90),
        adapter_report=_report(0.60),
        combined_report=_report(0.85),
    )

    assert ranking["status"] == "WARN"
    assert ranking["decision"]["lora_path_alive"] is False
    assert ranking["decision"]["focus_solver_first"] is True
    assert ranking["decision"]["scale_to_150_allowed"] is False


def test_incomplete_eval_blocks_decision_without_fake_score():
    ranking = build_candidate_ranking(
        base_report={"status": "BLOCKED", "failures": ["missing"]},
        solver_report=_report(0.90),
        adapter_report=_report(0.81),
        combined_report=_report(0.92),
    )

    assert ranking["status"] == "BLOCKED"
    assert ranking["decision"]["lora_path_alive"] is None
    assert ranking["decision"]["scale_to_150_allowed"] is False
    assert "incomplete_eval_reports:base" in ranking["failures"]


def test_rule_holdout_collapse_blocks_scale():
    ranking = build_candidate_ranking(
        base_report=_report(0.70),
        solver_report=_report(0.80),
        adapter_report=_report(0.78),
        combined_report=_report(0.90, rule=0.20),
    )

    assert ranking["decision"]["scale_to_150_allowed"] is False
    combined = next(row for row in ranking["ranking"] if row["candidate"] == "combined_v2a_50")
    assert "rule_holdout_collapse" in combined["blockers"]


def test_family_zero_and_invalid_regression_are_surfaced():
    ranking = build_candidate_ranking(
        base_report=_report(0.70),
        solver_report=_report(0.80),
        adapter_report=_report(0.78),
        combined_report=_report(0.90, invalid=0.20, no_family_zero=False),
    )

    combined = next(row for row in ranking["ranking"] if row["candidate"] == "combined_v2a_50")
    assert "family_zero_detected" in combined["blockers"]
    assert "invalid_answer_rate_high" in combined["blockers"]
    assert ranking["decision"]["scale_to_150_allowed"] is False


def test_ranking_order_is_deterministic():
    ranking = build_candidate_ranking(
        base_report=_report(0.70),
        solver_report=_report(0.90),
        adapter_report=_report(0.80),
        combined_report=_report(0.85),
    )
    assert [row["candidate"] for row in ranking["ranking"]] == [
        "solver_only",
        "combined_v2a_50",
        "v2a_50_adapter_only",
        "base",
    ]

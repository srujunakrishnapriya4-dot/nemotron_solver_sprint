from __future__ import annotations

from nemotron_engine.scoring.answer_extractor import find_boxed_spans
from nemotron_engine.traces.templates import (
    render_direct_answer_trace,
    render_hard_rejection_trace,
    render_minimal_induced_rule_trace,
    render_minimal_known_rule_trace,
)


def test_positive_templates_contain_exactly_one_box() -> None:
    traces = [
        render_direct_answer_trace(answer="4"),
        render_minimal_known_rule_trace(rule_summary="add one", example_checks="2 examples passed", target_application="computed", answer="4"),
        render_minimal_induced_rule_trace(rule_summary="add one", example_checks="2 examples passed", target_application="computed", answer="4"),
    ]

    assert all(len(find_boxed_spans(trace)) == 1 for trace in traces)


def test_hard_rejection_template_contains_no_box() -> None:
    trace = render_hard_rejection_trace(reason="ambiguous")

    assert find_boxed_spans(trace) == ()


def test_templates_are_deterministic() -> None:
    first = render_minimal_known_rule_trace(rule_summary="r", example_checks="c", target_application="a", answer="9")
    second = render_minimal_known_rule_trace(rule_summary="r", example_checks="c", target_application="a", answer="9")

    assert first == second


def test_positive_templates_do_not_place_answer_before_application_marker() -> None:
    trace = render_minimal_induced_rule_trace(rule_summary="not the answer", example_checks="checks passed", target_application="computed", answer="42")

    assert "42" not in trace.split("Target application", maxsplit=1)[0]


def test_known_rule_template_rejects_answer_in_rule_summary() -> None:
    import pytest

    with pytest.raises(ValueError):
        render_minimal_known_rule_trace(rule_summary="answer is 42", example_checks="checks", target_application="computed", answer="42")


def test_known_rule_template_rejects_answer_in_example_checks() -> None:
    import pytest

    with pytest.raises(ValueError):
        render_minimal_known_rule_trace(rule_summary="rule", example_checks="42 appeared", target_application="computed", answer="42")


def test_induced_rule_template_rejects_answer_in_rule_summary() -> None:
    import pytest

    with pytest.raises(ValueError):
        render_minimal_induced_rule_trace(rule_summary="answer is 42", example_checks="checks", target_application="computed", answer="42")


def test_induced_rule_template_rejects_answer_in_example_checks() -> None:
    import pytest

    with pytest.raises(ValueError):
        render_minimal_induced_rule_trace(rule_summary="rule", example_checks="42 appeared", target_application="computed", answer="42")


def test_direct_answer_template_remains_valid() -> None:
    trace = render_direct_answer_trace(answer="42")

    assert len(find_boxed_spans(trace)) == 1

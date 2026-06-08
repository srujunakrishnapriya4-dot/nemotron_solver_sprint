from __future__ import annotations

from kaggle_anti086.eval.day2_generate_eval_reports import _prediction_record
from kaggle_anti086.solvers.symbol_mapping_solver import SymbolMappingSolver
from kaggle_anti086.training.day2_failure_report_v2 import classify_row


FIVE_DAY2_V2_FAILURES = [
    (
        "day5_family_hard_symbol_mapping_0005",
        "[family_hard #5] []{ -> =-+; }<> -> :|~; ]{} -> ~=-; query: ]}> output ?",
    ),
    (
        "day5_family_hard_symbol_mapping_0020",
        "[family_hard #20] !@# -> =-+; $%^ -> :|~; @#$ -> ~=-; query: @$^ output ?",
    ),
    (
        "day5_family_hard_symbol_mapping_0050",
        "[family_hard #50] !@# -> =-+; $%^ -> :|~; @#$ -> ~=-; query: @$^ output ?",
    ),
    (
        "day5_family_hard_symbol_mapping_0065",
        "[family_hard #65] []{ -> =-+; }<> -> :|~; ]{} -> ~=-; query: ]}> output ?",
    ),
    (
        "day5_family_hard_symbol_mapping_0080",
        "[family_hard #80] !@# -> =-+; $%^ -> :|~; @#$ -> ~=-; query: @$^ output ?",
    ),
]


def _row(prompt: str, *, row_id: str = "row", answer: str = ":~-") -> dict:
    return {
        "id": row_id,
        "family": "symbol_mapping",
        "subfamily": "reversal_substitution",
        "prompt": prompt,
        "answer": answer,
        "metadata": {"expected_solver_behavior": "answer"},
    }


def test_day2_v2_symbol_mapping_failures_preserve_leading_colon_and_score_correct():
    solver = SymbolMappingSolver()
    for row_id, prompt in FIVE_DAY2_V2_FAILURES:
        row = _row(prompt, row_id=row_id)
        result = solver.solve(row)
        assert not result.abstained
        candidate = result.candidates[0]
        assert candidate.answer == ":~-"
        assert candidate.verified is True

        prediction = _prediction_record(
            row=row,
            idx=0,
            dataset="family_eval",
            mode="solver_only",
            source="solver",
            raw_output=candidate.answer,
            selected_answer=candidate.answer,
            extra={
                "solver_candidate": candidate.answer,
                "solver_confidence": candidate.confidence,
                "solver_verification_status": "PASS",
                "route_reason": "verified_solver",
            },
        )

        assert prediction["extracted_answer"] == ":~-"
        assert prediction["normalized_answer"] == ":~-"
        assert prediction["correct"] is True
        assert classify_row(prediction) == "answerable_correct"


def test_symbol_direct_substitution_still_works():
    result = SymbolMappingSolver().solve(_row("!@ -> ab; @# -> bc; query: !# output ?", answer="ac"))

    assert not result.abstained
    assert result.candidates[0].answer == "ac"


def test_symbol_reversal_substitution_still_works():
    result = SymbolMappingSolver().solve(_row("!@ -> ba; @# -> cb; query: !# output ?", answer="ca"))

    assert not result.abstained
    assert result.candidates[0].answer == "ca"


def test_symbol_unseen_query_symbol_still_abstains():
    result = SymbolMappingSolver().solve(_row("!@ -> ab; #$ -> cd; query: !Z output ?"))

    assert result.abstained
    assert result.reason == "unknown_symbol_coverage_too_low"


def test_symbol_contradictory_mapping_still_abstains():
    result = SymbolMappingSolver().solve(_row("!@ -> ab; !# -> xy; query: !# output ?"))

    assert result.abstained
    assert result.reason == "mapping_conflict"


def test_symbol_duplicate_target_ambiguity_still_abstains():
    result = SymbolMappingSolver().solve(_row("!@ -> aa; #$ -> bc; query: !# output ?"))

    assert result.abstained
    assert result.reason == "ambiguous_mapping_disagreement"


def test_symbol_multiple_valid_transformations_still_abstain():
    result = SymbolMappingSolver().solve(_row("!@ -> ab; #$ -> cd; query: !# output ?"))

    assert result.abstained
    assert result.reason == "ambiguous_mapping_disagreement"


def test_abstain_policy_rows_remain_safe_in_failure_report_v2():
    row = {
        "expected": "ABSTAIN",
        "gold_answer": "ABSTAIN",
        "family": "symbol_mapping",
        "raw_output": "",
        "normalized_answer": "",
        "extracted_answer": "",
        "solver_candidate": "",
        "correct": False,
        "valid": True,
        "failure_reason": "all_solvers_abstained",
        "route_reason": "solver_abstained",
    }

    assert classify_row(row) == "abstain_correct"

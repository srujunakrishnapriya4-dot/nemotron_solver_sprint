from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training.day1_family_recovery_audit import (
    build_audit_report,
    classify_row,
    detect_prompt_patterns,
    is_abstain,
    normalize_family,
    read_json,
    read_jsonl,
    write_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]], malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    if malformed:
        lines.append("{bad")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_json_handles_missing_file(tmp_path: Path) -> None:
    warnings: list[str] = []
    assert read_json(tmp_path / "missing.json", warnings) is None
    assert warnings


def test_read_jsonl_handles_malformed_lines_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"id": "1"}], malformed=True)
    warnings: list[str] = []
    rows = read_jsonl(path, warnings)
    assert rows == [{"id": "1"}]
    assert any("Malformed JSONL line" in warning for warning in warnings)


def test_normalize_family_maps_aliases_consistently() -> None:
    assert normalize_family("Custom Numerals") == "custom_numeral"
    assert normalize_family("equation ops") == "equation_operator"
    assert normalize_family("Sequences") == "sequence_pattern"


def test_is_abstain_detects_abstain_empty_and_null_like_values() -> None:
    assert is_abstain("ABSTAIN")
    assert is_abstain("")
    assert is_abstain(None)
    assert is_abstain("null")


def test_concrete_answer_row_with_matching_prediction_is_correct() -> None:
    row = {"family": "custom_numeral", "answer": "42", "prediction": "42"}
    assert classify_row(row).current_outcome == "correct"


def test_concrete_answer_row_with_empty_prediction_is_wrong_or_abstain() -> None:
    row = {"family": "custom_numeral", "answer": "42", "prediction": ""}
    assessment = classify_row(row)
    assert assessment.row_type == "concrete_answer"
    assert assessment.current_outcome == "wrong_or_abstain"


def test_expected_abstain_row_with_empty_output_is_safe_abstain() -> None:
    row = {"family": "sequence_pattern", "answer": "ABSTAIN", "output": ""}
    assessment = classify_row(row)
    assert assessment.row_type == "expected_abstain"
    assert assessment.current_outcome == "safe_abstain"


def test_expected_abstain_row_with_non_empty_output_is_unsafe_answer_on_abstain() -> None:
    row = {"family": "sequence_pattern", "answer": "ABSTAIN", "output": "17"}
    assert classify_row(row).current_outcome == "unsafe_answer_on_abstain"


def test_indexed_placeholder_prompt_is_detected() -> None:
    assert "indexed_placeholder_pattern" in detect_prompt_patterns("sequence 0: 1, 2, 3, ?")


def test_sequence_expected_abstain_only_family_becomes_blocked_for_local_recovery(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "sequence_pattern", "answer": "ABSTAIN", "prediction": ""}],
    )
    report = build_audit_report(tmp_path)
    family = report["families"]["sequence_pattern"]
    assert not family["safe_to_attempt_local_recovery"]
    assert family["teacher_priority"] == "BLOCKED"


def test_equation_operator_expected_abstain_only_family_becomes_blocked_for_local_recovery(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "equation_operator", "answer": "ABSTAIN", "prediction": ""}],
    )
    report = build_audit_report(tmp_path)
    family = report["families"]["equation_operator"]
    assert not family["safe_to_attempt_local_recovery"]
    assert family["teacher_priority"] == "BLOCKED"


def test_custom_numeral_concrete_wrong_rows_become_p0_local_recovery_candidate(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "custom_numeral", "answer": "17", "prediction": "18"}],
    )
    report = build_audit_report(tmp_path)
    family = report["families"]["custom_numeral"]
    assert family["safe_to_attempt_local_recovery"]
    assert family["teacher_priority"] == "P0"
    assert "custom_numeral" in report["local_recovery_candidate_families"]


def test_numeric_formula_known_ambiguity_risk_becomes_high_risk(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "numeric_formula", "answer": "7", "prediction": "8"}],
    )
    report = build_audit_report(tmp_path)
    assert report["families"]["numeric_formula"]["risk_level"] == "HIGH"
    assert "numeric_formula" in report["high_risk_families"]


def test_missing_artifacts_produce_warn_not_crash(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "symbol_mapping", "answer": "A", "prediction": "B"}],
    )
    report = build_audit_report(tmp_path)
    assert report["status"] == "WARN"
    assert report["missing_artifacts"]


def test_count_mismatches_are_recorded_not_hidden(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "symbol_mapping", "answer": "A", "prediction": "B"}],
    )
    report_path = tmp_path / "artifacts/sprint11/day2_final_decision_report.json"
    report_path.write_text(json.dumps({"family_counts": {"symbol_mapping": 3}}), encoding="utf-8")
    report = build_audit_report(tmp_path)
    assert report["cross_artifact_consistency"]["symbol_mapping"]["count_mismatch"]


def test_global_recovery_math_does_not_count_expected_abstain_rows(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "sequence_pattern", "answer": "ABSTAIN", "prediction": ""}],
    )
    report = build_audit_report(tmp_path)
    assert report["global_recovery_math"]["estimated_local_recovery_pool_total"] == 0


def test_global_recovery_math_does_not_count_synthetic_teacher_rows_as_local_accuracy(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "equation_operator", "answer": "ABSTAIN", "prediction": ""}],
    )
    report = build_audit_report(tmp_path)
    assert "equation_operator" in report["synthetic_teacher_candidate_families"]
    assert report["global_recovery_math"]["estimated_correct_if_all_safe_local_recovery_solved"] == 1383


def test_synthetic_teacher_candidates_include_safe_deterministic_families(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "bit_manipulation", "answer": "1010", "prediction": ""}],
    )
    report = build_audit_report(tmp_path)
    assert "bit_manipulation" in report["synthetic_teacher_candidate_families"]


def test_safe_to_train_lora_is_always_false_in_phase1(tmp_path: Path) -> None:
    report = build_audit_report(tmp_path)
    assert report["decision"]["safe_to_train_lora"] is False


def test_safe_to_package_and_submit_are_false(tmp_path: Path) -> None:
    report = build_audit_report(tmp_path)
    assert report["decision"]["safe_to_package"] is False
    assert report["decision"]["safe_to_submit"] is False


def test_no_093_and_no_095_evidence_are_true(tmp_path: Path) -> None:
    report = build_audit_report(tmp_path)
    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True


def test_report_output_is_deterministic_for_fixed_inputs(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "custom_numeral", "answer": "17", "prediction": "18", "prompt": "custom numeral 7: ★ -> 1"}],
    )
    first = build_audit_report(tmp_path)
    second = build_audit_report(tmp_path)
    assert first == second


def test_protected_file_diffs_are_represented_but_not_modified(tmp_path: Path) -> None:
    report = build_audit_report(tmp_path)
    protected = report["repo_state"]["protected_file_diffs"]
    assert {"router.py", "solver_ensemble.py", "verifier.py"}.issubset(protected)


def test_examples_are_capped_to_avoid_huge_report_files(tmp_path: Path) -> None:
    rows = [
        {"family": "custom_numeral", "answer": str(index), "prediction": "", "prompt": f"custom numeral {index}: ★"}
        for index in range(20)
    ]
    _write_jsonl(tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl", rows)
    report = build_audit_report(tmp_path)
    examples = report["families"]["custom_numeral"]["examples"]["concrete_wrong_or_abstain"]
    assert len(examples) == 5



def test_report_contains_split_risk_fields_for_every_family(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "symbol_mapping", "answer": "A", "prediction": "A"}],
    )
    report = build_audit_report(tmp_path)
    for family_report in report["families"].values():
        assert family_report["local_recovery_risk"] in {"LOW", "MEDIUM", "HIGH"}
        assert family_report["synthetic_teacher_risk"] in {"LOW", "MEDIUM", "HIGH"}
        assert "local_recovery_reason" in family_report
        assert "synthetic_generation_reason" in family_report


def test_expected_abstain_only_family_disallows_local_router_integration(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "sequence_pattern", "answer": "ABSTAIN", "prediction": ""}],
    )
    report = build_audit_report(tmp_path)
    assert report["families"]["sequence_pattern"]["local_router_integration_allowed"] is False


def test_expected_abstain_only_family_can_be_theoretical_synthetic_candidate(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "equation_operator", "answer": "ABSTAIN", "prediction": ""}],
    )
    report = build_audit_report(tmp_path)
    assert "equation_operator" in report["theoretical_synthetic_teacher_candidates"]
    assert "synthetic_only_no_local_recovery_claim" in report["families"]["equation_operator"]["synthetic_generation_reason"]


def test_concrete_solved_family_becomes_evidence_backed_synthetic_candidate(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "symbol_mapping", "answer": "XYZ", "prediction": "XYZ"}],
    )
    report = build_audit_report(tmp_path)
    assert "symbol_mapping" in report["evidence_backed_synthetic_teacher_candidates"]


def test_local_recovery_route_status_blocked_when_safe_pool_zero(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "symbol_mapping", "answer": "XYZ", "prediction": "XYZ"}],
    )
    report = build_audit_report(tmp_path)
    assert report["global_recovery_math"]["estimated_local_recovery_pool_total"] == 0
    assert report["phase2_interpretation"]["local_recovery_route_status"] == "BLOCKED_NO_SAFE_POOL"


def test_teacher_distillation_route_status_open_when_synthetic_candidates_exist(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "symbol_mapping", "answer": "XYZ", "prediction": "XYZ"}],
    )
    report = build_audit_report(tmp_path)
    assert report["phase2_interpretation"]["teacher_distillation_route_status"] == "OPEN"


def test_phase2_local_recovery_flag_is_false(tmp_path: Path) -> None:
    report = build_audit_report(tmp_path)
    assert report["decision"]["safe_to_start_phase2_local_recovery"] is False


def test_phase2_teacher_bank_flag_is_true_when_candidates_exist(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "char_cipher", "answer": "ABC", "prediction": "ABC"}],
    )
    report = build_audit_report(tmp_path)
    assert report["decision"]["safe_to_start_phase2_teacher_bank"] is True


def test_phase2_verified_data_factory_flag_is_true_when_candidates_exist(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "bit_manipulation", "answer": "1010", "prediction": "1010"}],
    )
    report = build_audit_report(tmp_path)
    assert report["decision"]["safe_to_start_phase2_verified_data_factory"] is True


def test_safe_to_train_lora_remains_false_after_phase2_split(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "unit_conversion", "answer": "12", "prediction": "12"}],
    )
    report = build_audit_report(tmp_path)
    assert report["decision"]["safe_to_train_lora"] is False


def test_phase2_teacher_priority_order_non_empty_for_evidence_backed_candidates(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "symbol_mapping", "answer": "XYZ", "prediction": "XYZ"}],
    )
    report = build_audit_report(tmp_path)
    assert report["phase2_teacher_priority_order"]
    assert report["phase2_teacher_priority_order"][0]["family"] == "symbol_mapping"


def test_priority_order_does_not_make_equation_or_sequence_local_recovery(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [
            {"family": "equation_operator", "answer": "ABSTAIN", "prediction": ""},
            {"family": "sequence_pattern", "answer": "ABSTAIN", "prediction": ""},
        ],
    )
    report = build_audit_report(tmp_path)
    for family in ("equation_operator", "sequence_pattern"):
        assert report["families"][family]["safe_to_attempt_local_recovery"] is False
        assert report["families"][family]["local_router_integration_allowed"] is False
    actions = {item["family"]: item["recommended_phase2_action"] for item in report["phase2_teacher_priority_order"]}
    assert "synthetic_only_no_local_recovery" in actions["equation_operator"]
    assert "synthetic_only_no_local_recovery" in actions["sequence_pattern"]


def test_evidence_backed_and_theoretical_candidate_sets_are_disjoint(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [
            {"family": "symbol_mapping", "answer": "XYZ", "prediction": "XYZ"},
            {"family": "sequence_pattern", "answer": "ABSTAIN", "prediction": ""},
        ],
    )
    report = build_audit_report(tmp_path)
    evidence = set(report["evidence_backed_synthetic_teacher_candidates"])
    theoretical = set(report["theoretical_synthetic_teacher_candidates"])
    assert evidence.isdisjoint(theoretical)


def test_numeric_formula_synthetic_teacher_risk_not_low_without_ambiguity_safe_override(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "numeric_formula", "answer": "7", "prediction": "7"}],
    )
    report = build_audit_report(tmp_path)
    assert report["families"]["numeric_formula"]["synthetic_teacher_risk"] != "LOW"
    assert "strict_ambiguity_rejection" in report["families"]["numeric_formula"]["synthetic_generation_reason"]


def test_no_093_and_no_095_evidence_flags_remain_true_after_phase2_split(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/sprint11/day2_predictions/x_solver_only_predictions.jsonl",
        [{"family": "symbol_mapping", "answer": "XYZ", "prediction": "XYZ"}],
    )
    report = build_audit_report(tmp_path)
    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True

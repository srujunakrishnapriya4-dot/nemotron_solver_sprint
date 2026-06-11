from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training.day1_teacher_bank_report import build_teacher_bank_report


def test_teacher_bank_report_script_creates_json(tmp_path: Path) -> None:
    out = tmp_path / "teacher_bank.json"
    report = build_teacher_bank_report(tmp_path, per_family=2, seed=123, out=out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == report
    assert report["status"] in {"PASS", "WARN", "FAIL"}


def test_all_p0_teachers_pass_in_report() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=5, seed=123)
    for family in ("symbol_mapping", "bit_manipulation", "char_cipher", "unit_conversion", "numeric_formula_safe"):
        assert report["families"][family]["teacher_status"] == "PASS"


def test_report_safety_flags_remain_false() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=2, seed=123)
    assert report["safe_to_train_lora"] is False
    assert report["safe_to_package"] is False
    assert report["safe_to_submit"] is False
    assert report["leaderboard_claim"] is False
    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True


def test_local_router_integration_disallowed_for_every_family() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=2, seed=123)
    assert all(not item["local_router_integration_allowed"] for item in report["families"].values())


def test_teacher_bank_ready_only_when_gates_pass() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=100, seed=123)
    assert report["format_error_count"] == 0
    assert report["verification_fail_count"] == 0
    assert report["ambiguity_accept_count"] == 0
    assert report["total_synthetic_examples_verified"] >= 600
    assert report["teacher_bank_ready_for_verified_data_factory"] is True


def test_numeric_formula_safe_risk_is_not_low_in_report() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=2, seed=123)
    assert report["families"]["numeric_formula_safe"]["synthetic_teacher_risk"] != "LOW"


def test_small_stress_report_alternate_seed_passes_core_gates(tmp_path: Path) -> None:
    out = tmp_path / "teacher_bank_stress_small.json"
    report = build_teacher_bank_report(tmp_path, per_family=5, seed=999, out=out)
    assert out.exists()
    assert report["status"] in {"PASS", "WARN"}
    assert report["teacher_bank_ready_for_verified_data_factory"] is False
    assert report["format_error_count"] == 0
    assert report["verification_fail_count"] == 0
    assert report["ambiguity_accept_count"] == 0
    assert report["safe_to_train_lora"] is False
    assert report["safe_to_package"] is False
    assert report["safe_to_submit"] is False
    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True


def test_implemented_p2_pass_and_protocol_only_optional_does_not_fail_report() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=5, seed=456)
    assert report["status"] == "WARN"
    for family in ("equation_operator", "gravity_numeric", "sequence_pattern"):
        assert report["families"][family]["teacher_status"] == "PASS"
    assert set(report["families_not_implemented"]) == {"composed_hidden_style", "model_verified_fallback"}
    assert not report["families_fail"]


def test_generated_verified_and_hard_negative_counts_match_for_p0_p1() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=7, seed=321)
    for family, family_report in report["families"].items():
        if family_report["priority"] in {"P0", "P1"}:
            assert family_report["teacher_status"] == "PASS"
            assert family_report["synthetic_examples_generated"] == 7
            assert family_report["synthetic_examples_verified"] == 7
            assert family_report["hard_negatives_generated"] == 7


def test_phase2b_report_includes_implemented_p2_families() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=5, seed=654)
    for family in ("equation_operator", "gravity_numeric", "sequence_pattern"):
        assert report["families"][family]["priority"] == "P2"
        assert report["families"][family]["teacher_status"] == "PASS"
        assert report["families"][family]["local_router_integration_allowed"] is False


def test_phase2b_counts_match_implemented_family_count() -> None:
    per_family = 6
    report = build_teacher_bank_report(Path.cwd(), per_family=per_family, seed=777)
    implemented = [
        family
        for family, family_report in report["families"].items()
        if family_report["teacher_status"] == "PASS"
    ]
    assert len(implemented) == 11
    assert report["total_synthetic_examples_generated"] == len(implemented) * per_family
    assert report["total_synthetic_examples_verified"] == report["total_synthetic_examples_generated"]
    assert report["total_hard_negatives_generated"] == report["total_synthetic_examples_generated"]
    assert report["teacher_bank_ready_for_verified_data_factory"] is False


def test_phase2b_ready_true_at_stress_gate_scale() -> None:
    report = build_teacher_bank_report(Path.cwd(), per_family=55, seed=888)
    assert report["total_synthetic_examples_verified"] >= 600
    assert report["teacher_bank_ready_for_verified_data_factory"] is True
    assert report["format_error_count"] == 0
    assert report["verification_fail_count"] == 0
    assert report["ambiguity_accept_count"] == 0

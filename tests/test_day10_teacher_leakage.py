from __future__ import annotations

from kaggle_anti086.training.day10_build_solver_teacher_corpus import (
    build_learnability_audit,
    build_overlap_audit,
    normalized_prompt_hash,
    prompt_hash,
)


def _teacher_row(index: int = 0, **overrides):
    prompt = f"Context label taga: {index + 1} -> {index + 4}; Query: {index + 2}"
    row = {
        "id": f"day10_teacher_numeric_formula_{index}",
        "source_id": f"day10_source_numeric_formula_{index}",
        "family": "numeric_formula",
        "subfamily": "linear_offset",
        "prompt": prompt,
        "answer": str(index + 7),
        "expected_behavior": "answer",
        "solver_name": "solver_ensemble",
        "solver_source": "numeric_formula_solver",
        "solver_confidence": 0.9,
        "verified": True,
        "risk": "low",
        "metadata": {
            "source_rule_signature": f"day10_rule_signature_{index}",
            "source_leakage_group": f"day10_lg_{index}",
            "source_parameter_tuple_hash": f"day10_param_{index}",
            "source_prompt_template_signature": f"day10_template_{index}",
        },
    }
    row.update(overrides)
    return row


def _empty_forbidden():
    return {
        "source_id": set(),
        "prompt_hash": set(),
        "normalized_prompt_hash": set(),
        "rule_signature": set(),
        "leakage_group": set(),
        "eval_row_id": set(),
        "parameter_tuple_hash": set(),
        "prompt_template_signature": set(),
        "files": set(),
    }


def test_clean_overlap_audit_passes():
    audit = build_overlap_audit([_teacher_row()], _empty_forbidden())

    assert audit["status"] == "PASS"
    assert audit["failures"] == []


def test_overlap_audit_hard_fails_each_forbidden_surface():
    row = _teacher_row()
    forbidden = _empty_forbidden()
    forbidden["source_id"].add(row["source_id"])
    forbidden["eval_row_id"].add(row["source_id"])
    forbidden["prompt_hash"].add(prompt_hash(row["prompt"]))
    forbidden["normalized_prompt_hash"].add(normalized_prompt_hash(row["prompt"]))
    forbidden["rule_signature"].add(row["metadata"]["source_rule_signature"])
    forbidden["leakage_group"].add(row["metadata"]["source_leakage_group"])
    forbidden["parameter_tuple_hash"].add(row["metadata"].get("source_parameter_tuple_hash", ""))
    forbidden["prompt_template_signature"].add(row["metadata"].get("source_prompt_template_signature", ""))

    audit = build_overlap_audit([row], forbidden)

    assert audit["status"] == "FAIL"
    assert audit["source_id_overlap_count"] == 1
    assert audit["eval_row_id_overlap_count"] == 1
    assert audit["prompt_hash_overlap_count"] == 1
    assert audit["normalized_prompt_hash_overlap_count"] == 1
    assert audit["rule_signature_overlap_count"] == 1
    assert audit["leakage_group_overlap_count"] == 1
    assert audit["parameter_tuple_hash_overlap_count"] == 1
    assert audit["prompt_template_signature_overlap_count"] == 1


def test_learnability_audit_fails_template_dominance_and_near_duplicates():
    rows = [_teacher_row(i, prompt="1 -> 2; 2 -> 3. Now solve: 5") for i in range(20)]

    audit = build_learnability_audit(rows)

    assert audit["status"] == "FAIL"
    assert "one_template_dominates" in audit["failures"]
    assert "near_duplicate_prompt_rate_high" in audit["failures"]

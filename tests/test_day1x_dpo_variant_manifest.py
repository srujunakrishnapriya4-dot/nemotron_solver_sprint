from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training.day1x_dpo_variant_manifest import (
    NEXT_PHASE,
    VARIANT_NAMES,
    build_dpo_variant_manifest,
)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _audit(status: str = "PASS") -> dict[str, object]:
    return {
        "status": status,
        "train_pairs": 20000,
        "eval_pairs": 2000,
        "negative_type_counts": {"near_miss_answer": 1000},
        "rejected_accidentally_trainable_count": 0,
        "rejected_accidentally_correct_count": 0,
        "train_eval_prompt_overlap_count": 0,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "safety": {
            "chosen_format_error_count": 0,
            "chosen_verification_fail_count": 0,
            "chosen_ambiguity_count": 0,
            "chosen_abstain_count": 0,
        },
    }


def _variant(name: str, status: str = "PASS", row_count: int = 10, **overrides: object) -> dict[str, object]:
    data = {
        "status": status,
        "row_count": row_count,
        "format_error_count": 0,
        "verification_fail_count": 0,
        "ambiguity_accepted": 0,
        "abstain_accepted": 0,
        "duplicate_prompt_count": 0,
        "duplicate_id_count": 0,
        "recommended_for_day2_training": status in {"PASS", "WARN"},
    }
    data.update(overrides)
    return data


def _write_stage(tmp_path: Path, audit: dict[str, object] | None = None, variants: dict[str, dict[str, object]] | None = None) -> None:
    _write_json(tmp_path / "day1x_dpo_audit_report.json", audit or _audit())
    variants = variants or {name: _variant(name) for name in VARIANT_NAMES}
    for name in VARIANT_NAMES:
        _write_json(tmp_path / f"day1x_variant_manifest_{name}.json", variants.get(name, _variant(name, "MISSING", 0)))


def _build(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    return build_dpo_variant_manifest(tmp_path, {"require_precheck": False})


def test_final_manifest_pass_with_clean_small_mocked_dpo_and_variants(tmp_path: Path) -> None:
    _write_stage(tmp_path)
    manifest, decision = _build(tmp_path)
    assert manifest["status"] == "PASS"
    assert decision["status"] == "PASS"
    assert manifest["ready_for_next_phase"] is True


def test_final_manifest_warn_if_dpo_warn_but_safety_zero(tmp_path: Path) -> None:
    _write_stage(tmp_path, _audit("WARN"))
    manifest, _decision = _build(tmp_path)
    assert manifest["status"] == "WARN"


def test_final_manifest_fail_if_dpo_fails(tmp_path: Path) -> None:
    _write_stage(tmp_path, _audit("FAIL"))
    manifest, _decision = _build(tmp_path)
    assert manifest["status"] == "FAIL"


def test_final_manifest_fail_if_mixed_curriculum_missing(tmp_path: Path) -> None:
    variants = {name: _variant(name) for name in VARIANT_NAMES}
    variants["mixed_curriculum"] = _variant("mixed_curriculum", "MISSING", 0)
    _write_stage(tmp_path, variants=variants)
    manifest, _decision = _build(tmp_path)
    assert manifest["status"] == "FAIL"
    assert "mixed_curriculum_not_pass" in manifest["blocked_reasons"]


def test_final_manifest_fail_if_any_safety_counter_nonzero(tmp_path: Path) -> None:
    variants = {name: _variant(name) for name in VARIANT_NAMES}
    variants["direct"] = _variant("direct", format_error_count=1)
    _write_stage(tmp_path, variants=variants)
    manifest, _decision = _build(tmp_path)
    assert manifest["status"] == "FAIL"
    assert manifest["format_error_count"] == 1


def test_decision_report_flags_and_next_phase(tmp_path: Path) -> None:
    _write_stage(tmp_path)
    _manifest, decision = _build(tmp_path)
    assert decision["next_phase"] == NEXT_PHASE
    assert decision["training_authorized"] is True
    assert decision["package_authorized"] is False
    assert decision["submission_authorized"] is False
    assert decision["leaderboard_claim"] is False
    assert decision["no_0_93_evidence"] is True
    assert decision["no_0_95_evidence"] is True

from __future__ import annotations

from kaggle_anti086.training.day1_dpo_pair_manifest import build_dpo_manifest


def _audit(status: str = "PASS") -> dict[str, object]:
    return {
        "status": status,
        "train_pair_count": 5500,
        "eval_pair_count": 550,
        "rejected_pair_count": 0,
        "family_train_counts": {"unit_conversion": 500},
        "family_eval_counts": {"unit_conversion": 50},
        "reason_counts": {"failed_unit_wrong_factor": 550},
        "validation_fail_count": 0 if status == "PASS" else 1,
        "duplicate_pair_count": 0,
        "train_eval_prompt_overlap_count": 0,
        "rejected_accidentally_trainable_count": 0,
        "unknown_reason_count": 0,
        "unsafe_metadata_count": 0,
        "dpo_pairs_ready_for_phase6": status == "PASS",
    }


def test_manifest_status_pass_when_audit_passes() -> None:
    manifest, decision = build_dpo_manifest(input_files={}, output_files={}, audit=_audit("PASS"))
    assert manifest["status"] == "PASS"
    assert manifest["gates"]["dpo_pairs_ready_for_phase6"] is True
    assert decision["dpo_pairs_ready_for_phase6"] is True
    assert manifest["gates"]["phase5_ready_for_lora_training"] is False


def test_manifest_status_fail_when_audit_fails() -> None:
    manifest, decision = build_dpo_manifest(input_files={}, output_files={}, audit=_audit("FAIL"))
    assert manifest["status"] == "FAIL"
    assert manifest["gates"]["all_pairs_valid"] is False
    assert decision["dpo_pairs_ready_for_phase6"] is False


def test_phase5_ready_for_lora_training_is_always_false() -> None:
    manifest, decision = build_dpo_manifest(input_files={}, output_files={}, audit=_audit("PASS"))
    assert manifest["gates"]["phase5_ready_for_lora_training"] is False
    assert decision["phase5_ready_for_lora_training"] is False
    assert manifest["safe_to_train_lora"] is False

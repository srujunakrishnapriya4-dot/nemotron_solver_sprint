from __future__ import annotations

from typing import Any


def build_dpo_manifest(
    *,
    input_files: dict[str, str],
    output_files: dict[str, str],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = {
        "all_pairs_valid": audit.get("validation_fail_count") == 0,
        "zero_duplicate_pairs": audit.get("duplicate_pair_count") == 0,
        "zero_train_eval_overlap": audit.get("train_eval_prompt_overlap_count") == 0,
        "chosen_passes_gate": audit.get("validation_fail_count") == 0,
        "rejected_fails_gate": audit.get("rejected_accidentally_trainable_count") == 0,
        "all_families_present": _all_families_present(audit),
        "known_rejection_reasons": audit.get("unknown_reason_count") == 0,
        "dpo_pairs_ready_for_phase6": audit.get("dpo_pairs_ready_for_phase6") is True,
        "phase5_ready_for_lora_training": False,
    }
    status = "PASS" if all(value for key, value in gates.items() if key != "phase5_ready_for_lora_training") else "FAIL"
    if audit.get("status") == "WARN" and status == "PASS":
        status = "WARN"
    manifest = {
        "schema_version": 1,
        "created_by": "DAY1_PHASE5_DPO_PAIR_FACTORY",
        "status": status,
        "input_files": input_files,
        "output_files": output_files,
        "counts": {
            "train_pairs": audit.get("train_pair_count", 0),
            "eval_pairs": audit.get("eval_pair_count", 0),
            "rejected_pairs": audit.get("rejected_pair_count", 0),
        },
        "family_counts": {
            "train": audit.get("family_train_counts", {}),
            "eval": audit.get("family_eval_counts", {}),
        },
        "reason_counts": audit.get("reason_counts", {}),
        "gates": gates,
        "safe_to_train_lora": False,
        "safe_to_package": False,
        "safe_to_submit": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }
    decision = {
        "schema_version": 1,
        "created_by": "DAY1_PHASE5_DECISION_REPORT",
        "status": status,
        "dpo_pairs_ready_for_phase6": gates["dpo_pairs_ready_for_phase6"] and status in {"PASS", "WARN"},
        "phase5_ready_for_lora_training": False,
        "safe_to_train_lora": False,
        "safe_to_package": False,
        "safe_to_submit": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "blocked_actions": {
            "lora_training": "blocked_until_phase6_final_training_manifest",
            "package": "blocked",
            "submission": "blocked",
            "leaderboard_claim": "blocked",
        },
        "next_phase": "PHASE6_FINAL_TRAINING_MANIFEST_AND_LORA_AUTHORIZATION_GATE",
    }
    return manifest, decision


def _all_families_present(audit: dict[str, Any]) -> bool:
    train = audit.get("family_train_counts", {})
    eval_counts = audit.get("family_eval_counts", {})
    return bool(train) and set(train) == set(eval_counts) and all(train.values()) and all(eval_counts.values())

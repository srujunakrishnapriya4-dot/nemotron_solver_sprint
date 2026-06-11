from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from kaggle_anti086.training.day1_verified_data_schema import FactoryManifest


def build_manifest(
    *,
    seed: int,
    input_reports: dict[str, str],
    output_files: dict[str, str],
    counts: dict[str, int],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
) -> tuple[FactoryManifest, dict[str, Any]]:
    all_positive = [*train_rows, *eval_rows, *probe_rows]
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "eval": 0, "probe": 0})
    prompt_style_counts = Counter()
    difficulty_counts = Counter()
    for row in all_positive:
        family_counts[str(row["family"])][str(row["split"])] += 1
        prompt_style_counts[str(row["prompt_style"])] += 1
        difficulty_counts[str(row["difficulty"])] += 1
    rejection_counts = Counter(row.get("reason", "unknown") for row in rejected_rows)
    audit_statuses = {name: audit["status"] for name, audit in audits.items()}
    gates = _gates(audit_statuses, counts)
    manifest_status = "PASS" if gates["phase4_ready_for_phase5"] else "FAIL"
    manifest = FactoryManifest(
        schema_version=1,
        created_by="DAY1_PHASE4_VERIFIED_SYNTHETIC_DATA_FACTORY",
        seed=seed,
        generated_at=datetime.now(timezone.utc).isoformat(),
        input_reports=input_reports,
        output_files=output_files,
        counts=counts,
        family_counts={family: dict(splits) for family, splits in sorted(family_counts.items())},
        prompt_style_counts=dict(sorted(prompt_style_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        rejection_counts=dict(sorted(rejection_counts.items())),
        gates=gates,
        safe_to_train_lora=False,
        safe_to_package=False,
        safe_to_submit=False,
        leaderboard_claim=False,
        no_0_93_evidence=True,
        no_0_95_evidence=True,
    )
    decision = {
        "schema_version": 1,
        "created_by": "DAY1_PHASE4_DECISION_REPORT",
        "status": manifest_status,
        "audits": audit_statuses,
        "gates": gates,
        "phase4_ready_for_phase5": gates["phase4_ready_for_phase5"],
        "phase4_ready_for_lora_training": False,
        "safe_to_train_lora": False,
        "safe_to_package": False,
        "safe_to_submit": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "blocked_actions": {
            "lora_training": "blocked_until_phase5_phase6_dataset_and_training_manifest",
            "package": "blocked",
            "submission": "blocked",
            "leaderboard_claim": "blocked",
        },
        "next_phase": "PHASE5_HARD_NEGATIVE_DPO_PAIR_FACTORY_AND_DATASET_GATE",
    }
    return manifest, decision


def manifest_status(manifest: FactoryManifest, audits: dict[str, dict[str, Any]]) -> str:
    del audits
    return "PASS" if manifest.gates.get("phase4_ready_for_phase5") else "FAIL"


def _gates(audit_statuses: dict[str, str], counts: dict[str, int]) -> dict[str, bool]:
    format_ok = audit_statuses.get("format") == "PASS"
    dedup_ok = audit_statuses.get("dedup_leakage") == "PASS"
    family_ok = audit_statuses.get("family_balance") == "PASS"
    teacher_ok = audit_statuses.get("teacher_verification") == "PASS"
    diversity_ok = audit_statuses.get("prompt_diversity") in {"PASS", "WARN"}
    expected_counts_ok = (
        counts.get("train_rows", 0) > 0
        and counts.get("eval_rows", 0) > 0
        and counts.get("probe_rows", 0) > 0
        and counts.get("hard_negative_pairs", 0) > 0
    )
    phase5_ready = format_ok and dedup_ok and family_ok and teacher_ok and diversity_ok and expected_counts_ok
    return {
        "all_rows_trainable": teacher_ok,
        "zero_format_errors": format_ok,
        "zero_ambiguity": teacher_ok,
        "zero_duplicate_prompts": dedup_ok,
        "zero_split_leakage": dedup_ok,
        "all_families_present": family_ok,
        "prompt_diversity_ok": diversity_ok,
        "hard_negatives_ok": counts.get("hard_negative_pairs", 0) > 0,
        "phase4_ready_for_phase5": phase5_ready,
        "phase4_ready_for_lora_training": False,
    }

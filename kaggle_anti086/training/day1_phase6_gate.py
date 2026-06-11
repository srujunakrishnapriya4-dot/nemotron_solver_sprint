from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def evaluate_training_authorization(teacher_quality: dict[str, Any] | None, lora_manifest: dict[str, Any] | None) -> dict[str, Any]:
    blocked: list[str] = []
    warnings: list[str] = []
    if not teacher_quality:
        blocked.append("teacher_quality_missing")
    if not lora_manifest:
        blocked.append("lora_manifest_missing")
    if blocked:
        return {"training_authorized": False, "safe_to_train_lora": False, "blocked_reasons": blocked, "warnings": warnings}

    if teacher_quality.get("status") == "FAIL":
        blocked.append("teacher_quality_fail")
    if lora_manifest.get("status") == "FAIL":
        blocked.append("lora_manifest_fail")
    if teacher_quality.get("teacher_ready_for_lora") is not True:
        blocked.append("teacher_ready_for_lora_false")
    if lora_manifest.get("training_authorized") is not True:
        blocked.append("lora_manifest_training_authorized_false")
    if int(teacher_quality.get("total_train_rows", 0)) < 10000:
        blocked.append("total_train_rows_below_10000")
    if int(teacher_quality.get("total_eval_rows", 0)) < 1000:
        blocked.append("total_eval_rows_below_1000")
    if int(lora_manifest.get("dpo_train_pair_count", 0)) < 3000:
        blocked.append("dpo_train_pairs_below_3000")
    if len(teacher_quality.get("families_pass", [])) < 6:
        blocked.append("fewer_than_6_families_pass")
    for key, reason in (
        ("format_error_count", "format_errors_nonzero"),
        ("verification_fail_count", "verification_failures_nonzero"),
        ("ambiguity_accepted", "ambiguity_accepted_nonzero"),
        ("abstain_placeholders_accepted", "abstain_placeholders_accepted_nonzero"),
        ("duplicate_prompt_count", "duplicate_prompt_count_nonzero"),
        ("split_leakage_count", "split_leakage_count_nonzero"),
    ):
        if int(teacher_quality.get(key, 0)) > 0:
            blocked.append(reason)
    for source_name, doc in (("teacher_quality", teacher_quality), ("lora_manifest", lora_manifest)):
        if doc.get("package_authorized") is True:
            blocked.append(f"{source_name}_package_authorized_true")
        if doc.get("submission_authorized") is True:
            blocked.append(f"{source_name}_submission_authorized_true")
        if doc.get("leaderboard_claim") is True:
            blocked.append(f"{source_name}_leaderboard_claim_true")
        if doc.get("no_0_93_evidence") is not True:
            blocked.append(f"{source_name}_no_0_93_evidence_false")
        if doc.get("no_0_95_evidence") is not True:
            blocked.append(f"{source_name}_no_0_95_evidence_false")
    if teacher_quality.get("status") == "WARN":
        warnings.append("teacher_quality_warn")
    if lora_manifest.get("status") == "WARN":
        warnings.append("lora_manifest_warn")
    authorized = not blocked
    return {
        "training_authorized": authorized,
        "safe_to_train_lora": authorized,
        "blocked_reasons": blocked,
        "warnings": warnings,
    }


def build_phase6_decision_report(
    *,
    artifacts_dir: Path,
    teacher_quality_path: Path,
    lora_manifest_path: Path,
    out: Path | None = None,
) -> dict[str, Any]:
    teacher_quality = _read_json(teacher_quality_path) if teacher_quality_path.exists() else None
    lora_manifest = _read_json(lora_manifest_path) if lora_manifest_path.exists() else None
    gate = evaluate_training_authorization(teacher_quality, lora_manifest)
    report = {
        "schema_version": 1,
        "created_by": "DAY1_PHASE6_FINAL_TRAINING_GATE",
        "status": "PASS" if gate["training_authorized"] else "FAIL",
        "teacher_quality_status": teacher_quality.get("status") if teacher_quality else "MISSING",
        "lora_manifest_status": lora_manifest.get("status") if lora_manifest else "MISSING",
        "training_authorized": gate["training_authorized"],
        "training_authorization_reason": "all_evidence_gates_passed_training_only" if gate["training_authorized"] else "blocked_by_evidence_gate",
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "next_phase": "PHASE7_LORA_TRAINING_AND_ADAPTER_EVAL",
        "blocked_reasons": gate["blocked_reasons"],
        "warnings": gate["warnings"],
        "required_training_files": {
            "sft_train": str(artifacts_dir / "phase4_verified_sft_train.jsonl"),
            "sft_eval": str(artifacts_dir / "phase4_verified_sft_eval.jsonl"),
            "probe": str(artifacts_dir / "phase4_verified_eval_probe.jsonl"),
            "dpo_train": str(artifacts_dir / "phase5_dpo_train_pairs.jsonl"),
            "dpo_eval": str(artifacts_dir / "phase5_dpo_eval_pairs.jsonl"),
        },
        "minimum_training_acceptance_before_package": {
            "adapter_eval_required": True,
            "must_beat_base": True,
            "must_beat_v2a_50": True,
            "must_pass_probe_format": True,
            "must_have_no_box_format_regression": True,
            "must_not_package_if_eval_worse": True,
        },
        "safe_to_train_lora": gate["safe_to_train_lora"],
        "safe_to_package": False,
        "safe_to_submit": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1 Phase6 final training authorization decision.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--teacher-quality", type=Path, required=True)
    parser.add_argument("--lora-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_phase6_decision_report(
        artifacts_dir=args.artifacts_dir,
        teacher_quality_path=args.teacher_quality,
        lora_manifest_path=args.lora_manifest,
        out=args.out,
    )
    print(json.dumps({"status": report["status"], "training_authorized": report["training_authorized"]}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())

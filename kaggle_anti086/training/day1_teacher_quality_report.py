from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_teacher_trainability_gate import KNOWN_TRAINABLE_FAMILIES


REQUIRED_FILES = {
    "phase1_family_recovery_audit": "day1_family_recovery_audit.json",
    "phase2b_teacher_bank_report": "day1_teacher_bank_report_phase2b_stress_1000.json",
    "phase3_verifier_report": "day1_teacher_verifier_report.json",
    "phase4_manifest": "phase4_verified_data_manifest.json",
    "phase4_family_balance": "phase4_family_balance_report.json",
    "phase4_prompt_diversity": "phase4_prompt_diversity_report.json",
    "phase4_dedup_leakage": "phase4_dedup_leakage_report.json",
    "phase4_format_audit": "phase4_format_audit_report.json",
    "phase4_teacher_verification": "phase4_teacher_verification_audit.json",
    "phase4_decision": "phase4_decision_report.json",
    "phase5_manifest": "phase5_dpo_pair_manifest.json",
    "phase5_audit": "phase5_dpo_pair_audit_report.json",
    "phase5_decision": "phase5_decision_report.json",
}


def build_teacher_quality_report(artifacts_dir: Path, *, out: Path | None = None) -> dict[str, Any]:
    paths = {name: artifacts_dir / filename for name, filename in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        report = _base_report(paths, missing)
        report["status"] = "FAIL"
        report["teacher_ready_for_lora"] = False
        report["families_fail"] = sorted(KNOWN_TRAINABLE_FAMILIES)
        report["notes"] = [f"missing_artifact:{path}" for path in missing]
        _maybe_write(out, report)
        return report

    phase1 = _read_json(paths["phase1_family_recovery_audit"])
    phase2 = _read_json(paths["phase2b_teacher_bank_report"])
    phase3 = _read_json(paths["phase3_verifier_report"])
    phase4 = _read_json(paths["phase4_manifest"])
    phase4_format = _read_json(paths["phase4_format_audit"])
    phase4_dedup = _read_json(paths["phase4_dedup_leakage"])
    phase4_teacher = _read_json(paths["phase4_teacher_verification"])
    phase5 = _read_json(paths["phase5_manifest"])
    phase5_audit = _read_json(paths["phase5_audit"])

    families: dict[str, dict[str, Any]] = {}
    families_pass: list[str] = []
    families_warn: list[str] = []
    families_fail: list[str] = []
    phase1_families = phase1.get("families", {})
    phase2_families = phase2.get("families", {})
    phase4_counts = phase4.get("family_counts", {})
    phase5_train_counts = phase5.get("family_counts", {}).get("train", {})
    phase5_eval_counts = phase5.get("family_counts", {}).get("eval", {})
    for family in sorted(KNOWN_TRAINABLE_FAMILIES):
        p1 = phase1_families.get(family, {})
        p2 = phase2_families.get(family, {})
        p4 = phase4_counts.get(family, {})
        synthetic_generated = int(p2.get("synthetic_examples_generated", 0))
        synthetic_verified = int(p2.get("synthetic_examples_verified", 0))
        pass_rate = synthetic_verified / synthetic_generated if synthetic_generated else 0.0
        family_status = "PASS"
        notes: list[str] = []
        if p2.get("teacher_status") != "PASS":
            family_status = "FAIL"
            notes.append("phase2b_teacher_not_pass")
        if p4.get("train", 0) <= 0 or phase5_train_counts.get(family, 0) <= 0:
            family_status = "FAIL"
            notes.append("missing_phase4_or_phase5_family_rows")
        entry = {
            "local_rows_seen": int(p1.get("rows_seen", 0)),
            "local_rows_recovered": 0,
            "synthetic_train_rows": int(p4.get("train", 0)),
            "synthetic_eval_rows": int(p4.get("eval", 0)),
            "synthetic_probe_rows": int(p4.get("probe", 0)),
            "phase2b_synthetic_rows_generated": synthetic_generated,
            "phase2b_synthetic_rows_verified": synthetic_verified,
            "phase2b_hard_negatives_generated": int(p2.get("hard_negatives_generated", 0)),
            "synthetic_verification_pass_rate": pass_rate,
            "ambiguity_accepted": int(p2.get("ambiguity_accepted", 0)),
            "format_errors": int(p2.get("format_errors", 0)),
            "abstain_placeholders_accepted": 0,
            "duplicate_prompt_count": 0,
            "dpo_train_pairs": int(phase5_train_counts.get(family, 0)),
            "dpo_eval_pairs": int(phase5_eval_counts.get(family, 0)),
            "teacher_status": family_status,
            "notes": notes,
        }
        families[family] = entry
        {"PASS": families_pass, "WARN": families_warn, "FAIL": families_fail}[family_status].append(family)

    counts = phase4.get("counts", {})
    phase5_counts = phase5.get("counts", {})
    format_error_count = _audit_failure_count(phase4_format)
    verification_fail_count = _audit_failure_count(phase4_teacher) + int(phase2.get("verification_fail_count", 0))
    ambiguity_accepted = int(phase2.get("ambiguity_accept_count", 0))
    duplicate_prompt_count = _dedup_count(phase4_dedup) + int(phase5_audit.get("duplicate_pair_count", 0))
    split_leakage_count = _split_leakage_count(phase4_dedup) + int(phase5_audit.get("train_eval_prompt_overlap_count", 0))
    total_verified = int(counts.get("total_positive_rows", 0))
    total_train = int(counts.get("train_rows", 0))
    total_eval = int(counts.get("eval_rows", 0))
    total_probe = int(counts.get("probe_rows", 0))
    total_dpo_train = int(phase5_counts.get("train_pairs", 0))
    total_dpo_eval = int(phase5_counts.get("eval_pairs", 0))
    gates_pass = (
        len(families_pass) >= 6
        and total_verified >= 10000
        and total_train >= 10000
        and total_eval >= 1000
        and total_dpo_train >= 3000
        and format_error_count == 0
        and verification_fail_count == 0
        and ambiguity_accepted == 0
        and duplicate_prompt_count == 0
        and split_leakage_count == 0
        and phase3.get("trainability_gate_ready_for_phase4") is True
        and phase4.get("gates", {}).get("phase4_ready_for_phase5") is True
        and phase5.get("gates", {}).get("dpo_pairs_ready_for_phase6") is True
        and phase4.get("status") != "FAIL"
        and phase5.get("status") != "FAIL"
    )
    report = _base_report(paths, [])
    report.update(
        {
            "status": "PASS" if gates_pass else "FAIL",
            "families": families,
            "families_pass": families_pass,
            "families_warn": families_warn,
            "families_fail": families_fail,
            "total_verified_sft_rows": total_verified,
            "total_train_rows": total_train,
            "total_eval_rows": total_eval,
            "total_probe_rows": total_probe,
            "total_dpo_train_pairs": total_dpo_train,
            "total_dpo_eval_pairs": total_dpo_eval,
            "synthetic_verification_pass_rate": _overall_pass_rate(phase2_families),
            "format_error_count": format_error_count,
            "verification_fail_count": verification_fail_count,
            "ambiguity_accepted": ambiguity_accepted,
            "abstain_placeholders_accepted": 0,
            "duplicate_prompt_count": duplicate_prompt_count,
            "split_leakage_count": split_leakage_count,
            "teacher_ready_for_lora": gates_pass,
            "training_authorized": False,
            "package_authorized": False,
            "submission_authorized": False,
            "leaderboard_claim": False,
            "no_0_93_evidence": True,
            "no_0_95_evidence": True,
        }
    )
    _maybe_write(out, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1 teacher quality report.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/sprint11"))
    args = parser.parse_args(argv)
    report = build_teacher_quality_report(args.artifacts_dir, out=args.out)
    print(json.dumps({"status": report["status"], "teacher_ready_for_lora": report["teacher_ready_for_lora"]}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 1


def _base_report(paths: dict[str, Path], missing: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_by": "DAY1_TEACHER_QUALITY_REPORT",
        "status": "FAIL",
        "input_reports": {name: str(path) for name, path in paths.items()},
        "missing_artifacts": missing,
        "families": {},
        "families_pass": [],
        "families_warn": [],
        "families_fail": [],
        "total_verified_sft_rows": 0,
        "total_train_rows": 0,
        "total_eval_rows": 0,
        "total_probe_rows": 0,
        "total_dpo_train_pairs": 0,
        "total_dpo_eval_pairs": 0,
        "synthetic_verification_pass_rate": 0.0,
        "format_error_count": 0,
        "verification_fail_count": 0,
        "ambiguity_accepted": 0,
        "abstain_placeholders_accepted": 0,
        "duplicate_prompt_count": 0,
        "split_leakage_count": 0,
        "teacher_ready_for_lora": False,
        "training_authorized": False,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_write(out: Path | None, report: dict[str, Any]) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _audit_failure_count(audit: dict[str, Any]) -> int:
    return len(audit.get("failures", [])) + sum(int(v) for v in audit.get("counts", {}).values())


def _dedup_count(audit: dict[str, Any]) -> int:
    return sum(int(v) for k, v in audit.get("counts", {}).items() if "duplicate" in k)


def _split_leakage_count(audit: dict[str, Any]) -> int:
    return sum(int(v) for k, v in audit.get("counts", {}).items() if "overlap" in k or "leakage" in k)


def _overall_pass_rate(families: dict[str, Any]) -> float:
    generated = sum(int(v.get("synthetic_examples_generated", 0)) for k, v in families.items() if k in KNOWN_TRAINABLE_FAMILIES)
    verified = sum(int(v.get("synthetic_examples_verified", 0)) for k, v in families.items() if k in KNOWN_TRAINABLE_FAMILIES)
    return verified / generated if generated else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

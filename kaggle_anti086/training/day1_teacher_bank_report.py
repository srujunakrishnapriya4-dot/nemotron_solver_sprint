from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_teacher_families import available_teachers
from kaggle_anti086.training.day1_teacher_verifier import validate_target_text


P0_FAMILIES = ("symbol_mapping", "bit_manipulation", "char_cipher", "unit_conversion", "numeric_formula_safe")
P1_FAMILIES = ("word_cipher", "custom_numeral", "permutation_sorting")
P2_FAMILIES = ("gravity_numeric", "equation_operator", "sequence_pattern")
PROTOCOL_ONLY_OPTIONAL = ("model_verified_fallback", "composed_hidden_style")

PRIORITY = (
    {family: "P0" for family in P0_FAMILIES}
    | {family: "P1" for family in P1_FAMILIES}
    | {family: "P2" for family in P2_FAMILIES}
)
SYNTHETIC_RISK = {
    "symbol_mapping": "LOW",
    "bit_manipulation": "LOW",
    "char_cipher": "LOW",
    "unit_conversion": "LOW",
    "numeric_formula_safe": "MEDIUM",
    "word_cipher": "MEDIUM",
    "custom_numeral": "MEDIUM",
    "permutation_sorting": "MEDIUM",
    "gravity_numeric": "HIGH",
    "equation_operator": "HIGH",
    "sequence_pattern": "HIGH",
}


def build_teacher_bank_report(
    root: Path,
    *,
    per_family: int = 100,
    seed: int = 123,
    out: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    phase1_path = root / "artifacts/sprint11/day1_family_recovery_audit.json"
    phase1 = _read_json(phase1_path)
    phase1_present = phase1 is not None
    teachers = available_teachers()
    rng = random.Random(seed)
    difficulties = ("easy", "medium", "hard")
    styles = ("direct", "examples_query", "table", "story", "minimal")

    family_reports: dict[str, dict[str, Any]] = {}
    total_generated = 0
    total_verified = 0
    total_hard_negatives = 0
    format_errors = 0
    verification_fails = 0
    ambiguity_accepts = 0

    for family in (*P0_FAMILIES, *P1_FAMILIES, *P2_FAMILIES):
        teacher = teachers.get(family)
        notes: list[str] = []
        generated = 0
        verified = 0
        hard_negatives = 0
        fam_format_errors = 0
        fam_verification_fails = 0
        fam_ambiguity_accepts = 0

        if teacher is None:
            family_reports[family] = _family_report(
                status="NOT_IMPLEMENTED",
                priority=PRIORITY[family],
                generated=0,
                verified=0,
                verification_failures=0,
                format_errors=0,
                ambiguity_accepted=0,
                hard_negatives=0,
                risk=SYNTHETIC_RISK[family],
                trainable=False,
                notes=["teacher_not_implemented"],
            )
            continue

        for index in range(per_family):
            difficulty = difficulties[index % len(difficulties)]
            style = styles[index % len(styles)]
            example = teacher.generate_synthetic(rng, difficulty, style)
            generated += 1
            ok, reason = validate_target_text(example.target_text, example.answer)
            if not ok:
                fam_format_errors += 1
                notes.append(f"format_error:{reason}")
            result = teacher.verify_example(example)
            if result.verification_status != "PASS" or result.answer != example.answer:
                fam_verification_fails += 1
                notes.append(f"verification_failed:{result.rejection_reason}")
            else:
                verified += 1
            if example.ambiguity_count != 0:
                fam_ambiguity_accepts += 1
            negative = teacher.generate_hard_negative(example, rng)
            if negative.rejected == negative.chosen or negative.verifier_status == "PASS":
                fam_verification_fails += 1
                notes.append("hard_negative_not_rejected")
            hard_negatives += 1

        status = "PASS"
        if fam_format_errors or fam_verification_fails or fam_ambiguity_accepts:
            status = "FAIL"
        family_reports[family] = _family_report(
            status=status,
            priority=PRIORITY[family],
            generated=generated,
            verified=verified,
            verification_failures=fam_verification_fails,
            format_errors=fam_format_errors,
            ambiguity_accepted=fam_ambiguity_accepts,
            hard_negatives=hard_negatives,
            risk=SYNTHETIC_RISK[family],
            trainable=status == "PASS",
            notes=sorted(set(notes)),
        )
        total_generated += generated
        total_verified += verified
        total_hard_negatives += hard_negatives
        format_errors += fam_format_errors
        verification_fails += fam_verification_fails
        ambiguity_accepts += fam_ambiguity_accepts

    for family in PROTOCOL_ONLY_OPTIONAL:
        family_reports[family] = _family_report(
            status="NOT_IMPLEMENTED",
            priority="PROTOCOL",
            generated=0,
            verified=0,
            verification_failures=0,
            format_errors=0,
            ambiguity_accepted=0,
            hard_negatives=0,
            risk="HIGH",
            trainable=False,
            notes=["optional_protocol_deferred", "not_in_teacher_registry", "no_local_recovery_claim"],
        )

    families_pass = sorted(name for name, item in family_reports.items() if item["teacher_status"] == "PASS")
    families_warn = sorted(name for name, item in family_reports.items() if item["teacher_status"] == "WARN")
    families_fail = sorted(name for name, item in family_reports.items() if item["teacher_status"] == "FAIL")
    families_not_implemented = sorted(
        name for name, item in family_reports.items() if item["teacher_status"] == "NOT_IMPLEMENTED"
    )

    implemented_core = (*P0_FAMILIES, *P1_FAMILIES, *P2_FAMILIES)
    p0_failures = [family for family in P0_FAMILIES if family_reports[family]["teacher_status"] != "PASS"]
    implemented_failures = [family for family in implemented_core if family_reports[family]["teacher_status"] != "PASS"]
    gates_pass = (
        not p0_failures
        and not implemented_failures
        and len(families_pass) >= 6
        and format_errors == 0
        and verification_fails == 0
        and ambiguity_accepts == 0
        and total_verified >= 600
    )
    if implemented_failures or format_errors or verification_fails or ambiguity_accepts:
        status = "FAIL"
    elif families_not_implemented:
        status = "WARN"
    else:
        status = "PASS"

    report = {
        "schema_version": 1,
        "created_by": "DAY1_TEACHER_BANK_REPORT",
        "status": status,
        "phase1_report_present": phase1_present,
        "phase1_route_status": None
        if not phase1_present
        else phase1.get("phase2_interpretation", {}).get("local_recovery_route_status"),
        "families": family_reports,
        "families_pass": families_pass,
        "families_warn": families_warn,
        "families_fail": families_fail,
        "families_not_implemented": families_not_implemented,
        "total_synthetic_examples_generated": total_generated,
        "total_synthetic_examples_verified": total_verified,
        "total_hard_negatives_generated": total_hard_negatives,
        "format_error_count": format_errors,
        "verification_fail_count": verification_fails,
        "ambiguity_accept_count": ambiguity_accepts,
        "teacher_bank_ready_for_verified_data_factory": gates_pass,
        "safe_to_train_lora": False,
        "safe_to_package": False,
        "safe_to_submit": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "model_results_faked": False,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1 Phase2 deterministic teacher bank report.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("artifacts/sprint11/day1_teacher_bank_report.json"))
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    report = build_teacher_bank_report(args.root, per_family=args.per_family, seed=args.seed, out=args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 1


def _family_report(
    *,
    status: str,
    priority: str,
    generated: int,
    verified: int,
    verification_failures: int,
    format_errors: int,
    ambiguity_accepted: int,
    hard_negatives: int,
    risk: str,
    trainable: bool,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "teacher_status": status,
        "priority": priority,
        "synthetic_examples_generated": generated,
        "synthetic_examples_verified": verified,
        "verification_failures": verification_failures,
        "format_errors": format_errors,
        "ambiguity_accepted": ambiguity_accepted,
        "hard_negatives_generated": hard_negatives,
        "synthetic_teacher_risk": risk,
        "local_router_integration_allowed": False,
        "trainable_candidate": trainable,
        "notes": notes,
    }


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


if __name__ == "__main__":
    raise SystemExit(main())

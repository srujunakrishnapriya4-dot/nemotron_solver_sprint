from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_verified_data_schema import write_jsonl
from kaggle_anti086.training.day1x_composed_teachers import (
    COMPOSED_TEACHER_REGISTRY,
    REQUIRED_COMPOSED_FAMILIES,
    generate_composed_rows,
    validate_composed_row,
)
from kaggle_anti086.training.day1x_prompt_augmentation import (
    augment_rows,
    supported_public_styles,
    validate_augmented_row,
)


CREATED_BY = "DAY1X_COMPOSED_TEACHERS_AND_PUBLIC_STYLE_PROMPTS"
DECISION_CREATED_BY = "DAY1X_PROMPT_STAGE_DECISION_REPORT"


def build_composed_teacher_report(rows: list[dict[str, Any]], *, per_family: int) -> dict[str, Any]:
    per_family_report: dict[str, dict[str, Any]] = {}
    total_verified_pass = 0
    total_verified_fail = 0
    format_error_count = 0
    ambiguity_accepted = 0
    abstain_accepted = 0
    prompt_counter: Counter[str] = Counter()

    for row in rows:
        prompt_counter[_prompt_key(str(row.get("prompt", "")))] += 1
    duplicate_prompt_count = sum(count - 1 for count in prompt_counter.values() if count > 1)

    for family in REQUIRED_COMPOSED_FAMILIES:
        family_rows = [row for row in rows if row.get("family") == family]
        prompt_styles = Counter(str(row.get("prompt_style")) for row in family_rows)
        verified_pass = 0
        verified_fail = 0
        fam_format_errors = 0
        fam_ambiguity = 0
        fam_abstain = 0
        for row in family_rows:
            ok, errors = validate_composed_row(row)
            if ok:
                verified_pass += 1
            else:
                verified_fail += 1
            if any("target" in error or "box" in error for error in errors):
                fam_format_errors += 1
            if row.get("ambiguity_count") != 0 or any("ambiguity" in error for error in errors):
                fam_ambiguity += 1
            joined = "\n".join(str(row.get(key, "")) for key in ("prompt", "answer", "trace", "target_text"))
            if "ABSTAIN" in joined.upper():
                fam_abstain += 1
        teacher_status = "PASS"
        if len(family_rows) < per_family or verified_fail or fam_format_errors or fam_ambiguity or fam_abstain:
            teacher_status = "FAIL"
        if len(COMPOSED_TEACHER_REGISTRY.get(family, ()).supported_prompt_styles()) < 4 if family in COMPOSED_TEACHER_REGISTRY else True:
            teacher_status = "FAIL"
        per_family_report[family] = {
            "generated": len(family_rows),
            "verified_pass": verified_pass,
            "verified_fail": verified_fail,
            "format_errors": fam_format_errors,
            "ambiguity_accepted": fam_ambiguity,
            "abstain_accepted": fam_abstain,
            "prompt_styles": dict(sorted(prompt_styles.items())),
            "hard_negative_generated": 0,
            "teacher_status": teacher_status,
        }
        total_verified_pass += verified_pass
        total_verified_fail += verified_fail
        format_error_count += fam_format_errors
        ambiguity_accepted += fam_ambiguity
        abstain_accepted += fam_abstain

    families_pass = sorted(family for family, item in per_family_report.items() if item["teacher_status"] == "PASS")
    families_fail = sorted(family for family, item in per_family_report.items() if item["teacher_status"] == "FAIL")
    families_warn: list[str] = []
    supported_count = len([family for family in REQUIRED_COMPOSED_FAMILIES if family in COMPOSED_TEACHER_REGISTRY])
    hard_fail = (
        format_error_count
        or ambiguity_accepted
        or abstain_accepted
        or total_verified_fail
        or duplicate_prompt_count
        or supported_count < 5
        or len(families_pass) < len(REQUIRED_COMPOSED_FAMILIES)
    )
    if not hard_fail and supported_count >= 9:
        status = "PASS"
    elif supported_count >= 5 and not (format_error_count or ambiguity_accepted or abstain_accepted):
        status = "WARN"
        families_warn = sorted(set(REQUIRED_COMPOSED_FAMILIES) - set(families_pass))
    else:
        status = "FAIL"
    return {
        "schema_version": 1,
        "created_by": CREATED_BY,
        "status": status,
        "per_family": per_family_report,
        "families_pass": families_pass,
        "families_warn": families_warn,
        "families_fail": families_fail,
        "total_generated": len(rows),
        "total_verified_pass": total_verified_pass,
        "total_verified_fail": total_verified_fail,
        "format_error_count": format_error_count,
        "ambiguity_accepted": ambiguity_accepted,
        "abstain_accepted": abstain_accepted,
        "duplicate_prompt_count": duplicate_prompt_count,
        "public_styles_supported": supported_public_styles(),
        "composed_families_supported": sorted(COMPOSED_TEACHER_REGISTRY),
        "ready_for_day1x_100k_factory": status == "PASS",
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def build_public_style_prompt_report(original_rows: list[dict[str, Any]], augmented_rows: list[dict[str, Any]]) -> dict[str, Any]:
    style_counts = Counter(str(row.get("augmentation_style")) for row in augmented_rows)
    validation_errors: Counter[str] = Counter()
    pass_rows = 0
    by_parent: defaultdict[str, int] = defaultdict(int)
    originals = {str(row.get("id")): row for row in original_rows}
    for row in augmented_rows:
        parent_id = str(row.get("parent_id"))
        by_parent[parent_id] += 1
        original = originals.get(parent_id)
        if original is None:
            validation_errors["missing_parent"] += 1
            continue
        ok, errors = validate_augmented_row(original, row)
        if ok:
            pass_rows += 1
        else:
            validation_errors.update(errors)
    status = "PASS" if len(supported_public_styles()) >= 8 and pass_rows == len(augmented_rows) and not validation_errors else "FAIL"
    return {
        "schema_version": 1,
        "created_by": CREATED_BY,
        "status": status,
        "original_rows": len(original_rows),
        "augmented_rows": len(augmented_rows),
        "augmented_verified_pass": pass_rows,
        "augmented_verified_fail": len(augmented_rows) - pass_rows,
        "public_styles_supported": supported_public_styles(),
        "style_counts": dict(sorted(style_counts.items())),
        "validation_errors": dict(sorted(validation_errors.items())),
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def build_decision_report(composed_report: dict[str, Any], public_report: dict[str, Any]) -> dict[str, Any]:
    blocked: list[str] = []
    if composed_report.get("status") != "PASS":
        blocked.append("composed_teacher_report_not_pass")
    if public_report.get("status") != "PASS":
        blocked.append("public_style_prompt_report_not_pass")
    for key in ("format_error_count", "ambiguity_accepted", "abstain_accepted", "duplicate_prompt_count"):
        if int(composed_report.get(key, 0)) != 0:
            blocked.append(f"{key}_nonzero")
    ready = not blocked
    return {
        "schema_version": 1,
        "created_by": DECISION_CREATED_BY,
        "status": "PASS" if ready else "FAIL",
        "ready_for_next_phase": ready,
        "ready_for_day1x_100k_factory": ready,
        "next_phase": "DAY1X_HOLDOUTS_AND_EXPANDED_DATA_FACTORY_100K",
        "blocked_reasons": blocked,
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def generate_reports(out_dir: Path, *, per_family: int = 200, seed: int = 123) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    composed_rows = generate_composed_rows(per_family=per_family, seed=seed)
    augmented_rows = augment_rows(
        composed_rows,
        [
            "public_style_minimal",
            "public_style_verbose",
            "public_style_table",
            "public_style_examples_then_query",
            "public_style_noisy_context",
            "public_style_symbol_heavy",
            "public_style_mixed_notation",
            "public_style_multiline",
        ],
        seed=seed + 500000,
        max_per_row=1,
    )
    composed_report = build_composed_teacher_report(composed_rows, per_family=per_family)
    public_report = build_public_style_prompt_report(composed_rows, augmented_rows)
    decision_report = build_decision_report(composed_report, public_report)
    files = {
        "composed_report": out_dir / "day1x_composed_teacher_report.json",
        "public_report": out_dir / "day1x_public_style_prompt_report.json",
        "composed_samples": out_dir / "day1x_composed_teacher_samples.jsonl",
        "augmented_samples": out_dir / "day1x_public_style_augmented_samples.jsonl",
        "decision_report": out_dir / "day1x_decision_report_prompt_stage.json",
    }
    _write_json(files["composed_report"], composed_report)
    _write_json(files["public_report"], public_report)
    write_jsonl(files["composed_samples"], composed_rows)
    write_jsonl(files["augmented_samples"], augmented_rows)
    _write_json(files["decision_report"], decision_report)
    return {
        "composed_report": composed_report,
        "public_report": public_report,
        "decision_report": decision_report,
        "output_files": {key: str(value) for key, value in files.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1X composed teacher and public-style prompt reports.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--per-family", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    result = generate_reports(args.out_dir, per_family=args.per_family, seed=args.seed)
    print(
        json.dumps(
            {
                "composed_status": result["composed_report"]["status"],
                "public_status": result["public_report"]["status"],
                "decision_status": result["decision_report"]["status"],
                "ready_for_day1x_100k_factory": result["decision_report"]["ready_for_day1x_100k_factory"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["decision_report"]["status"] == "PASS" else 1


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


if __name__ == "__main__":
    raise SystemExit(main())

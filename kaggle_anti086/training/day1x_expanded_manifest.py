from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_verified_data_schema import read_jsonl
from kaggle_anti086.training.day1x_composed_teachers import REQUIRED_COMPOSED_FAMILIES, validate_composed_row
from kaggle_anti086.training.day1x_expanded_data_factory import MIXTURE_TARGETS, ORIGINAL_FAMILIES, STAGE_SIZES
from kaggle_anti086.training.day1_teacher_trainability_gate import detect_abstain_placeholder, verify_trainable_row
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers, extract_boxed_answer


CREATED_BY = "DAY1X_EXPANDED_DATA_FACTORY_100K"
DECISION_CREATED_BY = "DAY1X_100K_STAGE_DECISION_REPORT"


def build_expanded_manifest(out_dir: str | Path, *, expected_stage_sizes: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    expected = expected_stage_sizes or STAGE_SIZES
    stage_reports: dict[str, dict[str, Any]] = {}
    all_accepted: list[dict[str, Any]] = []
    available_stages: list[str] = []
    blocked: list[str] = []
    for stage, sizes in expected.items():
        files = _stage_files(stage, out)
        if not all(path.exists() for path in files.values()):
            stage_reports[stage] = {"status": "MISSING", "counts": {"train_rows": 0, "eval_rows": 0, "probe_rows": 0}, "blocked_reasons": ["missing_stage_files"]}
            continue
        train, eval_rows, probe = read_jsonl(files["train"]), read_jsonl(files["eval"]), read_jsonl(files["probe"])
        rows = [*train, *eval_rows, *probe]
        all_accepted.extend(rows)
        counters = _safety_counters({"train": train, "eval": eval_rows, "probe": probe})
        counts = {"train_rows": len(train), "eval_rows": len(eval_rows), "probe_rows": len(probe)}
        reasons = []
        if counts["train_rows"] != sizes["train"]:
            reasons.append("train_count_mismatch")
        if counts["eval_rows"] != sizes["eval"]:
            reasons.append("eval_count_mismatch")
        if counts["probe_rows"] != sizes["probe"]:
            reasons.append("probe_count_mismatch")
        for key in ("format_error_count", "verification_fail_count", "ambiguity_accepted", "abstain_accepted", "duplicate_prompt_count", "split_leakage_count"):
            if counters[key] != 0:
                reasons.append(f"{key}_nonzero")
        mixture = _mixture_report(rows)
        if not mixture["within_tolerance"]:
            reasons.append("mixture_out_of_tolerance")
        stage_reports[stage] = {"status": "PASS" if not reasons else "FAIL", "counts": counts, "safety": counters, "mixture_report": mixture, "blocked_reasons": reasons}
        if not reasons:
            available_stages.append(stage)

    holdouts = _holdout_counts(out)
    if any(count <= 0 for count in holdouts.values()):
        blocked.append("missing_or_empty_holdout")
    selected = _select_stage(available_stages)
    selected_rows = _rows_for_stage(out, selected) if selected != "basic_day1" else []
    selected_counters = _safety_counters({"selected": selected_rows}) if selected_rows else _aggregate_stage_counters(stage_reports)
    rejected_count = _count_jsonl(out / "day1x_rejected_rows.jsonl") if (out / "day1x_rejected_rows.jsonl").exists() else 0
    rows_by_family = Counter(str(row.get("family")) for row in selected_rows)
    rows_by_prompt_style = Counter(str(row.get("prompt_style")) for row in selected_rows)
    rows_by_source = Counter(str(row.get("metadata", {}).get("mixture_source", row.get("source", "unknown"))) for row in selected_rows)
    rows_by_difficulty = Counter(str(row.get("difficulty")) for row in selected_rows)
    original_present = sorted(set(rows_by_family) & set(ORIGINAL_FAMILIES))
    composed_present = sorted(set(rows_by_family) & set(REQUIRED_COMPOSED_FAMILIES))

    if selected == "basic_day1":
        status = "FAIL"
        blocked.append("no_clean_expanded_stage")
    elif selected == "100k" and not blocked and len(original_present) == len(ORIGINAL_FAMILIES) and len(composed_present) == len(REQUIRED_COMPOSED_FAMILIES):
        status = "PASS"
    elif selected in {"50k", "25k", "100k"} and not any(selected_counters.values()):
        status = "WARN"
    else:
        status = "FAIL"
    if selected != "basic_day1":
        if len(original_present) < len(ORIGINAL_FAMILIES):
            blocked.append("missing_original_family")
        if len(composed_present) < min(5, len(REQUIRED_COMPOSED_FAMILIES)):
            blocked.append("fewer_than_5_composed_families")
    blocked = sorted(set(blocked))
    if blocked and status == "PASS":
        status = "FAIL"
    return {
        "schema_version": 1,
        "created_by": CREATED_BY,
        "status": status,
        "available_stages": available_stages,
        "selected_stage_for_day2": selected,
        "counts": {
            **{stage: report["counts"] for stage, report in stage_reports.items()},
            "holdouts": holdouts,
        },
        "stage_reports": stage_reports,
        "families": {
            "original_families_present": original_present,
            "composed_families_present": composed_present,
        },
        "rows_by_family": dict(sorted(rows_by_family.items())),
        "rows_by_prompt_style": dict(sorted(rows_by_prompt_style.items())),
        "rows_by_source": dict(sorted(rows_by_source.items())),
        "rows_by_difficulty": dict(sorted(rows_by_difficulty.items())),
        **selected_counters,
        "rejected_row_count": rejected_count,
        "mixture_report": _mixture_report(selected_rows) if selected_rows else {f"{key}_fraction": 0.0 for key in MIXTURE_TARGETS},
        "ready_for_day1x_dpo_and_variants": status in {"PASS", "WARN"} and selected != "basic_day1" and not any(selected_counters.values()),
        "recommended_for_day2_training": status in {"PASS", "WARN"} and selected != "basic_day1",
        "fallback_to_basic_day1": selected == "basic_day1",
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "blocked_reasons": blocked,
    }


def write_expanded_manifest(out_path: str | Path) -> dict[str, Any]:
    path = Path(out_path)
    manifest = build_expanded_manifest(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_day1x_100k_decision_report(manifest: dict[str, Any]) -> dict[str, Any]:
    recommended = manifest.get("recommended_for_day2_training") is True and manifest.get("selected_stage_for_day2") != "basic_day1"
    return {
        "schema_version": 1,
        "created_by": DECISION_CREATED_BY,
        "status": manifest.get("status", "FAIL"),
        "selected_stage_for_day2": manifest.get("selected_stage_for_day2", "basic_day1"),
        "recommended_for_next_phase": recommended,
        "next_phase": "DAY1X_DPO_HARD_NEGATIVE_EXPANSION_AND_VARIANTS",
        "blocked_reasons": list(manifest.get("blocked_reasons", [])),
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1X expanded data manifest and decision report.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_expanded_manifest(args.out_dir)
    decision = build_day1x_100k_decision_report(manifest)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.decision.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "selected_stage_for_day2": manifest["selected_stage_for_day2"]}, sort_keys=True))
    return 0 if manifest["status"] in {"PASS", "WARN"} else 1


def _safety_counters(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counters = _zero_counters()
    all_prompts: Counter[str] = Counter()
    for _split, rows in splits.items():
        for row in rows:
            prompt_key = _prompt_key(str(row.get("prompt", "")))
            all_prompts[prompt_key] += 1
            if row.get("verification_status") != "PASS":
                counters["verification_fail_count"] += 1
            if row.get("ambiguity_count") != 0:
                counters["ambiguity_accepted"] += 1
            if detect_abstain_placeholder(row):
                counters["abstain_accepted"] += 1
            if count_boxed_answers(str(row.get("target_text", ""))) != 1 or extract_boxed_answer(str(row.get("target_text", ""))) != str(row.get("answer")):
                counters["format_error_count"] += 1
            check = verify_trainable_row(row)
            if not check.trainable:
                counters["verification_fail_count"] += 1
            if str(row.get("family", "")).startswith("composed_") and not validate_composed_row(row)[0]:
                counters["verification_fail_count"] += 1
    counters["duplicate_prompt_count"] = sum(count - 1 for count in all_prompts.values() if count > 1)
    counters["split_leakage_count"] = counters["duplicate_prompt_count"]
    return counters


def _zero_counters() -> dict[str, int]:
    return {
        "format_error_count": 0,
        "verification_fail_count": 0,
        "ambiguity_accepted": 0,
        "abstain_accepted": 0,
        "duplicate_prompt_count": 0,
        "split_leakage_count": 0,
    }


def _aggregate_stage_counters(stage_reports: dict[str, dict[str, Any]]) -> dict[str, int]:
    counters = _zero_counters()
    for report in stage_reports.values():
        safety = report.get("safety") or {}
        for key in counters:
            counters[key] += int(safety.get(key, 0))
    return counters


def _holdout_counts(out: Path) -> dict[str, int]:
    return {
        "rule": _count_jsonl(out / "day1x_holdout_rule.jsonl"),
        "prompt_style": _count_jsonl(out / "day1x_holdout_prompt_style.jsonl"),
        "composed": _count_jsonl(out / "day1x_holdout_composed.jsonl"),
        "format_traps": _count_jsonl(out / "day1x_holdout_format_traps.jsonl"),
        "hard_p2": _count_jsonl(out / "day1x_holdout_hard_p2.jsonl"),
    }


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _select_stage(available: list[str]) -> str:
    for stage in ("100k", "50k", "25k"):
        if stage in available:
            return stage
    return sorted(available)[-1] if available else "basic_day1"


def _rows_for_stage(out: Path, stage: str) -> list[dict[str, Any]]:
    if stage == "basic_day1":
        return []
    files = _stage_files(stage, out)
    if not all(path.exists() for path in files.values()):
        return []
    return [*read_jsonl(files["train"]), *read_jsonl(files["eval"]), *read_jsonl(files["probe"])]


def _stage_files(stage: str, out: Path) -> dict[str, Path]:
    if stage == "25k":
        return {"train": out / "day1x_verified_sft_train_25k.jsonl", "eval": out / "day1x_verified_sft_eval_2k.jsonl", "probe": out / "day1x_verified_probe_2k.jsonl"}
    if stage == "50k":
        return {"train": out / "day1x_verified_sft_train_50k.jsonl", "eval": out / "day1x_verified_sft_eval_5k.jsonl", "probe": out / "day1x_verified_probe_5k.jsonl"}
    if stage == "100k":
        return {"train": out / "day1x_verified_sft_train_100k.jsonl", "eval": out / "day1x_verified_sft_eval_10k.jsonl", "probe": out / "day1x_verified_probe_10k.jsonl"}
    return {"train": out / f"day1x_verified_sft_train_{stage}.jsonl", "eval": out / f"day1x_verified_sft_eval_{stage}.jsonl", "probe": out / f"day1x_verified_probe_{stage}.jsonl"}


def _mixture_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    counts = Counter(str(row.get("metadata", {}).get("mixture_source", row.get("source", "unknown"))) for row in rows)
    grouped = {
        "original": counts.get("original_day1", 0),
        "composed": counts.get("composed_day1x", 0),
        "public_style": counts.get("public_style_day1x", 0),
        "format_stress": counts.get("format_stress", 0),
    }
    report = {f"{key}_fraction": grouped[key] / total for key in MIXTURE_TARGETS}
    report["counts"] = grouped
    report["within_tolerance"] = all(abs(report[f"{key}_fraction"] - target) <= 0.03 for key, target in MIXTURE_TARGETS.items())
    return report


def _prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


if __name__ == "__main__":
    raise SystemExit(main())

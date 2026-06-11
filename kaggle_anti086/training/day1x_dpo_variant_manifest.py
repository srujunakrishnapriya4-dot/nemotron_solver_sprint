from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


CREATED_BY = "DAY1X_DPO_HARD_NEGATIVE_EXPANSION_AND_VARIANTS"
DECISION_CREATED_BY = "DAY1X_DPO_VARIANT_STAGE_DECISION_REPORT"
NEXT_PHASE = "DAY1X_STRESS_DIVERSITY_LEAKAGE_AND_FINAL_MANIFEST"
HOLDOUT_FILES = {
    "rule": "day1x_holdout_rule.jsonl",
    "prompt_style": "day1x_holdout_prompt_style.jsonl",
    "composed": "day1x_holdout_composed.jsonl",
    "format_traps": "day1x_holdout_format_traps.jsonl",
    "hard_p2": "day1x_holdout_hard_p2.jsonl",
}
VARIANT_NAMES = (
    "direct",
    "short_trace",
    "mixed_curriculum",
    "format_heavy",
    "composed_heavy",
    "public_style_heavy",
    "hard_p2_heavy",
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count_jsonl(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_day1x_100k_inputs(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    blocked: list[str] = []
    manifest_path = out / "day1x_expanded_data_manifest.json"
    decision_path = out / "day1x_decision_report_100k_stage.json"
    manifest: dict[str, Any] = {}
    decision: dict[str, Any] = {}
    if not manifest_path.exists():
        blocked.append("missing_expanded_manifest")
    else:
        manifest = load_json(manifest_path)
    if not decision_path.exists():
        blocked.append("missing_100k_decision_report")
    else:
        decision = load_json(decision_path)
    if manifest and manifest.get("status") != "PASS":
        blocked.append("expanded_manifest_not_pass")
    if decision and decision.get("status") != "PASS":
        blocked.append("decision_report_not_pass")
    selected = manifest.get("selected_stage_for_day2") or decision.get("selected_stage_for_day2")
    if selected != "100k":
        blocked.append("selected_stage_not_100k")
    counts = {
        "train_rows": count_jsonl(out / "day1x_verified_sft_train_100k.jsonl"),
        "eval_rows": count_jsonl(out / "day1x_verified_sft_eval_10k.jsonl"),
        "probe_rows": count_jsonl(out / "day1x_verified_probe_10k.jsonl"),
    }
    expected = {"train_rows": 100000, "eval_rows": 10000, "probe_rows": 10000}
    for key, value in expected.items():
        if counts[key] != value:
            blocked.append(f"{key}_count_mismatch")
    holdouts = {key: count_jsonl(out / file_name) for key, file_name in HOLDOUT_FILES.items()}
    for key, value in holdouts.items():
        if value <= 0:
            blocked.append(f"holdout_{key}_missing_or_empty")
    for field in ("package_authorized", "submission_authorized", "leaderboard_claim"):
        if manifest.get(field) is not False or decision.get(field) is not False:
            blocked.append(f"{field}_not_false")
    return {
        "status": "PASS" if not blocked else "FAIL",
        "selected_stage_for_day2": selected,
        **counts,
        "holdout_counts": holdouts,
        "blocked_reasons": sorted(set(blocked)),
    }


def build_dpo_variant_manifest(out_dir: str | Path, thresholds: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    out = Path(out_dir)
    thresholds = thresholds or {}
    require_precheck = thresholds.get("require_precheck", True)
    precheck = validate_day1x_100k_inputs(out) if require_precheck else {"status": "PASS", "selected_stage_for_day2": "100k", "blocked_reasons": []}
    audit = load_json(out / "day1x_dpo_audit_report.json")
    variants = _load_variant_manifests(out)
    pass_variants = [name for name, manifest in variants.items() if manifest.get("status") == "PASS"]
    safety = _aggregate_safety(audit, variants)
    recommended = [name for name in VARIANT_NAMES if variants.get(name, {}).get("recommended_for_day2_training") is True and variants.get(name, {}).get("status") in {"PASS", "WARN"}]
    blocked: list[str] = list(precheck.get("blocked_reasons", []))
    if audit.get("status") == "FAIL":
        blocked.append("dpo_audit_fail")
    if len(pass_variants) < thresholds.get("min_pass_variants", 5):
        blocked.append("fewer_than_required_pass_variants")
    for required in thresholds.get("required_pass_variants", ("direct", "short_trace", "mixed_curriculum", "format_heavy")):
        if variants.get(required, {}).get("status") != "PASS":
            blocked.append(f"{required}_not_pass")
    for key, value in safety.items():
        if value != 0:
            blocked.append(f"{key}_nonzero")
    if audit.get("package_authorized") is not False or audit.get("submission_authorized") is not False or audit.get("leaderboard_claim") is not False:
        blocked.append("unsafe_dpo_flag_true")
    status = "PASS" if not blocked and audit.get("status") == "PASS" else "FAIL"
    if status == "FAIL" and audit.get("status") == "WARN" and len(pass_variants) >= thresholds.get("warn_min_pass_variants", 4) and variants.get("mixed_curriculum", {}).get("status") == "PASS" and not any(safety.values()):
        status = "WARN"
        blocked = [reason for reason in blocked if reason != "dpo_audit_fail"]
    manifest = {
        "schema_version": 1,
        "created_by": CREATED_BY,
        "status": status,
        "input_stage": "100k",
        "precheck": precheck,
        "dpo": {
            "train_pairs": audit.get("train_pairs", 0),
            "eval_pairs": audit.get("eval_pairs", 0),
            "status": audit.get("status", "FAIL"),
            "negative_type_counts": audit.get("negative_type_counts", {}),
            "rejected_accidentally_trainable_count": audit.get("rejected_accidentally_trainable_count", 0),
            "rejected_accidentally_correct_count": audit.get("rejected_accidentally_correct_count", 0),
        },
        "variants": {name: {"status": data.get("status"), "row_count": data.get("row_count", 0)} for name, data in variants.items()},
        **safety,
        "recommended_training_variants": recommended,
        "ready_for_next_phase": status in {"PASS", "WARN"},
        "next_phase": NEXT_PHASE,
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "blocked_reasons": sorted(set(blocked)),
    }
    decision = build_decision_report(manifest)
    return manifest, decision


def build_decision_report(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_by": DECISION_CREATED_BY,
        "status": manifest.get("status", "FAIL"),
        "ready_for_next_phase": manifest.get("ready_for_next_phase") is True,
        "next_phase": NEXT_PHASE,
        "selected_stage_for_day2": "100k",
        "recommended_training_variants": list(manifest.get("recommended_training_variants", [])),
        "dpo_train_pairs": manifest.get("dpo", {}).get("train_pairs", 0),
        "dpo_eval_pairs": manifest.get("dpo", {}).get("eval_pairs", 0),
        "variant_count": len(manifest.get("variants", {})),
        "blocked_reasons": list(manifest.get("blocked_reasons", [])),
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def write_dpo_variant_manifest(out_dir: str | Path, manifest_path: str | Path, decision_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, decision = build_dpo_variant_manifest(out_dir)
    manifest_path = Path(manifest_path)
    decision_path = Path(decision_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest, decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Day1X DPO and curriculum variant manifests.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/sprint11/day1x_dpo_variant_manifest.json"))
    parser.add_argument("--decision", type=Path, default=Path("artifacts/sprint11/day1x_decision_report_dpo_variant_stage.json"))
    args = parser.parse_args(argv)
    manifest, decision = write_dpo_variant_manifest(args.out_dir, args.manifest, args.decision)
    print(json.dumps({"status": manifest["status"], "ready_for_next_phase": decision["ready_for_next_phase"]}, sort_keys=True))
    return 0 if manifest["status"] in {"PASS", "WARN"} else 1


def _load_variant_manifests(out: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name in VARIANT_NAMES:
        path = out / f"day1x_variant_manifest_{name}.json"
        loaded[name] = load_json(path) if path.exists() else {"status": "MISSING", "row_count": 0}
    return loaded


def _aggregate_safety(audit: dict[str, Any], variants: dict[str, dict[str, Any]]) -> dict[str, int]:
    safety = audit.get("safety", {})
    return {
        "format_error_count": sum(int(item.get("format_error_count", 0)) for item in variants.values()) + int(safety.get("chosen_format_error_count", 0)),
        "verification_fail_count": sum(int(item.get("verification_fail_count", 0)) for item in variants.values()) + int(safety.get("chosen_verification_fail_count", 0)),
        "ambiguity_accepted": sum(int(item.get("ambiguity_accepted", 0)) for item in variants.values()) + int(safety.get("chosen_ambiguity_count", 0)),
        "abstain_accepted": sum(int(item.get("abstain_accepted", 0)) for item in variants.values()) + int(safety.get("chosen_abstain_count", 0)),
        "duplicate_prompt_count": sum(int(item.get("duplicate_prompt_count", 0)) for item in variants.values()),
        "duplicate_id_count": sum(int(item.get("duplicate_id_count", 0)) for item in variants.values()),
        "split_leakage_count": int(audit.get("train_eval_prompt_overlap_count", 0)),
    }


if __name__ == "__main__":
    raise SystemExit(main())

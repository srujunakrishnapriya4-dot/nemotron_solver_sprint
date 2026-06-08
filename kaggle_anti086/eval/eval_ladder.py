from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


REQUIRED_DATASETS = ("core_eval", "family_eval", "rule_holdout", "anti_leak")
REQUIRED_MODES = ("base_only", "solver_only", "adapter_only", "combined")
SCORE_FIELDS = ("exact_match", "overall_accuracy", "answerable_exact_match", "behavior_accuracy", "overall_score")
LEAKAGE_FIELDS = ("leakage_score", "duplicate_prompt_rate", "contamination_rate", "near_duplicate_rate")
PRIORITY_FAMILIES = {
    "numeric_formula",
    "gravity_numeric",
    "unit_conversion",
    "roman_numeral",
    "bit_manipulation",
    "symbol_mapping",
    "char_cipher",
    "word_cipher",
    "format_only",
}


def build_eval_ladder(manifest_path: str | Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("eval_ladder_manifest_schema_version_must_be_1")
    warnings: list[str] = []
    failures: list[str] = []
    datasets: dict[str, Any] = {}
    by_family: dict[str, Any] = {}
    for dataset in REQUIRED_DATASETS:
        if dataset not in manifest.get("datasets", {}):
            raise ValueError(f"eval_ladder_manifest_missing_dataset:{dataset}")
        dataset_modes = manifest["datasets"][dataset]
        modes: dict[str, Any] = {}
        primary_values: dict[str, float] = {}
        for mode in REQUIRED_MODES:
            if mode not in dataset_modes:
                raise ValueError(f"eval_ladder_manifest_missing_mode:{dataset}.{mode}")
            entry = _normalize_manifest_entry(dataset_modes[mode], dataset, mode, warnings)
            report = _load_report(entry)
            modes[mode] = {
                "path": entry["path"],
                "report_status": report.get("status"),
                "score": _extract_score_or_fail(report, dataset, mode),
                "format_error_rate": _rate(report, ("format_error_rate", "invalid_format_rate")),
                "invalid_answer_rate": _rate(report, ("invalid_answer_rate", "invalid_format_rate")),
                "leakage_score": _explicit_leakage_value(report),
                "by_family": _extract_by_family(report),
                "warnings": list(report.get("warnings", [])),
                "failures": list(report.get("failures", [])),
            }
            primary_values[mode] = modes[mode]["score"]
            if dataset == "core_eval" and mode == "combined":
                by_family = modes[mode]["by_family"]
        datasets[dataset] = {"modes": modes, "primary_values": primary_values}
    leakage_score = _extract_leakage(datasets["anti_leak"]["modes"]["combined"], warnings)
    format_error_rate = max(
        datasets["core_eval"]["modes"]["combined"]["format_error_rate"],
        datasets["core_eval"]["modes"]["combined"]["invalid_answer_rate"],
    )
    decision = _build_decision(datasets, leakage_score, format_error_rate)
    failures.extend(decision.get("failures", []))
    status = "FAIL" if failures else ("WARN" if warnings or decision["decision"] == "STOP_ADAPTER_SCALING" else "PASS")
    return {
        "schema_version": 1,
        "created_by": "DAY2_EVAL_LADDER",
        "status": status,
        "datasets": datasets,
        "by_family": by_family,
        "leakage_score": leakage_score,
        "format_error_rate": format_error_rate,
        "decision": {key: value for key, value in decision.items() if key != "failures"},
        "model_results_faked": False,
        "warnings": warnings,
        "failures": failures,
    }


def write_eval_ladder(report: dict[str, Any], out_path: str | Path) -> None:
    write_json_checked(out_path, report, field_name="day2_eval_ladder_report")


def _normalize_manifest_entry(entry: Any, dataset: str, mode: str, warnings: list[str]) -> dict[str, Any]:
    if isinstance(entry, str):
        warnings.append(f"manifest_entry_missing_sha256:{dataset}.{mode}")
        warnings.append(f"manifest_entry_missing_size_bytes:{dataset}.{mode}")
        return {"path": entry}
    if not isinstance(entry, dict) or not entry.get("path"):
        raise ValueError(f"eval_ladder_manifest_invalid_entry:{dataset}.{mode}")
    path = Path(entry["path"])
    if not path.exists():
        raise FileNotFoundError(str(path))
    if "sha256" in entry:
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"manifest_entry_sha256_mismatch:{dataset}.{mode}")
    else:
        warnings.append(f"manifest_entry_missing_sha256:{dataset}.{mode}")
    if "size_bytes" in entry:
        actual_size = path.stat().st_size
        if int(entry["size_bytes"]) != actual_size:
            raise ValueError(f"manifest_entry_size_bytes_mismatch:{dataset}.{mode}")
    else:
        warnings.append(f"manifest_entry_missing_size_bytes:{dataset}.{mode}")
    return {"path": str(path), "sha256": entry.get("sha256"), "size_bytes": entry.get("size_bytes")}


def _load_report(entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(entry["path"])
    if not path.exists():
        raise FileNotFoundError(str(path))
    return read_json(path)


def _extract_score_or_fail(report: dict[str, Any], dataset: str, mode: str) -> float:
    for field in SCORE_FIELDS:
        value = report.get(field)
        if value is not None:
            return float(value)
    raise ValueError(f"eval_report_missing_explicit_score:{dataset}.{mode}")


def _extract_leakage(anti_leak_combined: dict[str, Any], warnings: list[str]) -> float:
    report_like = anti_leak_combined
    for field in LEAKAGE_FIELDS:
        value = report_like.get(field)
        if value is not None:
            return float(value)
    warnings.append("anti_leak_no_explicit_leakage_metric")
    return 0.0


def _explicit_leakage_value(report: dict[str, Any]) -> float | None:
    for field in LEAKAGE_FIELDS:
        value = report.get(field)
        if value is not None:
            return float(value)
    return None


def _build_decision(datasets: dict[str, Any], leakage_score: float, format_error_rate: float) -> dict[str, Any]:
    core = datasets["core_eval"]["primary_values"]
    rule = datasets["rule_holdout"]["primary_values"]
    combined_core = core["combined"]
    adapter_core = core["adapter_only"]
    base_core = core["base_only"]
    solver_core = core["solver_only"]
    reason_codes: list[str] = []
    failures: list[str] = []
    if adapter_core <= base_core:
        reason_codes.append("adapter_not_better_than_base")
    if combined_core <= solver_core:
        reason_codes.append("combined_not_better_than_solver")
    if combined_core <= adapter_core:
        reason_codes.append("combined_not_better_than_adapter")
    if rule["combined"] < rule["solver_only"]:
        reason_codes.append("rule_holdout_combined_worse_than_solver")
    if leakage_score != 0.0:
        reason_codes.append("leakage_score_nonzero")
    family_regressions = _priority_family_regressions(datasets["core_eval"]["modes"]["solver_only"], datasets["core_eval"]["modes"]["combined"])
    if family_regressions:
        reason_codes.append("priority_family_regression")
    if format_error_rate > 0.05:
        reason_codes.append("format_error_rate_too_high")
    train = not reason_codes
    return {
        "train_v2a_150": train,
        "decision": "train_v2a_150" if train else "STOP_ADAPTER_SCALING",
        "reason_codes": reason_codes,
        "base_exact_match": base_core,
        "solver_exact_match": solver_core,
        "adapter_exact_match": adapter_core,
        "core_exact_match": combined_core,
        "rule_holdout_accuracy": rule["combined"],
        "leakage_score": leakage_score,
        "priority_family_regressions": family_regressions,
        "failures": failures,
    }


def _priority_family_regressions(solver_mode: dict[str, Any], combined_mode: dict[str, Any]) -> list[str]:
    regressions = []
    solver_families = solver_mode.get("by_family", {})
    combined_families = combined_mode.get("by_family", {})
    for family in sorted(PRIORITY_FAMILIES & set(solver_families) & set(combined_families)):
        if float(combined_families[family]) < float(solver_families[family]):
            regressions.append(family)
    return regressions


def _extract_by_family(report: dict[str, Any]) -> dict[str, float]:
    source = report.get("by_family_accuracy", report.get("by_family", {}))
    if not isinstance(source, dict):
        return {}
    output: dict[str, float] = {}
    for family, value in source.items():
        if isinstance(value, dict):
            score = None
            for field in SCORE_FIELDS + ("accuracy",):
                if value.get(field) is not None:
                    score = value[field]
                    break
        else:
            score = value
        if score is not None:
            output[str(family)] = float(score)
    return output


def _rate(report: dict[str, Any], fields: tuple[str, ...]) -> float:
    for field in fields:
        value = report.get(field)
        if value is not None:
            return float(value)
    return 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build strict Day2 eval ladder report from manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_eval_ladder(args.manifest)
    write_eval_ladder(report, args.out)
    print(json.dumps({"status": report["status"], "decision": report["decision"].get("decision"), "out": args.out}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

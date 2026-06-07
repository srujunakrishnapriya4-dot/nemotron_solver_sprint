# kaggle_anti086/eval/eval_ladder.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kaggle_anti086.kaggle_path_safety import safe_write_text

MODE_NAMES = ("base_only", "solver_only", "adapter_only", "combined")
DATASET_NAMES = ("core_eval", "family_eval", "rule_holdout", "anti_leak")
PRIMARY_METRIC = "exact_match"
METRICS = (
    "exact_match",
    "answerable_exact_match",
    "behavior_accuracy",
    "attempt_rate",
    "unsafe_answer_rate",
    "correct_abstain_rate",
)

FAMILY_PRIORITY = (
    "roman_numeral",
    "unit_conversion",
    "numeric_formula",
    "bit_manipulation",
    "symbol_mapping",
    "char_cipher",
    "word_cipher",
    "format_only",
)

def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default

def _int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default

def _is_pass(report: dict[str, Any]) -> bool:
    status = str(report.get("status", "")).upper()
    exec_status = str(report.get("execution_status", "")).upper()
    quality_status = str(report.get("quality_status", "")).upper()
    if status in {"FAIL", "ERROR"}:
        return False
    if exec_status in {"FAIL", "ERROR"}:
        return False
    if quality_status == "FAIL":
        return False
    return True

def _mode_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "execution_status": report.get("execution_status"),
        "quality_status": report.get("quality_status"),
        "row_count": _int(report.get("row_count")),
        "attempted_count": _int(report.get("attempted_count")),
        "exact_match": _float(report.get("exact_match")),
        "answerable_exact_match": _float(report.get("answerable_exact_match")),
        "behavior_accuracy": _float(report.get("behavior_accuracy")),
        "attempt_rate": _float(report.get("attempt_rate")),
        "unsafe_answer_rate": _float(report.get("unsafe_answer_rate")),
        "correct_abstain_rate": _float(report.get("correct_abstain_rate")),
        "unsafe_answer_rate_on_abstain_rows": _float(report.get("unsafe_answer_rate_on_abstain_rows")),
        "correct_abstain_count": _int(report.get("correct_abstain_count")),
        "wrong_abstain_count": _int(report.get("wrong_abstain_count")),
        "wrong_answer_count": _int(report.get("wrong_answer_count")),
        "unsafe_answer_count": _int(report.get("unsafe_answer_count")),
    }

def _family_summary(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = report.get("by_family") or {}
    out: dict[str, dict[str, Any]] = {}
    for family, stats in raw.items():
        if not isinstance(stats, dict):
            continue
        out[str(family)] = {
            "row_count": _int(stats.get("row_count")),
            "exact_match": _float(stats.get("exact_match")),
            "answerable_exact_match": _float(stats.get("answerable_exact_match")),
            "behavior_accuracy": _float(stats.get("behavior_accuracy")),
            "attempt_rate": _float(stats.get("attempt_rate")),
            "unsafe_answer_rate": _float(stats.get("unsafe_answer_rate")),
            "correct_abstain_rate": _float(stats.get("correct_abstain_rate")),
        }
    return out

def _pick_primary(summary: dict[str, Any]) -> float:
    return _float(summary.get(PRIMARY_METRIC))

def _delta(a: float, b: float) -> float:
    return round(a - b, 8)

def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if "reports" not in manifest:
        raise ValueError("manifest missing top-level 'reports' object")
    return manifest

def _load_mode_reports(dataset_manifest: dict[str, Any], dataset_name: str) -> dict[str, dict[str, Any]]:
    if dataset_name not in dataset_manifest:
        raise ValueError(f"manifest missing dataset block: {dataset_name}")
    block = dataset_manifest[dataset_name]
    if not isinstance(block, dict):
        raise ValueError(f"manifest block for {dataset_name} must be a dict")
    reports: dict[str, dict[str, Any]] = {}
    for mode in MODE_NAMES:
        if mode not in block:
            raise ValueError(f"manifest missing {dataset_name}.{mode}")
        path = Path(block[mode])
        reports[mode] = _load_json(path)
    return reports

def _best_mode(mode_summaries: dict[str, dict[str, Any]]) -> str:
    def key(mode: str) -> tuple[float, float, float]:
        s = mode_summaries[mode]
        return (
            _pick_primary(s),
            _float(s.get("behavior_accuracy")),
            -_float(s.get("unsafe_answer_rate")),
        )
    return max(MODE_NAMES, key=key)

def _build_dataset_block(mode_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries = {mode: _mode_summary(report) for mode, report in mode_reports.items()}
    primary = {mode: _pick_primary(summary) for mode, summary in summaries.items()}
    best = _best_mode(summaries)
    pass_all = all(_is_pass(report) for report in mode_reports.values())
    return {
        "status": "PASS" if pass_all else "FAIL",
        "modes": summaries,
        "primary_metric": PRIMARY_METRIC,
        "primary_values": primary,
        "best_mode": best,
        "best_primary_value": primary[best],
    }

def _merge_family_names(*family_maps: dict[str, dict[str, Any]]) -> list[str]:
    names = set(FAMILY_PRIORITY)
    for fm in family_maps:
        names.update(fm.keys())
    return sorted(names)

def _build_family_block(
    core_families: dict[str, dict[str, Any]],
    family_eval_families: dict[str, dict[str, Any]],
    mode_family_reports: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    family_names = _merge_family_names(core_families, family_eval_families)
    out: dict[str, Any] = {}
    for family in family_names:
        per_mode = {}
        for mode in MODE_NAMES:
            fam_map = mode_family_reports.get(mode, {})
            stats = fam_map.get(family, {})
            per_mode[mode] = {
                "row_count": _int(stats.get("row_count")),
                "exact_match": _float(stats.get("exact_match")),
                "answerable_exact_match": _float(stats.get("answerable_exact_match")),
                "behavior_accuracy": _float(stats.get("behavior_accuracy")),
                "attempt_rate": _float(stats.get("attempt_rate")),
                "unsafe_answer_rate": _float(stats.get("unsafe_answer_rate")),
                "correct_abstain_rate": _float(stats.get("correct_abstain_rate")),
            }
        best_mode = max(
            MODE_NAMES,
            key=lambda m: (
                per_mode[m]["exact_match"],
                per_mode[m]["behavior_accuracy"],
                -per_mode[m]["unsafe_answer_rate"],
            ),
        )
        out[family] = {
            "modes": per_mode,
            "best_mode": best_mode,
            "combined_vs_solver_delta": _delta(per_mode["combined"]["exact_match"], per_mode["solver_only"]["exact_match"]),
            "combined_vs_base_delta": _delta(per_mode["combined"]["exact_match"], per_mode["base_only"]["exact_match"]),
        }
    return out

def _build_decision(core_block: dict[str, Any], family_block: dict[str, Any], rule_block: dict[str, Any], anti_block: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    base = core_block["primary_values"]["base_only"]
    solver = core_block["primary_values"]["solver_only"]
    adapter = core_block["primary_values"]["adapter_only"]
    combined = core_block["primary_values"]["combined"]

    rule_combined = rule_block["primary_values"]["combined"]
    anti_combined = anti_block["primary_values"]["combined"]

    family_regressions = [
        fam for fam, block in family_block.items()
        if block["combined_vs_solver_delta"] < 0.0 and fam in FAMILY_PRIORITY
    ]

    leakage_score = _float(anti_block.get("leakage_score", 0.0))
    format_error_rate = _float(core_block["modes"]["combined"].get("unsafe_answer_rate", 0.0))

    train_v2a_150 = (
        adapter > base
        and combined > solver
        and combined > adapter
        and rule_combined >= solver
        and leakage_score <= 0.0
        and len(family_regressions) == 0
    )

    reasons = []
    if adapter <= base:
        reasons.append("adapter_did_not_beat_base")
    if combined <= solver:
        reasons.append("combined_did_not_beat_solver")
    if combined <= adapter:
        reasons.append("combined_did_not_beat_adapter")
    if rule_combined < solver:
        reasons.append("rule_holdout_regression")
    if leakage_score > 0.0:
        reasons.append("anti_leak_nonzero")
    if family_regressions:
        reasons.append(f"family_regressions:{','.join(family_regressions)}")

    if not train_v2a_150 and not reasons:
        reasons.append("insufficient_evidence")

    return {
        "train_v2a_150": train_v2a_150,
        "decision": "train_v2a_150" if train_v2a_150 else "STOP_ADAPTER_SCALING",
        "reason_codes": reasons,
        "family_regression_count": len(family_regressions),
        "leakage_score": leakage_score,
        "format_error_rate": format_error_rate,
        "rule_holdout_accuracy": rule_combined,
        "core_exact_match": combined,
    }

def build_eval_ladder(manifest_path: str | Path) -> dict[str, Any]:
    manifest = _load_manifest(Path(manifest_path))
    reports_block = manifest["reports"]

    core = _load_mode_reports(reports_block, "core_eval")
    family = _load_mode_reports(reports_block, "family_eval")
    rule = _load_mode_reports(reports_block, "rule_holdout")
    anti = _load_mode_reports(reports_block, "anti_leak")

    core_block = _build_dataset_block(core)
    family_block_dataset = _build_dataset_block(family)
    rule_block = _build_dataset_block(rule)
    anti_block = _build_dataset_block(anti)

    core_families = _family_summary(core["combined"])
    family_eval_families = _family_summary(family["combined"])

    family_block = _build_family_block(
        core_families=core_families,
        family_eval_families=family_eval_families,
        mode_family_reports={
            "base_only": _family_summary(core["base_only"]),
            "solver_only": _family_summary(core["solver_only"]),
            "adapter_only": _family_summary(core["adapter_only"]),
            "combined": _family_summary(core["combined"]),
        },
    )

    leakage_score = _float(manifest.get("leakage_score", 0.0))
    if leakage_score == 0.0:
        anti_combined = anti["combined"]
        leakage_score = _float(
            anti_combined.get("leakage_score")
            or anti_combined.get("duplicate_prompt_count")
            or anti_combined.get("unsafe_answer_rate_on_abstain_rows")
            or 0.0
        )
        if leakage_score > 1.0 and "row_count" in anti_combined:
            leakage_score = leakage_score / max(1.0, _float(anti_combined.get("row_count"), 1.0))

    format_error_rate = _float(
        core["combined"].get("format_error_rate", core["combined"].get("unsafe_answer_rate", 0.0))
    )

    warnings = []
    for name, block in {
        "core_eval": core_block,
        "family_eval": family_block_dataset,
        "rule_holdout": rule_block,
        "anti_leak": anti_block,
    }.items():
        if block["status"] != "PASS":
            warnings.append(f"{name}_dataset_not_pass")

    if leakage_score > 0.0:
        warnings.append("nonzero_leakage_score")

    decision = _build_decision(core_block, family_block, rule_block, anti_block, warnings)

    provenance = {
        dataset: {
            mode: {
                "path": reports_block[dataset][mode],
                "sha256": _sha256(Path(reports_block[dataset][mode])),
            }
            for mode in MODE_NAMES
        }
        for dataset in DATASET_NAMES
    }

    report = {
        "schema_version": 1,
        "created_by": "DAY2_EVAL_LADDER",
        "status": "PASS" if not warnings and all(b["status"] == "PASS" for b in (core_block, family_block_dataset, rule_block, anti_block)) else "WARN",
        "model_results_faked": False,
        "datasets": {
            "core_eval": core_block,
            "family_eval": family_block_dataset,
            "rule_holdout": rule_block,
            "anti_leak": anti_block,
        },
        "mode_metrics": {
            "core_eval": core_block["modes"],
            "family_eval": family_block_dataset["modes"],
            "rule_holdout": rule_block["modes"],
            "anti_leak": anti_block["modes"],
        },
        "by_family": family_block,
        "deltas": {
            "core_exact_match": {
                "solver_vs_base": _delta(core_block["primary_values"]["solver_only"], core_block["primary_values"]["base_only"]),
                "adapter_vs_base": _delta(core_block["primary_values"]["adapter_only"], core_block["primary_values"]["base_only"]),
                "combined_vs_base": _delta(core_block["primary_values"]["combined"], core_block["primary_values"]["base_only"]),
                "combined_vs_solver": _delta(core_block["primary_values"]["combined"], core_block["primary_values"]["solver_only"]),
                "combined_vs_adapter": _delta(core_block["primary_values"]["combined"], core_block["primary_values"]["adapter_only"]),
            }
        },
        "leakage_score": leakage_score,
        "format_error_rate": format_error_rate,
        "decision": decision,
        "warnings": sorted(set(warnings)),
        "provenance": provenance,
    }
    return report

def write_eval_ladder(report: dict[str, Any], out_path: str | Path) -> None:
    safe_write_text(out_path, json.dumps(report, indent=2, sort_keys=True) + "\n", field_name="day2_eval_ladder_report")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 2 evaluation ladder from real eval reports.")
    parser.add_argument("--manifest", required=True, help="JSON manifest with report paths for core/family/rule_holdout/anti_leak")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    report = build_eval_ladder(args.manifest)
    write_eval_ladder(report, args.out)
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"]["decision"],
        "train_v2a_150": report["decision"]["train_v2a_150"],
        "leakage_score": report["leakage_score"],
        "format_error_rate": report["format_error_rate"],
        "out": args.out,
    }, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

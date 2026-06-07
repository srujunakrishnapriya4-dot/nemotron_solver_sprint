from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MODE_NAMES = ("base_only", "solver_only", "adapter_only", "combined")
DATASET_NAMES = ("core_eval", "family_eval", "rule_holdout", "anti_leak")

PRIMARY_SCORE_KEYS = (
    "overall_accuracy",
    "exact_match",
    "answerable_exact_match",
    "behavior_accuracy",
    "overall_score",
)

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _validate_blob(blob: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(blob, dict):
        raise TypeError("Each manifest entry must be an object")
    path = Path(blob["path"])
    expected_sha = str(blob["sha256"])
    expected_size = int(blob["size_bytes"])

    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")

    actual_size = path.stat().st_size
    actual_sha = _sha256(path)

    if actual_size != expected_size:
        raise ValueError(f"Size mismatch for {path}: expected {expected_size}, got {actual_size}")
    if actual_sha != expected_sha:
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected_sha}, got {actual_sha}")

    return _read_json(path)

def _score_from_report(report: dict[str, Any]) -> float:
    for key in PRIMARY_SCORE_KEYS:
        value = report.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    # Common nested locations used by our eval stack.
    evals = report.get("evals", {})
    if isinstance(evals, dict):
        for key in PRIMARY_SCORE_KEYS:
            value = evals.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    raise KeyError(
        f"Could not locate a primary score in report keys: {list(report.keys())[:20]}"
    )

def _mode_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "execution_status": report.get("execution_status"),
        "quality_status": report.get("quality_status"),
        "row_count": int(report.get("row_count", 0) or 0),
        "attempted_count": int(report.get("attempted_count", 0) or 0),
        "exact_match": float(report.get("exact_match", report.get("overall_accuracy", 0.0)) or 0.0),
        "answerable_exact_match": float(report.get("answerable_exact_match", 0.0) or 0.0),
        "behavior_accuracy": float(report.get("behavior_accuracy", 0.0) or 0.0),
        "attempt_rate": float(report.get("attempt_rate", 0.0) or 0.0),
        "unsafe_answer_rate": float(report.get("unsafe_answer_rate", 0.0) or 0.0),
        "correct_abstain_rate": float(report.get("correct_abstain_rate", 0.0) or 0.0),
        "wrong_answer_count": int(report.get("wrong_answer_count", 0) or 0),
        "unsafe_answer_count": int(report.get("unsafe_answer_count", 0) or 0),
        "correct_abstain_count": int(report.get("correct_abstain_count", 0) or 0),
    }

def _family_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = report.get("by_family") or report.get("by_family_accuracy") or {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for family, stats in raw.items():
        if not isinstance(stats, dict):
            continue
        out[str(family)] = {
            "row_count": int(stats.get("row_count", 0) or 0),
            "exact_match": float(stats.get("exact_match", stats.get("overall_accuracy", 0.0)) or 0.0),
            "answerable_exact_match": float(stats.get("answerable_exact_match", 0.0) or 0.0),
            "behavior_accuracy": float(stats.get("behavior_accuracy", 0.0) or 0.0),
            "attempt_rate": float(stats.get("attempt_rate", 0.0) or 0.0),
            "unsafe_answer_rate": float(stats.get("unsafe_answer_rate", 0.0) or 0.0),
            "correct_abstain_rate": float(stats.get("correct_abstain_rate", 0.0) or 0.0),
        }
    return out

def _delta(a: float, b: float) -> float:
    return round(a - b, 8)

def _best_mode(mode_summaries: dict[str, dict[str, Any]]) -> str:
    def key(mode: str) -> tuple[float, float, float]:
        s = mode_summaries[mode]
        return (
            float(s["exact_match"]),
            float(s["behavior_accuracy"]),
            -float(s["unsafe_answer_rate"]),
        )
    return max(MODE_NAMES, key=key)

def _dataset_block(mode_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries = {mode: _mode_summary(report) for mode, report in mode_reports.items()}
    return {
        "status": "PASS" if all(str(r.get("status", "")).upper() == "PASS" for r in mode_reports.values()) else "WARN",
        "modes": summaries,
        "best_mode": _best_mode(summaries),
        "primary_values": {mode: _score_from_report(report) for mode, report in mode_reports.items()},
    }

def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if int(manifest.get("schema_version", 0) or 0) != 1:
        raise ValueError("manifest schema_version must be 1")
    reports = manifest.get("reports")
    if not isinstance(reports, dict):
        raise ValueError("manifest must contain a top-level 'reports' object")
    for dataset in DATASET_NAMES:
        if dataset not in reports:
            raise ValueError(f"manifest missing dataset block: {dataset}")
        block = reports[dataset]
        if not isinstance(block, dict):
            raise TypeError(f"{dataset} block must be an object")
        for mode in MODE_NAMES:
            if mode not in block:
                raise ValueError(f"manifest missing {dataset}.{mode}")
            _validate_blob(block[mode])
    return manifest

def _decision(core_block: dict[str, Any], family_block: dict[str, Any], rule_block: dict[str, Any], anti_block: dict[str, Any]) -> dict[str, Any]:
    base = core_block["primary_values"]["base_only"]
    solver = core_block["primary_values"]["solver_only"]
    adapter = core_block["primary_values"]["adapter_only"]
    combined = core_block["primary_values"]["combined"]

    family_regressions = [
        fam for fam, row in family_block.items()
        if fam in ("roman_numeral", "unit_conversion", "numeric_formula", "bit_manipulation", "symbol_mapping", "char_cipher", "word_cipher", "format_only")
        and row["combined_vs_solver_delta"] < 0.0
    ]

    leakage_score = float(anti_block.get("leakage_score", 0.0) or 0.0)
    rule_holdout = float(rule_block["primary_values"]["combined"])

    train_v2a_150 = (
        adapter > base
        and combined > solver
        and combined > adapter
        and rule_holdout >= solver
        and leakage_score <= 0.0
        and not family_regressions
    )

    reason_codes = []
    if adapter <= base:
        reason_codes.append("adapter_not_better_than_base")
    if combined <= solver:
        reason_codes.append("combined_not_better_than_solver")
    if combined <= adapter:
        reason_codes.append("combined_not_better_than_adapter")
    if rule_holdout < solver:
        reason_codes.append("rule_holdout_regressed")
    if leakage_score > 0.0:
        reason_codes.append("nonzero_leakage")
    if family_regressions:
        reason_codes.append("family_regressions:" + ",".join(family_regressions))

    return {
        "train_v2a_150": train_v2a_150,
        "decision": "train_v2a_150" if train_v2a_150 else "STOP_ADAPTER_SCALING",
        "reason_codes": reason_codes,
        "family_regression_count": len(family_regressions),
        "rule_holdout_accuracy": rule_holdout,
        "leakage_score": leakage_score,
        "core_exact_match": combined,
        "adapter_exact_match": adapter,
        "solver_exact_match": solver,
        "base_exact_match": base,
    }

def build_eval_ladder(manifest_path: str | Path) -> dict[str, Any]:
    manifest = _load_manifest(Path(manifest_path))
    blocks = manifest["reports"]

    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASET_NAMES:
        loaded[dataset] = {}
        for mode in MODE_NAMES:
            loaded[dataset][mode] = _validate_blob(blocks[dataset][mode])

    core = _dataset_block(loaded["core_eval"])
    family = _dataset_block(loaded["family_eval"])
    rule = _dataset_block(loaded["rule_holdout"])
    anti = _dataset_block(loaded["anti_leak"])

    # Build family-wise comparison from the combined reports and available per-family metrics.
    base_families = _family_map(loaded["core_eval"]["base_only"])
    solver_families = _family_map(loaded["core_eval"]["solver_only"])
    adapter_families = _family_map(loaded["core_eval"]["adapter_only"])
    combined_families = _family_map(loaded["core_eval"]["combined"])

    by_family: dict[str, Any] = {}
    all_families = sorted(set(base_families) | set(solver_families) | set(adapter_families) | set(combined_families))
    for family_name in all_families:
        b = base_families.get(family_name, {})
        s = solver_families.get(family_name, {})
        a = adapter_families.get(family_name, {})
        c = combined_families.get(family_name, {})
        by_family[family_name] = {
            "base_only": b,
            "solver_only": s,
            "adapter_only": a,
            "combined": c,
            "combined_vs_solver_delta": _delta(float(c.get("exact_match", 0.0)), float(s.get("exact_match", 0.0))),
            "combined_vs_base_delta": _delta(float(c.get("exact_match", 0.0)), float(b.get("exact_match", 0.0))),
            "combined_vs_adapter_delta": _delta(float(c.get("exact_match", 0.0)), float(a.get("exact_match", 0.0))),
        }

    leakage_score = float(
        anti["modes"]["combined"].get("unsafe_answer_rate", 0.0)
    )
    format_error_rate = float(core["modes"]["combined"].get("unsafe_answer_rate", 0.0))

    decision = _decision(core, by_family, rule, {"leakage_score": leakage_score})

    report = {
        "schema_version": 1,
        "created_by": "DAY2_EVAL_LADDER",
        "status": "PASS" if all(x["status"] == "PASS" for x in (core, family, rule, anti)) else "WARN",
        "datasets": {
            "core_eval": core,
            "family_eval": family,
            "rule_holdout": rule,
            "anti_leak": anti,
        },
        "by_family": by_family,
        "leakage_score": leakage_score,
        "format_error_rate": format_error_rate,
        "decision": decision,
        "warnings": [],
        "model_results_faked": False,
    }
    return report

def write_eval_ladder(report: dict[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 2 eval ladder from strict JSON reports.")
    parser.add_argument("--manifest", required=True)
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

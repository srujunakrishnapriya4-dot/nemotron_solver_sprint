from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked
from kaggle_anti086.eval.eval_ladder import build_eval_ladder


EVAL_KEYS = ("private_like_512", "family_hard_512", "rule_holdout_512", "anti_leak_256")
ORDER = ("combined_v2a_50", "v2a_50_adapter_only", "solver_only", "base")
THRESHOLDS = {
    "minimum_scale_to_150": {
        "combined_beats_solver_only": True,
        "combined_beats_adapter_only": True,
        "adapter_invalid_rate_not_worse": True,
        "rule_holdout_not_collapsed": True,
    },
    "public_0_93_plausibility": {
        "private_like_512": 0.90,
        "family_hard_512": 0.85,
        "rule_holdout_512": 0.85,
        "no_family_below": 0.75,
    },
}


def load_report_or_blocked(path: str | Path | None, candidate: str) -> dict[str, Any]:
    if not path:
        return {"status": "BLOCKED", "candidate_name": candidate, "failures": ["report_path_missing"]}
    p = Path(path)
    if not p.exists():
        return {"status": "BLOCKED", "candidate_name": candidate, "failures": ["report_missing"]}
    data = read_json(p)
    data.setdefault("candidate_name", candidate)
    return data


def normalize_eval_report(report: dict[str, Any], candidate: str) -> dict[str, Any]:
    failures = list(report.get("failures", []))
    warnings = list(report.get("warnings", []))
    status = str(report.get("status", "BLOCKED"))
    available = status in {"PASS", "WARN"} and not failures and _has_any_score(report)
    scores = {key: _score_for(report, key) for key in EVAL_KEYS}
    known_scores = [value for value in scores.values() if value is not None]
    overall_raw = _first_present(report, ("overall_score", "overall_accuracy", "exact_match"))
    overall = float(overall_raw) if overall_raw is not None else (sum(known_scores) / len(known_scores) if known_scores else 0.0)
    invalid_raw = _first_present(report, ("invalid_answer_rate", "invalid_format_rate"))
    format_raw = _first_present(report, ("format_error_rate", "invalid_format_rate"))
    invalid_rate = float(invalid_raw) if invalid_raw is not None else 0.0
    format_rate = float(format_raw) if format_raw is not None else 0.0
    no_family_zero = bool(report.get("no_family_zero", _no_family_zero(report)))
    blockers = []
    if not available:
        blockers.append("eval_incomplete_or_missing")
    if scores["rule_holdout_512"] is not None and scores["rule_holdout_512"] < 0.85:
        blockers.append("rule_holdout_collapse")
    if not no_family_zero:
        blockers.append("family_zero_detected")
    if invalid_rate > 0.05:
        blockers.append("invalid_answer_rate_high")
    return {
        "candidate": candidate,
        "available": available,
        "overall_score": overall if available else None,
        "private_like_512": scores["private_like_512"],
        "family_hard_512": scores["family_hard_512"],
        "rule_holdout_512": scores["rule_holdout_512"],
        "anti_leak_256": scores["anti_leak_256"],
        "format_error_rate": format_rate,
        "invalid_answer_rate": invalid_rate,
        "no_family_zero": no_family_zero,
        "beats_base": None,
        "beats_solver_only": None,
        "beats_adapter_only": None,
        "regression_flags": list(report.get("regression_flags", [])),
        "blockers": blockers,
        "raw_status": status,
        "failures": failures,
        "warnings": warnings,
    }


def build_candidate_ranking(
    *,
    base_report: dict[str, Any],
    solver_report: dict[str, Any],
    adapter_report: dict[str, Any],
    combined_report: dict[str, Any],
    eval_ladder_report: dict[str, Any] | None = None,
    eval_ladder_path: str | None = None,
) -> dict[str, Any]:
    candidates = {
        "base": normalize_eval_report(base_report, "base"),
        "solver_only": normalize_eval_report(solver_report, "solver_only"),
        "v2a_50_adapter_only": normalize_eval_report(adapter_report, "v2a_50_adapter_only"),
        "combined_v2a_50": normalize_eval_report(combined_report, "combined_v2a_50"),
    }
    _attach_comparisons(candidates)
    ranking = sorted(candidates.values(), key=lambda row: (row["available"], row["overall_score"] if row["overall_score"] is not None else -1.0, -ORDER.index(row["candidate"])), reverse=True)
    failures: list[str] = []
    warnings: list[str] = []
    incomplete = [name for name, row in candidates.items() if not row["available"]]
    if incomplete:
        failures.append("incomplete_eval_reports:" + ",".join(sorted(incomplete)))
    if eval_ladder_report is None:
        warnings.append("ladder_missing_legacy_mode")
    ladder_decision = (eval_ladder_report or {}).get("decision", {})
    reason_codes = list(ladder_decision.get("reason_codes", []))
    scale_to_150 = bool(ladder_decision.get("train_v2a_150", False))
    lora_path_alive = _lora_path_from_ladder(ladder_decision)
    decision = {
        "lora_path_alive": lora_path_alive,
        "scale_to_150_allowed": scale_to_150,
        "scale_to_300_allowed": False,
        "fix_router_first": any(code in reason_codes for code in ("combined_not_better_than_solver", "priority_family_regression")),
        "focus_solver_first": any(code in reason_codes for code in ("adapter_not_better_than_base", "combined_not_better_than_adapter")),
        "submit_recommended": False,
        "packaging_allowed": False,
        "submission_allowed": False,
    }
    if eval_ladder_report is None:
        status = "BLOCKED" if failures else "WARN"
    else:
        status = "PASS" if scale_to_150 else ("FAIL" if eval_ladder_report.get("status") == "FAIL" else "WARN")
    return {
        "status": status,
        "ranking": ranking,
        "decision": decision,
        "thresholds": THRESHOLDS,
        "eval_ladder_path": eval_ladder_path,
        "eval_ladder": eval_ladder_report,
        "ladder_decision_reason_codes": reason_codes,
        "failures": failures,
        "warnings": warnings,
        "leaderboard_claim": False,
        "no_0_93_evidence": not scale_to_150,
        "no_0_95_evidence": True,
    }


def _attach_comparisons(candidates: dict[str, dict[str, Any]]) -> None:
    base = candidates["base"]
    solver = candidates["solver_only"]
    adapter = candidates["v2a_50_adapter_only"]
    for row in candidates.values():
        row["beats_base"] = _gt(row, base)
        row["beats_solver_only"] = _gt(row, solver)
        row["beats_adapter_only"] = _gt(row, adapter)


def _gt(left: dict[str, Any], right: dict[str, Any]) -> bool | None:
    if not left["available"] or not right["available"]:
        return None
    return float(left["overall_score"]) > float(right["overall_score"])


def _lora_path_from_ladder(ladder_decision: dict[str, Any]) -> bool | None:
    if not ladder_decision:
        return None
    reason_codes = set(ladder_decision.get("reason_codes", []))
    if "adapter_not_better_than_base" in reason_codes:
        return False
    adapter = ladder_decision.get("adapter_exact_match")
    base = ladder_decision.get("base_exact_match")
    if adapter is None or base is None:
        return None
    return float(adapter) > float(base)


def _has_any_score(report: dict[str, Any]) -> bool:
    return any(_score_for(report, key) is not None for key in EVAL_KEYS) or _first_present(report, ("overall_score", "overall_accuracy", "exact_match")) is not None


def _first_present(report: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if report.get(key) is not None:
            return report[key]
    return None


def _score_for(report: dict[str, Any], key: str) -> float | None:
    aliases = {
        "private_like_512": ("private_like_512", "private_like_answerable_512"),
        "family_hard_512": ("family_hard_512", "family_hard_answerable_512"),
        "rule_holdout_512": ("rule_holdout_512", "rule_holdout_answerable_512"),
        "anti_leak_256": ("anti_leak_256", "anti_leak_answerable_256"),
    }
    for alias in aliases[key]:
        value = report.get(alias)
        if isinstance(value, dict):
            return _dict_score(value)
        if isinstance(value, (int, float)):
            return float(value)
    evals = report.get("evals", {})
    if isinstance(evals, dict):
        for alias in aliases[key]:
            value = evals.get(alias)
            if isinstance(value, dict):
                return _dict_score(value)
    return None


def _dict_score(value: dict[str, Any]) -> float | None:
    for key in ("overall_accuracy", "exact_match", "adapter_only_accuracy", "solver_override_accuracy", "v2a_exact", "base_exact"):
        if key in value and value[key] is not None:
            return float(value[key])
    return None


def _no_family_zero(report: dict[str, Any]) -> bool:
    by_family = report.get("by_family_accuracy", report.get("by_family", {}))
    if not isinstance(by_family, dict) or not by_family:
        return True
    for value in by_family.values():
        if isinstance(value, dict):
            score = value.get("overall_accuracy", value.get("exact_match", value.get("accuracy")))
        else:
            score = value
        if score is not None and float(score) <= 0.0:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day 11A candidate ranking and scale decision gate.")
    parser.add_argument("--base-report")
    parser.add_argument("--solver-report")
    parser.add_argument("--adapter-report")
    parser.add_argument("--combined-report")
    parser.add_argument("--eval-ladder-report")
    parser.add_argument("--eval-ladder-manifest")
    parser.add_argument("--out", default="artifacts/sprint11/day11a_candidate_ranking.json")
    parser.add_argument("--decision-out", default="artifacts/sprint11/day11a_decision_report.json")
    args = parser.parse_args(argv)
    ladder_report = None
    ladder_path = None
    if args.eval_ladder_report:
        ladder_path = args.eval_ladder_report
        ladder_report = read_json(args.eval_ladder_report)
    elif args.eval_ladder_manifest:
        ladder_path = args.eval_ladder_manifest
        ladder_report = build_eval_ladder(args.eval_ladder_manifest)
    elif not all([args.base_report, args.solver_report, args.adapter_report, args.combined_report]):
        raise SystemExit("eval_ladder_report_or_manifest_required")
    report = build_candidate_ranking(
        base_report=load_report_or_blocked(args.base_report, "base"),
        solver_report=load_report_or_blocked(args.solver_report, "solver_only"),
        adapter_report=load_report_or_blocked(args.adapter_report, "v2a_50_adapter_only"),
        combined_report=load_report_or_blocked(args.combined_report, "combined_v2a_50"),
        eval_ladder_report=ladder_report,
        eval_ladder_path=ladder_path,
    )
    write_json_checked(args.out, report, field_name="day11a_candidate_ranking")
    write_json_checked(args.decision_out, report["decision"] | {"status": report["status"], "failures": report["failures"], "warnings": report["warnings"]}, field_name="day11a_decision_report")
    print(json.dumps({"status": report["status"], "scale_to_150_allowed": report["decision"]["scale_to_150_allowed"], "submit_recommended": False}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

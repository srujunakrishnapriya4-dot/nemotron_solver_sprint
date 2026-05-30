from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash

from .component_scorecard import score_component
from .critical_gap_detector import detect_critical_gaps, readiness_from_gaps
from .implementation_contract import build_implementation_contract


COMPONENTS = {
    "parser_safety": ("src/nemotron_engine/competition_sprint/competition_prompt_adapter.py", "parse_competition_prompt"),
    "solver_family_coverage": ("src/nemotron_engine/competition_sprint/solver_coverage_report.py", "build_solver_coverage_report"),
    "verified_rule_extraction": ("artifacts/vex_progress/verified_rules.jsonl", ""),
    "synthetic_novelty": ("src/nemotron_engine/anti086_barrier/novelty_filter_v2.py", "filter_novelty_v2"),
    "private_like_validation": ("src/nemotron_engine/anti086_barrier/private_like_benchmark_v2.py", "build_private_like_benchmark_v2"),
    "token_masking_correctness": ("src/nemotron_engine/training_contract/token_mask_contract.py", "validate_token_mask_rows"),
    "training_path_correctness": ("kaggle_anti086/kaggle_train_anti086_adapter.py", "loss_weights"),
    "eval_metric_correctness": ("kaggle_anti086/kaggle_eval_anti086_vllm.py", "compute_eval_metrics"),
    "candidate_selection_correctness": ("kaggle_anti086/kaggle_final_candidate_selector.py", "select_candidate"),
    "package_gate_correctness": ("kaggle_anti086/kaggle_package_anti086_adapter.py", "decide_promotion"),
}


def run_repo_readiness_audit(
    repo_root: str | Path = ".",
    output_dir: str | Path = "artifacts/win_system",
) -> dict[str, Any]:
    root = Path(repo_root)
    scores = {}
    for name, (rel, token) in COMPONENTS.items():
        path = root / rel
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() and path.is_file() else ""
        scores[name] = score_component(
            name,
            evidence={
                "exists": path.exists(),
                "contains_token": (token in text) if token else path.exists(),
                "checks": {"exists": 50, "contains_token": 50},
                "fake_metric_detected": name == "eval_metric_correctness" and ('"prompt_copy_rate": 0.0' in text or '"answer_format_pass_rate": 1.0' in text),
                "full_prompt_loss_path": name == "training_path_correctness" and "labels = input_ids" in text,
            },
        ).__dict__
    gaps = detect_critical_gaps(root)
    coverage_path = root / "artifacts" / "win_system" / "solver_coverage_report.json"
    verified_total = None
    if coverage_path.exists():
        try:
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            verified_total = sum(coverage.get("solver_verified_correct_by_family", coverage.get("verified_correct_by_family", {})).values())
            if verified_total == 0:
                gaps.append("zero_verified_train_solver_coverage")
            if verified_total < 5000:
                gaps.append("verified_train_solver_coverage_below_5000")
            acceptance = coverage.get("sprint72_acceptance", {})
            if not acceptance.get("success_or_documented_bottleneck", False):
                gaps.append("hard_family_coverage_gate_not_satisfied")
            sprint75 = coverage.get("sprint75_acceptance")
            if sprint75 is not None and not sprint75.get("success", False):
                gaps.append("digit_symbol_induction_gate_not_satisfied")
            sprint77 = coverage.get("sprint77_acceptance")
            if sprint77 is not None and not sprint77.get("success", False):
                gaps.append("bit_breakthrough_gate_not_satisfied")
            sprint78 = coverage.get("sprint78_acceptance")
            if sprint78 is not None and not sprint78.get("success", False):
                gaps.append("bit_truth_table_mining_gate_not_satisfied")
        except Exception:
            gaps.append("solver_coverage_report_unreadable")
    verdict = readiness_from_gaps(gaps)
    if "zero_verified_train_solver_coverage" in gaps and verdict == "MAIN_READY":
        verdict = "MICRO_READY"
    if verdict == "MAIN_READY" and min(item["score"] for item in scores.values()) < 80:
        verdict = "MICRO_READY"
    payload = {
        "verdict": verdict,
        "component_scores": scores,
        "critical_gaps": gaps,
        "solver_verified_correct_total": verified_total,
        "contract": build_implementation_contract(),
    }
    payload["audit_hash"] = stable_hash(payload)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "readiness_audit.json").write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    (out / "CRITICAL_GAPS.md").write_text(_gaps_md(payload), encoding="utf-8")
    return payload


def _gaps_md(payload: dict[str, Any]) -> str:
    lines = [f"# Critical Gaps", "", f"Verdict: {payload['verdict']}", ""]
    if payload["critical_gaps"]:
        lines.extend(f"- {gap}" for gap in payload["critical_gaps"])
    else:
        lines.append("- No critical blockers detected by static audit.")
    return "\n".join(lines) + "\n"

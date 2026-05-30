from __future__ import annotations

from pathlib import Path
from typing import Any

from .adversarial_curriculum_optimizer import build_winmode_curricula
from .backend_capability_matrix import build_backend_capability_matrix
from .compute_budget_planner import build_compute_budget_plan
from .day8_submission_strategy import write_day8_submission_strategy
from .family_error_taxonomy import build_family_error_taxonomy
from .final_candidate_selector import select_final_candidate
from .parent_child_eval_protocol import build_parent_child_eval_protocol
from .rule_space_coverage_optimizer import optimize_rule_space_coverage
from .score_lift_hypothesis import build_score_lift_hypotheses


def build_win_mode_program(output_dir: str | Path = "artifacts/anti086") -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    backend = build_backend_capability_matrix(out / "backend_capability_matrix.json")
    compute = build_compute_budget_plan(out / "compute_budget_plan.json", out / "COMPUTE_BUDGET_PLAN.md")
    score = build_score_lift_hypotheses(out / "score_lift_hypotheses.json")
    taxonomy = build_family_error_taxonomy(out / "family_error_taxonomy.json", out / "FAMILY_ERROR_TAXONOMY.md")
    coverage = optimize_rule_space_coverage(output_dir=out)
    curriculum = build_winmode_curricula(output_dir=out)
    eval_protocol = build_parent_child_eval_protocol(out / "parent_child_eval_protocol.json")
    selector = select_final_candidate(
        [
            {"name": "known_public_parent", "is_parent": True, "rank": 32, "adapter_size_mb": 900, "adapter_size_limit_mb": 1500, "package_clean": True, "known_good_public": True},
            {"name": "pending_child", "rank": 32, "adapter_size_mb": 10**9, "adapter_size_limit_mb": 1500, "package_clean": False, "private_like_delta": 0.0},
        ],
        output_path=out / "final_candidate_decision.json",
    )
    strategy_text = write_day8_submission_strategy(out / "DAY8_SUBMISSION_STRATEGY.md")
    return {
        "backend": backend,
        "compute": compute,
        "score_hypotheses_count": len(score.get("interventions", [])),
        "taxonomy_families": len(taxonomy.get("families", {})),
        "coverage": coverage,
        "curriculum": curriculum,
        "eval_protocol_hash": eval_protocol.get("protocol_hash"),
        "final_selector": selector,
        "day8_strategy_written": bool(strategy_text),
    }

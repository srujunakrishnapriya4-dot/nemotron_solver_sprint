from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def build_parent_child_eval_protocol(
    output_path: str | Path = "artifacts/anti086/parent_child_eval_protocol.json",
) -> dict[str, Any]:
    protocol = {
        "modes": {
            "smoke_8": {"rows": 8, "purpose": "nonsense and format smoke only", "promotion_evidence": False},
            "public_like_60": {"rows": 60, "purpose": "detect public-like collapse", "promotion_evidence": True},
            "family_hard_60": {"rows": 60, "purpose": "hard-family lift", "promotion_evidence": True},
            "rule_holdout_60": {"rows": 60, "purpose": "rule generalization", "promotion_evidence": True},
            "adversarial_holdout_60": {"rows": 60, "purpose": "synthetic hard holdout", "promotion_evidence": True},
            "final_300": {"rows": 300, "purpose": "final confirmation only if time permits", "promotion_evidence": True},
        },
        "metrics": (
            "exact_match",
            "answer_format_pass",
            "empty_output_rate",
            "prompt_copy_rate",
            "by_family_accuracy",
            "hard_family_aggregate",
            "private_like_weighted_score",
            "regression_list",
        ),
        "generation_defaults": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64, "use_vllm": True},
        "minimum_promotion_rows": 60,
        "must_compare": ("parent_adapter", "child_adapter"),
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, sort_keys=True, indent=2), encoding="utf-8")
    return protocol


def validate_eval_report_for_promotion(report: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    rows = int(report.get("rows_evaluated", 0))
    if rows < 8:
        reasons.append("one_row_or_too_small_eval")
    if rows < 60 and report.get("stage") not in {"micro", "smoke"}:
        reasons.append("insufficient_rows_for_stage_promotion")
    if float(report.get("answer_format_pass", 0.0)) < 0.98:
        reasons.append("answer_format_instability")
    if float(report.get("empty_output_rate", 1.0)) > 0.0:
        reasons.append("empty_outputs")
    if float(report.get("prompt_copy_rate", 1.0)) > 0.0:
        reasons.append("prompt_copy_outputs")
    if float(report.get("public_like_delta", 0.0)) < -0.005:
        reasons.append("public_like_regression")
    if report.get("stage") not in {"micro", "smoke"} and float(report.get("private_like_delta", 0.0)) <= 0.0:
        reasons.append("no_private_like_lift")
    return {"valid_for_promotion": not reasons, "reasons": sorted(set(reasons))}

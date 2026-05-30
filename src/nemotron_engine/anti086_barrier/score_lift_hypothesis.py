from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


INTERVENTIONS = {
    "raw_direct_sft": (0.00, 0.01, "low", "low", "low", "micro loss and no format regression"),
    "verified_short_rule_traces": (0.005, 0.02, "low", "medium", "medium", "rule_holdout improvement"),
    "adversarial_filtered_synthetic": (0.00, 0.04, "medium", "medium", "high", "family_hard and rule_holdout gains without public collapse"),
    "contrastive_wrong_rule_examples": (0.00, 0.025, "medium", "low", "medium", "fewer plausible wrong-rule outputs"),
    "family_priority_oversampling": (0.00, 0.02, "medium", "medium", "low", "hard family aggregate improves"),
    "rule_holdout_curriculum": (0.00, 0.035, "medium", "low", "medium", "unseen rule IDs improve"),
    "lm_head_lora": (0.00, 0.02, "medium", "medium", "medium", "answer format and hidden family gains"),
    "reset_weights_true": (-0.02, 0.04, "high", "medium", "high", "fresh VEX-style run beats parent on private-like"),
    "parent_continuation": (0.00, 0.015, "medium", "medium", "medium", "non-regression against parent"),
    "grpo_dpo_after_stable_sft": (0.00, 0.03, "high", "high", "high", "validated reward distribution and stable SFT"),
}


def build_score_lift_hypotheses(output_path: str | Path = "artifacts/anti086/score_lift_hypotheses.json") -> dict[str, Any]:
    interventions = {
        name: {
            "expected_public_lift": vals[0],
            "expected_private_lift": vals[1],
            "risk_public_collapse": vals[2],
            "risk_private_overfit": vals[3],
            "compute_cost": vals[4],
            "required_evidence_to_scale": vals[5],
        }
        for name, vals in INTERVENTIONS.items()
    }
    payload = {
        "interventions": interventions,
        "hard_truths": (
            "0.95 requires large hidden-family gains, not small tuning.",
            "0.86_to_0.95 likely requires solving most equation/bit/gravity/unit hidden failures.",
            "If private-like hard splits do not improve, 0.95 is not credible.",
        ),
    }
    payload["hypothesis_hash"] = stable_hash(payload)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload

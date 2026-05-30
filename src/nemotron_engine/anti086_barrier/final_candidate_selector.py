from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def select_final_candidate(
    candidates: list[dict[str, Any]],
    *,
    parent_name: str = "known_public_parent",
    output_path: str | Path = "artifacts/anti086/final_candidate_decision.json",
) -> dict[str, Any]:
    reasons: list[str] = []
    viable = []
    for candidate in candidates:
        reject = _rejection_reasons(candidate)
        if reject:
            reasons.append(f"{candidate.get('name', 'candidate')}:{','.join(reject)}")
            continue
        viable.append(candidate)
    viable_children = [c for c in viable if not c.get("is_parent", False)]
    improving_children = [
        c
        for c in viable_children
        if float(c.get("private_like_delta", 0.0)) > 0.0
        and float(c.get("public_like_delta", 0.0)) >= -0.005
        and not bool(c.get("synthetic_only_gain", False))
    ]
    if improving_children:
        chosen = max(
            improving_children,
            key=lambda c: (
                float(c.get("private_like_weighted_score", 0.0)),
                float(c.get("family_hard_delta", 0.0)),
                float(c.get("public_score", 0.0)),
                str(c.get("name", "")),
            ),
        )
        decision = "SELECT_CHILD"
        rationale = "child improves private-like hard metrics without public-like collapse"
    else:
        parent = next((c for c in candidates if c.get("is_parent", False)), {"name": parent_name, "adapter_path": None})
        chosen = parent
        decision = "FALLBACK_PARENT"
        rationale = "no child beat parent on private-like gates"
    payload = {
        "decision": decision,
        "selected_name": chosen.get("name", parent_name),
        "selected_adapter_path": chosen.get("adapter_path"),
        "rationale": rationale,
        "rejected_reasons": sorted(reasons),
        "policy": "prefer_private_like_lift_reject_public_like_only_gain",
    }
    payload["decision_hash"] = stable_hash(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload


def _rejection_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons = []
    if int(candidate.get("rank", 999)) > 32:
        reasons.append("rank_gt_32")
    if float(candidate.get("adapter_size_mb", 10**9)) > float(candidate.get("adapter_size_limit_mb", 1500)) and not candidate.get("known_good_public", False):
        reasons.append("adapter_too_large")
    if not candidate.get("package_clean", False):
        reasons.append("package_not_clean")
    if candidate.get("failed_lineage", False):
        reasons.append("failed_lineage")
    if candidate.get("answer_format_pass", 1.0) < 0.98:
        reasons.append("answer_format_instability")
    if candidate.get("synthetic_only_gain", False):
        reasons.append("synthetic_only_gain")
    if float(candidate.get("public_like_delta", 0.0)) > 0.0 and float(candidate.get("private_like_delta", 0.0)) < 0.0:
        reasons.append("public_like_only_gain_private_regression")
    return reasons

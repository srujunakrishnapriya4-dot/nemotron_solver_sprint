from __future__ import annotations


def decide_promotion(stage: str, evidence: dict) -> dict:
    reasons: list[str] = []
    if not evidence.get("token_mask_valid", False):
        reasons.append("token_mask_invalid")
    if not evidence.get("loss_finite", False):
        reasons.append("loss_not_finite")
    if evidence.get("adapter_size_mb", 10**9) > evidence.get("adapter_size_limit_mb", 1500):
        reasons.append("adapter_size_exceeds_limit")
    if evidence.get("empty_output_count", 1) > 0:
        reasons.append("empty_or_nonsense_outputs")
    if evidence.get("prompt_copy_count", 0) > 0:
        reasons.append("prompt_copy_outputs")
    if stage in {"v1_to_v2", "v2_to_v3", "submission"}:
        if evidence.get("public_like_delta", 0.0) < -0.005:
            reasons.append("public_like_collapse")
        if evidence.get("private_like_delta", 0.0) <= 0.0:
            reasons.append("no_private_like_improvement")
    if stage == "v2_to_v3" and evidence.get("synthetic_gain_only", False):
        reasons.append("synthetic_only_gain")
    if stage == "submission":
        if not evidence.get("package_clean", False):
            reasons.append("package_not_clean")
        if evidence.get("rank", 999) > 32:
            reasons.append("rank_gt_32")
        if evidence.get("failed_lineage", False):
            reasons.append("failed_lineage")
    return {"decision": "PROMOTE" if not reasons else "REJECT", "reasons": sorted(set(reasons))}

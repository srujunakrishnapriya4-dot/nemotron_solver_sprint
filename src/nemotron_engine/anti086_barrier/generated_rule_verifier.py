from __future__ import annotations

from typing import Any


def verify_generated_rule(row: dict[str, Any]) -> dict[str, Any]:
    required = ("prompt", "answer", "family", "rule_id", "generation_hash")
    missing = [key for key in required if not row.get(key)]
    if missing:
        return {"verifier_status": "reject", "reason": f"missing:{','.join(missing)}"}
    if float(row.get("ambiguity_score", 1.0)) > 0.30:
        return {"verifier_status": "reject", "reason": "ambiguous_rule"}
    if float(row.get("difficulty_score", 0.0)) < 0.45:
        return {"verifier_status": "reject", "reason": "too_easy"}
    return {"verifier_status": "verified", "reason": "verified"}

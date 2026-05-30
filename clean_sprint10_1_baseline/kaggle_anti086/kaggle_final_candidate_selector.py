from __future__ import annotations

import json
from pathlib import Path


MICRO_LABEL = "INFRASTRUCTURE STACK TEST ONLY - NOT A 0.95 CANDIDATE - NOT MAIN TRAINING - NOT SUBMISSION READY"
V1_LABEL = "V1 SMALL TRAINING EXPERIMENT - NOT SUBMISSION READY - NO PACKAGE - NO KAGGLE SUBMIT"
V1B_LABEL = "V1B DIRECT-ANSWER PARENT-CONTINUATION EXPERIMENT - NO V2 - NO PACKAGE - NO SUBMISSION"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def score_stage(stage: str) -> tuple[float, list[str]]:
    gate_dir = Path("/kaggle/working/anti086_stage_logs")
    if stage == "micro":
        eval_gate = _read_json(gate_dir / "eval_micro.gate.json")
        train_gate = _read_json(gate_dir / "train_micro.gate.json")
        reasons = []
        if not eval_gate or eval_gate.get("decision") != "PASS":
            reasons.append("missing_eval_evidence")
        if not train_gate or train_gate.get("decision") != "PASS":
            reasons.append("missing_train_evidence")
        reasons.append("micro_child_not_submittable")
        return 0.0, reasons
    if stage == "v1":
        return _score_parent_child_stage("v1")
    if stage == "v1b":
        return _score_parent_child_stage("v1b")
    return 0.0, ["v2_v3_forbidden_in_sprint10"]


def _score_parent_child_stage(stage: str) -> tuple[float, list[str]]:
    gate_dir = Path("/kaggle/working/anti086_stage_logs")
    eval_gate = _read_json(gate_dir / f"eval_{stage}.gate.json")
    train_gate = _read_json(gate_dir / f"train_{stage}.gate.json")
    reasons = []
    if not train_gate or train_gate.get("decision") != "PASS":
        reasons.append("missing_train_evidence")
    if not eval_gate:
        reasons.append("missing_eval_evidence")
    elif eval_gate.get("decision") != "PASS":
        reasons.append(f"{stage}_gate_not_passed")
    if train_gate and int(train_gate.get("rank", 999)) > 16:
        reasons.append("rank_gt_16")
    if train_gate and float(train_gate.get("adapter_size_mb", 10**9)) > 512:
        reasons.append("adapter_too_large")
    if eval_gate:
        status = eval_gate.get("v1_gate_status")
        if status in {"INCONCLUSIVE_PARENT_MISSING", "FAIL_CHILD_REGRESSED", "INCONCLUSIVE_BOTH_ZERO", "INVALID_EVAL_PARENT_ZERO", "FAIL_CHILD_ZERO"}:
            reasons.append(str(status).lower())
        if eval_gate.get("parent_eval_status") != "available":
            reasons.append("parent_eval_missing")
    return float(eval_gate.get("child_minus_parent_delta", 0.0) or 0.0) if eval_gate else 0.0, reasons


def select_candidate() -> dict:
    v1_score, v1_reasons = score_stage("v1")
    v1b_score, v1b_reasons = score_stage("v1b")
    decision = "NEED_V1_V2_EVIDENCE"
    if v1b_reasons and v1_reasons:
        decision = "MICRO_CHILD_NOT_SUBMITTABLE"
    elif not v1b_reasons and v1b_score >= 0:
        decision = "NEED_V1_V2_EVIDENCE"
    elif not v1_reasons and v1_score >= 0:
        decision = "NEED_V1_V2_EVIDENCE"
    payload = {
        "decision": decision,
        "allowed_submission_decision": "KEEP_PARENT_BASELINE",
        "required_for_child_submission": "NEED_V1_V2_EVIDENCE",
        "v1_score_delta": v1_score,
        "v1_reasons": v1_reasons,
        "v1b_score_delta": v1b_score,
        "v1b_reasons": v1b_reasons,
        "micro_stack_test_label": MICRO_LABEL,
        "v1_experiment_label": V1_LABEL,
        "v1b_experiment_label": V1B_LABEL,
        "not_submission_ready": True,
        "not_095_candidate": True,
    }
    output = Path("/kaggle/working/final_candidate_decision.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    print(json.dumps(select_candidate(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

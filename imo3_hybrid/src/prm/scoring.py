from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .base import (
    ProcessObligationInput,
    ProcessScoreInput,
    ProcessScoreSummary,
    ProcessStepInput,
    StepQualityScore,
)


_ASSUMPTION_RE = re.compile(r"\b(assume|suppose|obvious|obviously|wlog|without loss)\b", re.IGNORECASE)


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _clamp01(sum(float(value) for value in values) / len(values))


def coerce_process_input(value: ProcessScoreInput | Mapping[str, Any]) -> ProcessScoreInput:
    if isinstance(value, ProcessScoreInput):
        return value

    steps = []
    for raw in list(value.get("steps", ()) or ()):
        if isinstance(raw, ProcessStepInput):
            steps.append(raw)
            continue
        if not isinstance(raw, Mapping):
            continue
        steps.append(
            ProcessStepInput(
                step_index=max(1, int(raw.get("step_index", raw.get("index", 1)))),
                description=_normalize_text(raw.get("description", "")),
                operator_name=_normalize_text(raw.get("operator_name", raw.get("operator_used", ""))) or None,
                symbolic_valid=bool(raw.get("symbolic_valid", True)),
                proof_obligation_ids=tuple(
                    _normalize_text(item)
                    for item in list(raw.get("proof_obligation_ids", ()) or ())
                    if _normalize_text(item)
                ),
                state_fingerprint=_normalize_text(raw.get("state_fingerprint", "")) or None,
            )
        )

    obligations = []
    for raw in list(value.get("proof_obligations", ()) or ()):
        if isinstance(raw, ProcessObligationInput):
            obligations.append(raw)
            continue
        if not isinstance(raw, Mapping):
            continue
        obligation_id = _normalize_text(raw.get("obligation_id", ""))
        if not obligation_id:
            continue
        obligations.append(
            ProcessObligationInput(
                obligation_id=obligation_id,
                status=_normalize_text(raw.get("status", "open")) or "open",
                claim=_normalize_text(raw.get("claim", "")),
                evidence_kind_required=_normalize_text(raw.get("evidence_kind_required", "other")) or "other",
                originating_node_id=_normalize_text(raw.get("originating_node_id", "")),
                target_goal_id=_normalize_text(raw.get("target_goal_id", "")) or None,
            )
        )

    return ProcessScoreInput(
        problem_id=_normalize_text(value.get("problem_id", "")),
        branch_id=_normalize_text(value.get("branch_id", "")),
        steps=tuple(steps),
        proof_obligations=tuple(obligations),
        route_uncertainty=_clamp01(value.get("route_uncertainty", 0.0)),
        retrieval_support=_clamp01(value.get("retrieval_support", 0.0)),
        symbolic_score=_clamp01(value.get("symbolic_score", 0.0)),
        symbolic_passed=bool(value.get("symbolic_passed", False)),
        exact_symbolic_match=bool(value.get("exact_symbolic_match", False)),
        logical_consistency_hint=(
            _clamp01(value["logical_consistency_hint"])
            if value.get("logical_consistency_hint") is not None
            else None
        ),
        completeness_hint=(
            _clamp01(value["completeness_hint"])
            if value.get("completeness_hint") is not None
            else None
        ),
        answer_present=bool(value.get("answer_present", False)),
        contradiction_count=max(0, int(value.get("contradiction_count", 0))),
        metadata=dict(value.get("metadata", {}) or {}),
    )


def score_process(value: ProcessScoreInput | Mapping[str, Any]) -> ProcessScoreSummary:
    data = coerce_process_input(value)
    obligations = list(data.proof_obligations)
    total_obligations = len(obligations)
    open_obligations = [
        item for item in obligations
        if item.status in {"open", "partial", "blocked"}
    ]
    contradicted_obligations = [item for item in obligations if item.status == "contradicted"]
    discharged_ids = {item.obligation_id for item in obligations if item.status == "discharged"}
    obligation_status = {item.obligation_id: item.status for item in obligations}

    open_obligation_burden = _clamp01(len(open_obligations) / max(1, total_obligations))
    contradiction_risk = _clamp01(
        max(
            len(contradicted_obligations) / max(1, total_obligations or 1),
            data.contradiction_count / max(1, len(data.steps) or 1),
        )
    )

    step_scores: list[StepQualityScore] = []
    running: list[float] = []
    previous_prefix_quality = 0.0

    for step in data.steps:
        obligation_ids = [item for item in step.proof_obligation_ids if item]
        if obligation_ids:
            resolved = sum(1 for item in obligation_ids if obligation_status.get(item) == "discharged")
            contradicted = sum(1 for item in obligation_ids if obligation_status.get(item) == "contradicted")
            obligation_coverage = _clamp01(resolved / max(1, len(obligation_ids)))
            local_contradiction = _clamp01(contradicted / max(1, len(obligation_ids)))
        else:
            obligation_coverage = 0.55 if step.description else 0.0
            local_contradiction = contradiction_risk

        assumption_penalty = 0.18 if _ASSUMPTION_RE.search(step.description) else 0.0
        symbolic_signal = 1.0 if step.symbolic_valid else 0.0

        step_quality = _clamp01(
            0.36 * symbolic_signal
            + 0.20 * (1.0 if step.description else 0.0)
            + 0.18 * obligation_coverage
            + 0.12 * (1.0 - open_obligation_burden)
            + 0.08 * data.symbolic_score
            + 0.06 * (1.0 - data.route_uncertainty)
            - 0.18 * local_contradiction
            - assumption_penalty
        )
        running.append(step_quality)

        computed_prefix_quality = _clamp01(
            0.64 * _mean(running)
            + 0.12 * data.symbolic_score
            + 0.08 * (1.0 if data.answer_present else 0.0)
            + 0.08 * (1.0 - open_obligation_burden)
            + 0.08 * data.retrieval_support
            - 0.18 * contradiction_risk
        )

        # Critical compatibility fix:
        # the first strong valid step should not have prefix quality below its own step quality.
        if step.step_index <= 1:
            prefix_quality = max(computed_prefix_quality, step_quality)
        else:
            prefix_quality = max(computed_prefix_quality, min(step_quality, previous_prefix_quality))

        repairability_hint = _clamp01(
            0.72
            - 0.30 * step_quality
            + 0.12 * open_obligation_burden
            - 0.10 * local_contradiction
        )

        notes = []
        if obligation_ids and resolved < len(obligation_ids):
            notes.append("open_obligation_support")
        if local_contradiction > 0.0:
            notes.append("local_contradiction")
        if assumption_penalty > 0.0:
            notes.append("assumption_language")

        step_scores.append(
            StepQualityScore(
                step_index=step.step_index,
                step_quality=step_quality,
                prefix_quality=prefix_quality,
                obligation_coverage=obligation_coverage,
                contradiction_risk=max(local_contradiction, contradiction_risk),
                repairability_hint=repairability_hint,
                notes=tuple(notes),
            )
        )
        previous_prefix_quality = prefix_quality

    mean_step_quality = _mean([item.step_quality for item in step_scores])
    prefix_quality = step_scores[-1].prefix_quality if step_scores else _clamp01(
        0.32 * data.symbolic_score
        + 0.18 * (1.0 if data.answer_present else 0.0)
        + 0.16 * (1.0 - open_obligation_burden)
        + 0.12 * data.retrieval_support
        + 0.10 * (1.0 - data.route_uncertainty)
        - 0.18 * contradiction_risk
    )

    branch_search_value = _clamp01(
        0.34 * prefix_quality
        + 0.24 * mean_step_quality
        + 0.14 * data.symbolic_score
        + 0.10 * (1.0 - open_obligation_burden)
        + 0.08 * data.retrieval_support
        + 0.06 * (1.0 if data.answer_present else 0.0)
        + 0.04 * (1.0 if data.exact_symbolic_match else 0.0)
        - 0.16 * contradiction_risk
        - 0.08 * data.route_uncertainty
    )

    return ProcessScoreSummary(
        step_scores=tuple(step_scores),
        mean_step_quality=mean_step_quality,
        prefix_quality=prefix_quality,
        open_obligation_burden=open_obligation_burden,
        contradiction_risk=contradiction_risk,
        branch_search_value=branch_search_value,
        metadata={
            "proof_obligation_count": total_obligations,
            "open_proof_obligation_count": len(open_obligations),
            "contradicted_proof_obligation_count": len(contradicted_obligations),
            "discharged_proof_obligation_count": len(discharged_ids),
        },
    )


class DeterministicProcessRewardModel:
    def score_process(self, value: ProcessScoreInput | Mapping[str, Any]) -> ProcessScoreSummary:
        return score_process(value)


__all__ = [
    "DeterministicProcessRewardModel",
    "coerce_process_input",
    "score_process",
]
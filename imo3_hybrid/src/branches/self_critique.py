# src/branches/self_critique.py
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.aggregation.canonicalize import canonicalize_competition_answer
from src.common.constants import ANSWER_MAX, ANSWER_MIN, SELF_CRITIQUE_TOP_K
from src.common.schemas import BranchTrace, FailureType, ReasoningStep


class CritiqueIssueCategory(str, Enum):
    ARITHMETIC_ERROR = "arithmetic_error"
    LOGICAL_GAP = "logical_gap"
    MISSING_CASE = "missing_case"
    SYMBOLIC_MISMATCH = "symbolic_mismatch"
    UNSUPPORTED_ASSUMPTION = "unsupported_assumption"
    COMPLETENESS_GAP = "completeness_gap"
    ANSWER_FORMAT = "answer_format"
    BRANCH_INCONSISTENCY = "branch_inconsistency"
    WEAK_EVIDENCE = "weak_evidence"
    FAILURE_PROPAGATION = "failure_propagation"


class CritiqueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CritiqueMode(str, Enum):
    MODEL = "model"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class CritiqueStage(str, Enum):
    MID_SEARCH = "mid_search"
    PRE_AGGREGATION = "pre_aggregation"


class CritiqueModelProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...


class CritiqueEvidence(BaseModel):
    source: str
    excerpt: str
    step_num: int | None = None
    operator_used: str | None = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class CritiqueFinding(BaseModel):
    category: CritiqueIssueCategory
    severity: CritiqueSeverity
    summary: str
    detail: str
    affected_steps: list[int] = Field(default_factory=list)
    evidence: list[CritiqueEvidence] = Field(default_factory=list)
    suggested_fix: str | None = None
    blocks_submission: bool = False


class CritiqueCorrection(BaseModel):
    corrected_answer: str | None = None
    corrected_reasoning: str | None = None
    correction_applied: bool = False
    correction_confidence: float = Field(0.0, ge=0.0, le=1.0)


class ConfidenceDelta(BaseModel):
    prior_confidence: float = Field(0.0, ge=0.0, le=1.0)
    posterior_confidence: float = Field(0.0, ge=0.0, le=1.0)
    delta: float = Field(0.0, ge=-1.0, le=1.0)
    rationale: str


class SelfCritiqueRecord(BaseModel):
    branch_id: str
    problem_id: str
    mode: CritiqueMode
    branch_rank: int = Field(ge=1)
    critique_summary: str
    findings: list[CritiqueFinding] = Field(default_factory=list)
    issue_categories: list[CritiqueIssueCategory] = Field(default_factory=list)
    correction: CritiqueCorrection = Field(default_factory=CritiqueCorrection)
    confidence_delta: ConfidenceDelta
    ready_for_aggregation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfCritiqueResult(BaseModel):
    original_branch: BranchTrace
    critiqued_branch: BranchTrace
    critique: SelfCritiqueRecord


class SelfCritiqueBatchResult(BaseModel):
    branches: list[BranchTrace]
    critiques: list[SelfCritiqueResult]
    critiqued_branch_ids: list[str]
    top_k: int = Field(ge=0)


class SelfCritiqueConfig(BaseModel):
    top_k: int = Field(default=SELF_CRITIQUE_TOP_K, ge=1)
    max_findings: int = Field(default=5, ge=1)
    max_evidence_per_finding: int = Field(default=2, ge=1)
    max_reasoning_chars: int = Field(default=3500, ge=256)
    fallback_extract_answer: bool = True
    apply_corrections_to_branch: bool = True

    enable_mid_search_targeting: bool = True
    enable_pre_aggregation_targeting: bool = True
    max_mid_search_branches: int = Field(default=2, ge=0)
    max_pre_aggregation_branches: int = Field(default=SELF_CRITIQUE_TOP_K, ge=0)
    min_repairability_for_early: float = Field(default=0.25, ge=0.0, le=1.0)
    min_value_for_early: float = Field(default=0.30, ge=0.0, le=1.0)
    uncertain_band_low: float = Field(default=0.25, ge=0.0, le=1.0)
    uncertain_band_high: float = Field(default=0.85, ge=0.0, le=1.0)
    critique_priority_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    contradiction_penalty_weight: float = Field(default=0.18, ge=0.0, le=1.0)
    obligation_penalty_weight: float = Field(default=0.18, ge=0.0, le=1.0)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_INTEGER_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
_ASSUMPTION_RE = re.compile(r"\b(assume|suppose|clearly|obvious|obviously|wlog|without loss)\b", re.IGNORECASE)


def _resolve_stage(stage: CritiqueStage | str) -> CritiqueStage:
    if isinstance(stage, CritiqueStage):
        return stage

    text = str(stage).strip()
    if not text:
        return CritiqueStage.PRE_AGGREGATION

    if text.startswith("CritiqueStage."):
        text = text.split(".", 1)[1]

    normalized = text.strip().lower()
    alias_map = {
        CritiqueStage.MID_SEARCH.value: CritiqueStage.MID_SEARCH,
        CritiqueStage.PRE_AGGREGATION.value: CritiqueStage.PRE_AGGREGATION,
        "midsearch": CritiqueStage.MID_SEARCH,
        "mid_search": CritiqueStage.MID_SEARCH,
        "mid-search": CritiqueStage.MID_SEARCH,
        "preaggregation": CritiqueStage.PRE_AGGREGATION,
        "pre_aggregation": CritiqueStage.PRE_AGGREGATION,
        "pre-aggregation": CritiqueStage.PRE_AGGREGATION,
    }
    if normalized in alias_map:
        return alias_map[normalized]

    upper_name = normalized.upper().replace("-", "_")
    if upper_name in CritiqueStage.__members__:
        return CritiqueStage[upper_name]

    raise ValueError(f"{stage!r} is not a valid CritiqueStage")


def critique_top_k_branches(
    branches: list[BranchTrace],
    *,
    model: CritiqueModelProtocol | None = None,
    config: SelfCritiqueConfig | None = None,
    stage: CritiqueStage | str = CritiqueStage.PRE_AGGREGATION,
) -> SelfCritiqueBatchResult:
    cfg = config or SelfCritiqueConfig()
    ordered = select_top_k_for_critique(branches, k=cfg.top_k, stage=stage, config=cfg)
    critiqued: dict[str, SelfCritiqueResult] = {}

    for rank, branch in enumerate(ordered, start=1):
        critiqued[branch.branch_id] = critique_branch(
            branch,
            branch_rank=rank,
            model=model,
            config=cfg,
            stage=stage,
        )

    merged_branches = [
        critiqued[branch.branch_id].critiqued_branch if branch.branch_id in critiqued else branch
        for branch in branches
    ]
    return SelfCritiqueBatchResult(
        branches=merged_branches,
        critiques=[critiqued[branch.branch_id] for branch in ordered],
        critiqued_branch_ids=[branch.branch_id for branch in ordered],
        top_k=min(len(ordered), len(branches)),
    )


def select_top_k_for_critique(
    branches: list[BranchTrace],
    *,
    k: int = SELF_CRITIQUE_TOP_K,
    stage: CritiqueStage | str = CritiqueStage.PRE_AGGREGATION,
    config: SelfCritiqueConfig | None = None,
) -> list[BranchTrace]:
    cfg = config or SelfCritiqueConfig()
    resolved_stage = _resolve_stage(stage)
    bounded_k = _bounded_top_k(resolved_stage, requested_k=k, config=cfg)
    eligible = [
        branch
        for branch in branches
        if should_critique_branch(branch, stage=resolved_stage, config=cfg)
    ]
    scored = sorted(
        eligible,
        key=lambda branch: _critique_priority_key(branch, stage=resolved_stage, config=cfg),
        reverse=True,
    )
    return scored[: max(0, bounded_k)]


def should_critique_branch(
    branch: BranchTrace,
    *,
    stage: CritiqueStage | str = CritiqueStage.PRE_AGGREGATION,
    config: SelfCritiqueConfig | None = None,
) -> bool:
    cfg = config or SelfCritiqueConfig()
    resolved_stage = _resolve_stage(stage)
    if resolved_stage is CritiqueStage.MID_SEARCH and not cfg.enable_mid_search_targeting:
        return False
    if resolved_stage is CritiqueStage.PRE_AGGREGATION and not cfg.enable_pre_aggregation_targeting:
        return False

    priority = _critique_priority_score(branch, stage=resolved_stage, config=cfg)
    if priority < cfg.critique_priority_threshold:
        return False

    if resolved_stage is CritiqueStage.MID_SEARCH:
        return (
            _value_score(branch) >= cfg.min_value_for_early
            and _repairability_score(branch) >= cfg.min_repairability_for_early
        )
    return True


def critique_branch(
    branch: BranchTrace,
    *,
    branch_rank: int = 1,
    model: CritiqueModelProtocol | None = None,
    config: SelfCritiqueConfig | None = None,
    stage: CritiqueStage | str = CritiqueStage.PRE_AGGREGATION,
) -> SelfCritiqueResult:
    cfg = config or SelfCritiqueConfig()
    resolved_stage = _resolve_stage(stage)

    if model is not None:
        record = _model_backed_critique(
            branch,
            branch_rank=branch_rank,
            model=model,
            config=cfg,
            stage=resolved_stage,
        )
        if record is None:
            record = _deterministic_fallback_critique(
                branch,
                branch_rank=branch_rank,
                config=cfg,
                stage=resolved_stage,
            )
    else:
        record = _deterministic_fallback_critique(
            branch,
            branch_rank=branch_rank,
            config=cfg,
            stage=resolved_stage,
        )

    updated_branch = _apply_critique(branch, record, config=cfg)
    return SelfCritiqueResult(
        original_branch=branch,
        critiqued_branch=updated_branch,
        critique=record,
    )


def build_self_critique_prompt(
    branch: BranchTrace,
    *,
    config: SelfCritiqueConfig | None = None,
    stage: CritiqueStage | str = CritiqueStage.PRE_AGGREGATION,
) -> str:
    cfg = config or SelfCritiqueConfig()
    resolved_stage = _resolve_stage(stage)
    reasoning = (getattr(branch, "full_reasoning", "") or "").strip()[: cfg.max_reasoning_chars]
    steps = "\n".join(
        (
            f"- Step {step.step_num}: {step.description} | "
            f"operator={step.operator_used or 'unknown'} | symbolic_valid={step.symbolic_valid}"
        )
        for step in list(getattr(branch, "steps", []) or [])[:20]
    )

    signals = _signal_snapshot(branch)
    trigger_signals = _critique_trigger_signals(branch)

    return (
        f"Critique this single reasoning branch during `{resolved_stage.value}`.\n"
        "Return strict JSON only with keys: critique_summary, findings, correction.\n"
        "Each finding must include category, severity, summary, detail, affected_steps, evidence, suggested_fix, blocks_submission.\n"
        "Allowed categories: arithmetic_error, logical_gap, missing_case, symbolic_mismatch, unsupported_assumption, "
        "completeness_gap, answer_format, branch_inconsistency, weak_evidence, failure_propagation.\n"
        "Only suggest local, bounded fixes. Prefer preserving valid prefix steps.\n"
        f"problem_id={getattr(branch, 'problem_id', '')}\n"
        f"branch_id={getattr(branch, 'branch_id', '')}\n"
        f"candidate_answer={getattr(branch, 'answer', '')}\n"
        f"symbolic_valid={bool(getattr(branch, 'symbolic_valid', False))}\n"
        f"failure_type={getattr(getattr(branch, 'failure_type', None), 'value', 'none')}\n"
        f"critique_priority={_critique_priority_score(branch, stage=resolved_stage, config=cfg):.6f}\n"
        f"suggested_reasoning_mode={_suggest_reasoning_mode(branch)}\n"
        "Typed signals:\n"
        + "\n".join(f"{key}={value}" for key, value in sorted(signals.items()))
        + "\nTrigger signals:\n"
        + "\n".join(f"{key}={value}" for key, value in sorted(trigger_signals.items()))
        + "\nStructured steps:\n"
        + f"{steps or '- none'}\n"
        + "Reasoning:\n"
        + f"{reasoning or '[empty]'}"
    )


def _model_backed_critique(
    branch: BranchTrace,
    *,
    branch_rank: int,
    model: CritiqueModelProtocol,
    config: SelfCritiqueConfig,
    stage: CritiqueStage,
) -> SelfCritiqueRecord | None:
    prompt = build_self_critique_prompt(branch, config=config, stage=stage)
    try:
        raw = model.generate(prompt)
        payload = _parse_model_payload(raw)
        findings = _coerce_findings(payload.get("findings", []), max_findings=config.max_findings)
        correction = _coerce_correction(payload.get("correction", {}))
        confidence_delta = _estimate_confidence_delta(branch, findings, correction)
        categories = sorted({finding.category for finding in findings}, key=lambda item: item.value)
        metadata = {
            "timing_stage": stage.value,
            "critique_priority": round(_critique_priority_score(branch, stage=stage, config=config), 6),
            "trigger_signals": _critique_trigger_signals(branch),
            "suggested_reasoning_mode": _suggest_reasoning_mode(branch),
            "raw_response_present": bool(raw),
            "prompt_budget_chars": config.max_reasoning_chars,
        }
        return SelfCritiqueRecord(
            branch_id=getattr(branch, "branch_id", ""),
            problem_id=getattr(branch, "problem_id", ""),
            mode=CritiqueMode.MODEL,
            branch_rank=branch_rank,
            critique_summary=str(payload.get("critique_summary", "")).strip() or _summarize_findings(findings),
            findings=findings,
            issue_categories=categories,
            correction=correction,
            confidence_delta=confidence_delta,
            ready_for_aggregation=_ready_for_aggregation(findings),
            metadata=metadata,
        )
    except Exception:
        return None


def _deterministic_fallback_critique(
    branch: BranchTrace,
    *,
    branch_rank: int,
    config: SelfCritiqueConfig,
    stage: CritiqueStage,
) -> SelfCritiqueRecord:
    findings: list[CritiqueFinding] = []
    steps = list(getattr(branch, "steps", []) or [])
    full_reasoning = (getattr(branch, "full_reasoning", "") or "").strip()

    failure_type = getattr(branch, "failure_type", None)
    if failure_type is not None:
        findings.append(_finding_from_failure(failure_type, branch, config=config))

    if not bool(getattr(branch, "symbolic_valid", False)):
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.SYMBOLIC_MISMATCH,
                severity=CritiqueSeverity.HIGH,
                summary="Symbolic validation did not hold for this branch.",
                detail=(
                    "The branch reached critique with symbolic_valid=False, or contradiction provenance was attached, "
                    "so at least one derived step is not trusted by the deterministic checker."
                ),
                affected_steps=_invalid_step_numbers(steps),
                evidence=_step_evidence(steps, invalid_only=True, limit=config.max_evidence_per_finding),
                suggested_fix="Re-evaluate only the failing suffix or local contradiction source.",
                blocks_submission=True,
            )
        )

    if _signal_value(branch, "logical_consistency") < 0.40:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.LOGICAL_GAP,
                severity=CritiqueSeverity.HIGH,
                summary="Verifier logical consistency is too low for a reliable branch.",
                detail="Typed verifier decomposition indicates the derivation chain is weak or partially inconsistent.",
                evidence=_signal_evidence(branch, "logical_consistency", "verifier.logical_consistency"),
                suggested_fix="Repair the weakest logical step instead of rewriting the whole branch.",
                blocks_submission=False,
            )
        )

    if _signal_value(branch, "completeness") < 0.35:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.COMPLETENESS_GAP,
                severity=CritiqueSeverity.MEDIUM,
                summary="Completeness evidence is weak.",
                detail="The branch appears under-justified, under-covered, or missing case closure.",
                evidence=_signal_evidence(branch, "completeness", "verifier.completeness"),
                suggested_fix="Close the missing case split or discharge remaining proof obligations.",
                blocks_submission=False,
            )
        )

    if _signal_value(branch, "open_obligation_burden") > 0.55:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.COMPLETENESS_GAP,
                severity=CritiqueSeverity.HIGH,
                summary="Open proof obligations remain unresolved.",
                detail="Proof-obligation burden is high enough that the branch should be treated as repairable-but-not-final.",
                evidence=_signal_evidence(
                    branch,
                    "open_obligation_burden",
                    "proof_state.open_obligation_burden",
                ),
                suggested_fix="Prefer obligation-closing or symbolic-heavy repair before aggregation.",
                blocks_submission=False,
            )
        )

    if _signal_value(branch, "retrieval_support") < 0.20 and bool(getattr(branch, "retrieval_used", False)):
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.WEAK_EVIDENCE,
                severity=CritiqueSeverity.MEDIUM,
                summary="Retrieved support was used but remains weakly compatible.",
                detail="The branch appears to have retrieval-conditioned reasoning without strong retrieval compatibility/support evidence.",
                evidence=_signal_evidence(
                    branch,
                    "retrieval_support",
                    "retrieval.compatibility_support",
                ),
                suggested_fix="Switch to critique-heavy or symbolic-heavy local validation rather than trusting the retrieved pattern.",
                blocks_submission=False,
            )
        )

    if _signal_value(branch, "operator_reliability") < 0.25 and steps:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.BRANCH_INCONSISTENCY,
                severity=CritiqueSeverity.MEDIUM,
                summary="Operator reliability is low for the current reasoning path.",
                detail="Historical or typed operator reliability suggests the chosen local transformation sequence is fragile.",
                evidence=_step_evidence(steps, invalid_only=False, limit=config.max_evidence_per_finding),
                suggested_fix="Substitute the least reliable operator rather than discarding the whole prefix.",
                blocks_submission=False,
            )
        )

    answer = str(getattr(branch, "answer", "") or "").strip()
    if not answer:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.ANSWER_FORMAT,
                severity=CritiqueSeverity.CRITICAL,
                summary="Final answer is missing.",
                detail="The branch has reasoning but no explicit final answer string suitable for aggregation.",
                evidence=_reasoning_evidence(full_reasoning, limit=config.max_evidence_per_finding),
                suggested_fix="Emit one explicit final integer answer in competition range.",
                blocks_submission=True,
            )
        )
    elif canonicalize_competition_answer(answer) is None:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.ANSWER_FORMAT,
                severity=CritiqueSeverity.HIGH,
                summary="Final answer is not a clean competition integer.",
                detail=f"Observed answer `{answer}` cannot be deterministically normalized into [{ANSWER_MIN}, {ANSWER_MAX}].",
                evidence=[CritiqueEvidence(source="branch.answer", excerpt=answer, confidence=0.95)],
                suggested_fix="Normalize or correct the answer to a single non-negative integer.",
                blocks_submission=True,
            )
        )

    if not full_reasoning and not steps:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.COMPLETENESS_GAP,
                severity=CritiqueSeverity.CRITICAL,
                summary="Branch has no recoverable reasoning trace.",
                detail="Neither free-form reasoning nor structured steps were available, so the branch cannot be audited or safely aggregated.",
                suggested_fix="Preserve structured branch reasoning before critique.",
                blocks_submission=True,
            )
        )
    elif len(steps) < 2 and len(full_reasoning) < 80:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.COMPLETENESS_GAP,
                severity=CritiqueSeverity.MEDIUM,
                summary="Reasoning trace is too thin for a reliable branch.",
                detail="The branch contains very little structured support, which raises the risk of missing steps or hidden assumptions.",
                evidence=_reasoning_evidence(full_reasoning, limit=config.max_evidence_per_finding),
                suggested_fix="Add at least one explicit derivation or verification step before aggregation.",
                blocks_submission=False,
            )
        )

    assumption_hits = _assumption_step_numbers(branch)
    if assumption_hits:
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory.UNSUPPORTED_ASSUMPTION,
                severity=CritiqueSeverity.MEDIUM,
                summary="Reasoning relies on unsupported assumption language.",
                detail="The branch uses assumption-style wording without showing the corresponding justification or case closure.",
                affected_steps=assumption_hits,
                evidence=_step_evidence_by_numbers(steps, assumption_hits, limit=config.max_evidence_per_finding),
                suggested_fix="Justify each assumption explicitly or convert it into a checked case split.",
                blocks_submission=False,
            )
        )

    findings = _dedupe_findings(findings)[: config.max_findings]
    correction = _fallback_correction(branch) if config.fallback_extract_answer else CritiqueCorrection()
    confidence_delta = _estimate_confidence_delta(branch, findings, correction)

    return SelfCritiqueRecord(
        branch_id=getattr(branch, "branch_id", ""),
        problem_id=getattr(branch, "problem_id", ""),
        mode=CritiqueMode.DETERMINISTIC_FALLBACK,
        branch_rank=branch_rank,
        critique_summary=_summarize_findings(findings),
        findings=findings,
        issue_categories=sorted({finding.category for finding in findings}, key=lambda item: item.value),
        correction=correction,
        confidence_delta=confidence_delta,
        ready_for_aggregation=_ready_for_aggregation(findings),
        metadata={
            "timing_stage": stage.value,
            "fallback_reason": "model_unavailable_or_unparseable",
            "critique_priority": round(_critique_priority_score(branch, stage=stage, config=config), 6),
            "trigger_signals": _critique_trigger_signals(branch),
            "suggested_reasoning_mode": _suggest_reasoning_mode(branch),
            "invalid_step_count": len(_invalid_step_numbers(steps)),
        },
    )


def _apply_critique(branch: BranchTrace, record: SelfCritiqueRecord, *, config: SelfCritiqueConfig) -> BranchTrace:
    corrected_answer = record.correction.corrected_answer if config.apply_corrections_to_branch else None
    corrected_reasoning = record.correction.corrected_reasoning if config.apply_corrections_to_branch else None
    answer = corrected_answer or getattr(branch, "answer", None)
    reasoning = corrected_reasoning or getattr(branch, "full_reasoning", None)
    answer_canonical = canonicalize_competition_answer(answer) or getattr(branch, "answer_canonical", None)

    metadata = dict(getattr(branch, "metadata", {}) or {})
    metadata.update(
        {
            "self_critique": {
                "timing_stage": record.metadata.get("timing_stage"),
                "summary": record.critique_summary,
                "issue_categories": [category.value for category in record.issue_categories],
                "ready_for_aggregation": record.ready_for_aggregation,
                "confidence_delta": record.confidence_delta.model_dump(mode="json"),
                "trigger_signals": record.metadata.get("trigger_signals", {}),
                "suggested_reasoning_mode": record.metadata.get("suggested_reasoning_mode"),
                "critique_priority": record.metadata.get("critique_priority"),
            }
        }
    )

    return branch.model_copy(
        update={
            "answer": answer,
            "answer_canonical": answer_canonical,
            "full_reasoning": reasoning,
            "self_critiqued": True,
            "branch_score": record.confidence_delta.posterior_confidence,
            "metadata": metadata,
        }
    )


def _parse_model_payload(raw: str) -> dict[str, Any]:
    match = _JSON_BLOCK_RE.search(raw.strip())
    if match is None:
        raise ValueError("No JSON object found in critique response.")
    return json.loads(match.group(0))


def _coerce_findings(items: list[Any], *, max_findings: int) -> list[CritiqueFinding]:
    findings: list[CritiqueFinding] = []
    for item in items[:max_findings]:
        evidence_items = []
        for evidence in list(item.get("evidence", []))[:2]:
            evidence_items.append(
                CritiqueEvidence(
                    source=str(evidence.get("source", "model")),
                    excerpt=str(evidence.get("excerpt", "")),
                    step_num=_maybe_int(evidence.get("step_num")),
                    operator_used=_maybe_str(evidence.get("operator_used")),
                    confidence=_clamp01(evidence.get("confidence", 0.6)),
                )
            )
        findings.append(
            CritiqueFinding(
                category=CritiqueIssueCategory(str(item.get("category", CritiqueIssueCategory.WEAK_EVIDENCE.value))),
                severity=CritiqueSeverity(str(item.get("severity", CritiqueSeverity.MEDIUM.value))),
                summary=str(item.get("summary", "")).strip(),
                detail=str(item.get("detail", "")).strip(),
                affected_steps=[
                    step
                    for step in (_maybe_int(v) for v in item.get("affected_steps", []))
                    if step is not None
                ],
                evidence=evidence_items,
                suggested_fix=_maybe_str(item.get("suggested_fix")),
                blocks_submission=bool(item.get("blocks_submission", False)),
            )
        )
    return _dedupe_findings(findings)


def _coerce_correction(payload: dict[str, Any]) -> CritiqueCorrection:
    corrected_answer = _maybe_str(payload.get("corrected_answer"))
    corrected_reasoning = _maybe_str(payload.get("corrected_reasoning"))
    applied = bool(corrected_answer or corrected_reasoning)
    return CritiqueCorrection(
        corrected_answer=corrected_answer,
        corrected_reasoning=corrected_reasoning,
        correction_applied=applied,
        correction_confidence=_clamp01(payload.get("correction_confidence", 0.65 if applied else 0.0)),
    )


def _estimate_confidence_delta(
    branch: BranchTrace,
    findings: list[CritiqueFinding],
    correction: CritiqueCorrection,
) -> ConfidenceDelta:
    prior = _clamp01(max(_value_score(branch), _signal_value(branch, "answer_correctness_likelihood")))
    severity_penalty = sum(
        {
            CritiqueSeverity.LOW: 0.03,
            CritiqueSeverity.MEDIUM: 0.08,
            CritiqueSeverity.HIGH: 0.15,
            CritiqueSeverity.CRITICAL: 0.24,
        }[finding.severity]
        for finding in findings
    )
    block_penalty = 0.07 * sum(1 for finding in findings if finding.blocks_submission)
    contradiction_penalty = 0.10 if not bool(getattr(branch, "symbolic_valid", False)) else 0.0
    correction_bonus = 0.06 * correction.correction_confidence if correction.correction_applied else 0.0
    posterior = _clamp01(prior - severity_penalty - block_penalty - contradiction_penalty + correction_bonus)
    return ConfidenceDelta(
        prior_confidence=prior,
        posterior_confidence=posterior,
        delta=round(posterior - prior, 6),
        rationale=(
            "Posterior confidence adjusts value/correctness by issue severity, blocking findings, symbolic contradiction "
            "signals, and any bounded high-confidence correction."
        ),
    )


def _finding_from_failure(
    failure_type: FailureType,
    branch: BranchTrace,
    *,
    config: SelfCritiqueConfig,
) -> CritiqueFinding:
    mapping = {
        "arithmetic_error": (
            CritiqueIssueCategory.ARITHMETIC_ERROR,
            CritiqueSeverity.HIGH,
            "Branch carries an arithmetic failure signal.",
        ),
        "symbolic_mismatch": (
            CritiqueIssueCategory.SYMBOLIC_MISMATCH,
            CritiqueSeverity.HIGH,
            "Branch carries a symbolic mismatch failure signal.",
        ),
        "logic_error": (
            CritiqueIssueCategory.LOGICAL_GAP,
            CritiqueSeverity.HIGH,
            "Branch carries a logical inconsistency failure signal.",
        ),
        "bad_case_split": (
            CritiqueIssueCategory.MISSING_CASE,
            CritiqueSeverity.HIGH,
            "Branch carries a bad case split failure signal.",
        ),
        "incomplete_proof": (
            CritiqueIssueCategory.COMPLETENESS_GAP,
            CritiqueSeverity.HIGH,
            "Branch carries an incomplete proof failure signal.",
        ),
    }
    category, severity, summary = mapping.get(
        getattr(failure_type, "value", str(failure_type)),
        (
            CritiqueIssueCategory.FAILURE_PROPAGATION,
            CritiqueSeverity.MEDIUM,
            "Branch carries a generic failure signal.",
        ),
    )
    return CritiqueFinding(
        category=category,
        severity=severity,
        summary=summary,
        detail=f"Failure type `{getattr(failure_type, 'value', str(failure_type))}` was attached before critique and should be preserved during aggregation.",
        affected_steps=_invalid_step_numbers(list(getattr(branch, "steps", []) or [])),
        evidence=_step_evidence(
            list(getattr(branch, "steps", []) or []),
            invalid_only=False,
            limit=config.max_evidence_per_finding,
        ),
        suggested_fix="Use local repair on the flagged substep instead of treating the branch as fully reliable.",
        blocks_submission=severity in {CritiqueSeverity.HIGH, CritiqueSeverity.CRITICAL},
    )


def _fallback_correction(branch: BranchTrace) -> CritiqueCorrection:
    normalized_answer = canonicalize_competition_answer(getattr(branch, "answer", None))
    branch_answer = str(getattr(branch, "answer", "") or "").strip()
    if normalized_answer is not None:
        return CritiqueCorrection(
            corrected_answer=normalized_answer if normalized_answer != branch_answer else None,
            corrected_reasoning=None,
            correction_applied=normalized_answer != branch_answer,
            correction_confidence=0.95 if normalized_answer != branch_answer else 0.0,
        )

    inferred = _extract_integer_from_reasoning(branch)
    if inferred is None:
        return CritiqueCorrection()

    return CritiqueCorrection(
        corrected_answer=inferred,
        corrected_reasoning=getattr(branch, "full_reasoning", None) or None,
        correction_applied=True,
        correction_confidence=0.42,
    )


def _extract_integer_from_reasoning(branch: BranchTrace) -> str | None:
    candidates = _INTEGER_RE.findall(getattr(branch, "full_reasoning", "") or "")
    for candidate in reversed(candidates):
        if ANSWER_MIN <= int(candidate) <= ANSWER_MAX:
            return str(int(candidate))
    return None


def _signal_value(branch: BranchTrace, key: str, default: float = 0.0) -> float:
    direct = getattr(branch, key, None)
    if isinstance(direct, (int, float)):
        return _clamp01(float(direct))

    metadata = getattr(branch, "metadata", None)
    if isinstance(metadata, dict):
        candidates: list[Any] = [
            metadata.get(key),
            metadata.get("signal_decomposition", {}).get(key) if isinstance(metadata.get("signal_decomposition"), dict) else None,
            metadata.get("verifier_decomposition", {}).get(key) if isinstance(metadata.get("verifier_decomposition"), dict) else None,
            metadata.get("retrieval", {}).get(key) if isinstance(metadata.get("retrieval"), dict) else None,
            metadata.get("operator", {}).get(key) if isinstance(metadata.get("operator"), dict) else None,
            metadata.get("proof_state", {}).get(key) if isinstance(metadata.get("proof_state"), dict) else None,
            metadata.get("self_critique", {}).get("trigger_signals", {}).get(key)
            if isinstance(metadata.get("self_critique"), dict)
            else None,
        ]
        for value in candidates:
            if isinstance(value, (int, float)):
                return _clamp01(float(value))

    alias_map = {
        "symbolic_agreement": ("tool_consistency", "symbolic_score"),
        "answer_correctness_likelihood": ("verifier_score", "branch_score"),
        "repairability": ("verifier_repairability", "repairability_score"),
        "step_quality": ("prefix_quality", "prm_prefix_quality"),
        "prefix_quality": ("prm_prefix_quality", "step_quality"),
    }
    for alias in alias_map.get(key, ()):
        if alias != key:
            value = _signal_value(branch, alias, default=-1.0)
            if value >= 0.0:
                return value
    return _clamp01(default)


def _signal_snapshot(branch: BranchTrace) -> dict[str, float]:
    keys = (
        "logical_consistency",
        "completeness",
        "repairability",
        "answer_correctness_likelihood",
        "step_quality",
        "prefix_quality",
        "prm_prefix_quality",
        "retrieval_support",
        "retrieval_compatibility",
        "operator_reliability",
        "open_obligation_burden",
        "discharge_fraction",
        "symbolic_agreement",
        "branch_score",
    )
    return {key: round(_signal_value(branch, key), 6) for key in keys}


def _contradiction_flag(branch: BranchTrace) -> float:
    if not bool(getattr(branch, "symbolic_valid", True)):
        return 1.0
    metadata = getattr(branch, "metadata", None)
    if isinstance(metadata, dict):
        proof_state = metadata.get("proof_state")
        if isinstance(proof_state, dict) and proof_state.get("contradiction_found"):
            return 1.0
        symbolic = metadata.get("symbolic")
        if isinstance(symbolic, dict) and symbolic.get("contradiction_found"):
            return 1.0
    return 0.0


def _value_score(branch: BranchTrace) -> float:
    return _clamp01(
        max(
            _signal_value(branch, "branch_score"),
            _signal_value(branch, "answer_correctness_likelihood"),
            _signal_value(branch, "verifier_score"),
        )
    )


def _repairability_score(branch: BranchTrace) -> float:
    base = _signal_value(branch, "repairability")
    return _clamp01(
        max(
            base,
            0.45 * _signal_value(branch, "step_quality")
            + 0.25 * _signal_value(branch, "retrieval_compatibility")
            + 0.20 * _signal_value(branch, "operator_reliability")
            + 0.10 * _signal_value(branch, "discharge_fraction"),
        )
    )


def _uncertainty_score(branch: BranchTrace, *, config: SelfCritiqueConfig) -> float:
    value = _value_score(branch)
    if value < config.uncertain_band_low:
        return 0.0
    if value > config.uncertain_band_high:
        return 0.0
    midpoint = (config.uncertain_band_low + config.uncertain_band_high) / 2.0
    width = max(1e-6, (config.uncertain_band_high - config.uncertain_band_low) / 2.0)
    return _clamp01(1.0 - abs(value - midpoint) / width)


def _critique_trigger_signals(branch: BranchTrace) -> dict[str, float]:
    triggers = {
        "value": _value_score(branch),
        "repairability": _repairability_score(branch),
        "uncertainty": _uncertainty_score(branch, config=SelfCritiqueConfig()),
        "logical_consistency": _signal_value(branch, "logical_consistency"),
        "completeness": _signal_value(branch, "completeness"),
        "open_obligation_burden": _signal_value(branch, "open_obligation_burden"),
        "retrieval_support": _signal_value(branch, "retrieval_support"),
        "retrieval_compatibility": _signal_value(branch, "retrieval_compatibility"),
        "operator_reliability": _signal_value(branch, "operator_reliability"),
        "symbolic_contradiction": _contradiction_flag(branch),
    }
    return {key: round(_clamp01(value), 6) for key, value in triggers.items()}


def _critique_priority_score(
    branch: BranchTrace,
    *,
    stage: CritiqueStage,
    config: SelfCritiqueConfig,
) -> float:
    value = _value_score(branch)
    repairability = _repairability_score(branch)
    uncertainty = _uncertainty_score(branch, config=config)
    completeness_gap = 1.0 - _signal_value(branch, "completeness")
    logical_gap = 1.0 - _signal_value(branch, "logical_consistency")
    retrieval_gap = 1.0 - _signal_value(branch, "retrieval_compatibility")
    contradiction = _contradiction_flag(branch)
    obligation_pressure = _signal_value(branch, "open_obligation_burden")

    base = _clamp01(
        0.22 * value
        + 0.18 * repairability
        + 0.20 * uncertainty
        + 0.14 * logical_gap
        + 0.10 * completeness_gap
        + 0.08 * retrieval_gap
        + config.contradiction_penalty_weight * contradiction
        + config.obligation_penalty_weight * obligation_pressure
    )
    if stage is CritiqueStage.MID_SEARCH:
        return _clamp01(base * 0.95)
    return base


def _critique_priority_key(
    branch: BranchTrace,
    *,
    stage: CritiqueStage,
    config: SelfCritiqueConfig,
) -> tuple[float, float, float, float, float]:
    return (
        _critique_priority_score(branch, stage=stage, config=config),
        _repairability_score(branch),
        _value_score(branch),
        _signal_value(branch, "prm_prefix_quality"),
        1.0 - _signal_value(branch, "open_obligation_burden"),
    )


def _bounded_top_k(stage: CritiqueStage, *, requested_k: int, config: SelfCritiqueConfig) -> int:
    if stage is CritiqueStage.MID_SEARCH:
        return min(requested_k, config.max_mid_search_branches)
    return min(requested_k, config.max_pre_aggregation_branches)


def _suggest_reasoning_mode(branch: BranchTrace) -> str:
    contradiction = _contradiction_flag(branch)
    obligation = _signal_value(branch, "open_obligation_burden")
    retrieval = _signal_value(branch, "retrieval_support")
    logical = _signal_value(branch, "logical_consistency")
    operator = _signal_value(branch, "operator_reliability")
    if contradiction > 0.5 or obligation > 0.6:
        return "symbolic-heavy"
    if retrieval > 0.55 and logical < 0.55:
        return "critique-heavy"
    if operator < 0.30:
        return "search-heavy"
    if retrieval < 0.20:
        return "retrieval-heavy"
    return "critique-heavy"


def _invalid_step_numbers(steps: list[ReasoningStep]) -> list[int]:
    return [step.step_num for step in steps if not bool(getattr(step, "symbolic_valid", True))]


def _assumption_step_numbers(branch: BranchTrace) -> list[int]:
    hits: list[int] = []
    for step in list(getattr(branch, "steps", []) or []):
        if _ASSUMPTION_RE.search(getattr(step, "description", "") or ""):
            hits.append(step.step_num)
    if not hits and _ASSUMPTION_RE.search(getattr(branch, "full_reasoning", "") or ""):
        steps = list(getattr(branch, "steps", []) or [])
        return [steps[0].step_num] if steps else []
    return hits


def _step_evidence(
    steps: list[ReasoningStep],
    *,
    invalid_only: bool,
    limit: int,
) -> list[CritiqueEvidence]:
    evidence: list[CritiqueEvidence] = []
    for step in steps:
        if invalid_only and bool(getattr(step, "symbolic_valid", True)):
            continue
        evidence.append(
            CritiqueEvidence(
                source="branch.steps",
                excerpt=getattr(step, "description", "") or "",
                step_num=getattr(step, "step_num", None),
                operator_used=getattr(step, "operator_used", None),
                confidence=0.85 if not bool(getattr(step, "symbolic_valid", True)) else 0.65,
            )
        )
        if len(evidence) >= limit:
            break
    return evidence


def _step_evidence_by_numbers(
    steps: list[ReasoningStep],
    step_numbers: list[int],
    *,
    limit: int,
) -> list[CritiqueEvidence]:
    allowed = set(step_numbers)
    evidence = [
        CritiqueEvidence(
            source="branch.steps",
            excerpt=getattr(step, "description", "") or "",
            step_num=getattr(step, "step_num", None),
            operator_used=getattr(step, "operator_used", None),
            confidence=0.80,
        )
        for step in steps
        if getattr(step, "step_num", None) in allowed
    ]
    return evidence[:limit]


def _reasoning_evidence(reasoning: str, *, limit: int) -> list[CritiqueEvidence]:
    text = (reasoning or "").strip()
    if not text:
        return []
    chunks = [text[i : i + 160] for i in range(0, min(len(text), 160 * limit), 160)]
    return [
        CritiqueEvidence(source="branch.full_reasoning", excerpt=chunk, confidence=0.60)
        for chunk in chunks[:limit]
    ]


def _signal_evidence(branch: BranchTrace, key: str, source_name: str) -> list[CritiqueEvidence]:
    return [
        CritiqueEvidence(
            source=source_name,
            excerpt=f"{key}={_signal_value(branch, key):.6f}",
            confidence=0.90,
        )
    ]


def _ready_for_aggregation(findings: list[CritiqueFinding]) -> bool:
    return not any(finding.blocks_submission for finding in findings)


def _summarize_findings(findings: list[CritiqueFinding]) -> str:
    if not findings:
        return "No material critique findings; branch is acceptable for bounded aggregation."
    ordered = sorted(findings, key=lambda item: _severity_rank(item.severity), reverse=True)
    top = ordered[0]
    if len(ordered) == 1:
        return top.summary
    return f"{top.summary} (+{len(ordered) - 1} additional critique findings)"


def _dedupe_findings(findings: list[CritiqueFinding]) -> list[CritiqueFinding]:
    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    deduped: list[CritiqueFinding] = []
    for finding in findings:
        key = (
            finding.category.value,
            finding.summary.strip().lower(),
            tuple(sorted(finding.affected_steps)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _severity_rank(severity: CritiqueSeverity) -> int:
    return {
        CritiqueSeverity.LOW: 0,
        CritiqueSeverity.MEDIUM: 1,
        CritiqueSeverity.HIGH: 2,
        CritiqueSeverity.CRITICAL: 3,
    }[severity]


def _maybe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "ConfidenceDelta",
    "CritiqueCorrection",
    "CritiqueEvidence",
    "CritiqueFinding",
    "CritiqueIssueCategory",
    "CritiqueMode",
    "CritiqueModelProtocol",
    "CritiqueSeverity",
    "CritiqueStage",
    "SelfCritiqueBatchResult",
    "SelfCritiqueConfig",
    "SelfCritiqueRecord",
    "SelfCritiqueResult",
    "build_self_critique_prompt",
    "critique_branch",
    "critique_top_k_branches",
    "select_top_k_for_critique",
    "should_critique_branch",
]

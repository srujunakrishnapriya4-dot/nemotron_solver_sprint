from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from src.common.schemas import BranchTrace, ReasoningStep

from .operator_types import (
    FailureMode,
    MiningEvidenceKind,
    OperatorCondition,
    OperatorMiningCandidate,
    OperatorMiningEvidence,
    OperatorMiningReport,
    OperatorNeighbor,
    OperatorPostcondition,
    PostconditionKind,
    PreconditionKind,
    ToolCapability,
    operator_family_for_name,
)


class OperatorMiner:
    """
    Structured operator miner.
    """

    def __init__(
        self,
        *,
        min_support: int = 3,
        min_symbolic_success_rate: float = 0.55,
        min_mean_verifier_gain: float = 0.02,
    ) -> None:
        self.min_support = min_support
        self.min_symbolic_success_rate = min_symbolic_success_rate
        self.min_mean_verifier_gain = min_mean_verifier_gain

        self._trace_count = 0
        self._structured_steps_seen = 0
        self._evidence_items = 0

        self._support: dict[str, int] = defaultdict(int)
        self._symbolic_successes: dict[str, int] = defaultdict(int)
        self._verifier_gains: dict[str, list[float]] = defaultdict(list)
        self._domains: dict[str, set[str]] = defaultdict(set)
        self._archetypes: dict[str, set[str]] = defaultdict(set)
        self._failure_modes: dict[str, set[FailureMode]] = defaultdict(set)
        self._neighbor_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._precondition_votes: dict[str, dict[tuple[str, str, str | None], int]] = defaultdict(lambda: defaultdict(int))
        self._postcondition_votes: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
        self._domain_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._archetype_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._evidence: dict[str, list[OperatorMiningEvidence]] = defaultdict(list)

    def ingest_trace(self, trace: BranchTrace | Mapping[str, Any] | Any) -> None:
        trace = self._coerce_trace(trace)
        self._trace_count += 1

        structured_steps = self._extract_structured_steps(trace)
        if not structured_steps and not list(getattr(trace, "operator_sequence", []) or []):
            return

        sequence = list(getattr(trace, "operator_sequence", []) or [])
        if not sequence:
            sequence = [str(step.operator_used) for step in structured_steps if getattr(step, "operator_used", None)]

        symbolic_valid = bool(getattr(trace, "symbolic_valid", False))
        verifier_score = float(getattr(trace, "verifier_score", 0.0) or 0.0)
        branch_score = float(getattr(trace, "branch_score", 0.0) or 0.0)
        answer = getattr(trace, "answer", None)
        domain = self._normalize_domain(getattr(trace, "archetype_used", None))
        archetype = getattr(trace, "archetype_used", None)

        for idx, op_name in enumerate(sequence):
            if not op_name:
                continue
            self._support[op_name] += 1
            self._verifier_gains[op_name].append(verifier_score - branch_score)
            if symbolic_valid:
                self._symbolic_successes[op_name] += 1

            if domain:
                self._domains[op_name].add(domain)
                self._domain_votes[op_name][domain] += 1
            if archetype:
                self._archetypes[op_name].add(str(archetype))
                self._archetype_votes[op_name][str(archetype)] += 1

            self._record_precondition_votes(op_name, sequence=sequence, index=idx)
            self._record_postcondition_votes(op_name, symbolic_valid=symbolic_valid, verifier_score=verifier_score)

            if idx + 1 < len(sequence):
                self._neighbor_counts[op_name][sequence[idx + 1]] += 1

            evidence = OperatorMiningEvidence(
                kind=MiningEvidenceKind.OPERATOR_SEQUENCE,
                operator_name=op_name,
                source_branch_id=str(getattr(trace, "branch_id", "")),
                source_problem_id=str(getattr(trace, "problem_id", "")),
                step_index=idx,
                symbolic_valid=symbolic_valid,
                verifier_score=verifier_score,
                answer=str(answer) if answer is not None else None,
                metadata={
                    "operator_sequence_length": len(sequence),
                    "repair_count": int(getattr(trace, "repair_count", 0) or 0),
                    "retrieval_used": bool(getattr(trace, "retrieval_used", False)),
                },
            )
            self._evidence[op_name].append(evidence)
            self._evidence_items += 1

        if getattr(trace, "failure_type", None):
            for op_name in sequence:
                self._failure_modes[op_name].add(self._map_failure(str(trace.failure_type)))

        if getattr(trace, "repaired", False) and len(sequence) >= 2:
            for src, dst in zip(sequence[:-1], sequence[1:]):
                self._neighbor_counts[src][dst] += 2

        self._structured_steps_seen += len(structured_steps)

    def emit_candidates(self) -> list[OperatorMiningCandidate]:
        candidates: list[OperatorMiningCandidate] = []
        for op_name in sorted(self._support.keys()):
            support = self._support[op_name]
            symbolic_rate = self._symbolic_success_rate(op_name)
            mean_gain = self._mean_gain(op_name)
            purity = self._operator_purity(op_name)
            reliability = self._reliability(op_name)

            candidate = OperatorMiningCandidate(
                operator_name=op_name,
                family=operator_family_for_name(op_name),
                description=self._description_for(op_name),
                compatible_domains=sorted(self._domains.get(op_name, set())),
                compatible_archetypes=sorted(self._archetypes.get(op_name, set())),
                tool_capabilities=self._tool_caps_for(op_name),
                preconditions=self._build_preconditions(op_name),
                postconditions=self._build_postconditions(op_name),
                failure_modes=sorted(self._failure_modes.get(op_name, set()), key=lambda x: x.value),
                repair_neighbors=self._build_neighbors(op_name),
                support_count=support,
                mean_verifier_gain=round(mean_gain, 6),
                symbolic_success_rate=round(symbolic_rate, 6),
                operator_purity=round(purity, 6),
                reliability=round(reliability, 6),
                promotion_safe=(
                    support >= self.min_support
                    and symbolic_rate >= self.min_symbolic_success_rate
                    and mean_gain >= self.min_mean_verifier_gain
                    and purity >= 0.50
                ),
                evidence=self._evidence.get(op_name, [])[:10],
                promotion_thresholds={
                    "min_support": self.min_support,
                    "min_symbolic_success_rate": self.min_symbolic_success_rate,
                    "min_mean_verifier_gain": self.min_mean_verifier_gain,
                    "min_operator_purity": 0.50,
                },
                provenance_summary={
                    "dominant_domains": self._top_votes(self._domain_votes.get(op_name, {}), top_k=3),
                    "dominant_archetypes": self._top_votes(self._archetype_votes.get(op_name, {}), top_k=3),
                    "evidence_kinds": sorted({item.kind.value for item in self._evidence.get(op_name, [])[:10]}),
                },
                metadata={
                    "total_neighbors": len(self._neighbor_counts.get(op_name, {})),
                    "min_support_threshold": self.min_support,
                    "reliability": round(reliability, 6),
                    "operator_purity": round(purity, 6),
                },
            )
            candidates.append(candidate)

        candidates.sort(
            key=lambda c: (
                not c.promotion_safe,
                -c.support_count,
                -c.symbolic_success_rate,
                -c.mean_verifier_gain,
                c.operator_name,
            )
        )
        return candidates

    def report(self) -> OperatorMiningReport:
        candidates = self.emit_candidates()
        return OperatorMiningReport(
            total_traces_seen=self._trace_count,
            structured_steps_seen=self._structured_steps_seen,
            evidence_items=self._evidence_items,
            candidates_emitted=len(candidates),
            candidates_promoted=sum(1 for c in candidates if c.promotion_safe),
            warnings=[],
        )

    def export_library_rows(self, candidates: list[OperatorMiningCandidate]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            rows.append(
                {
                    "operator_name": candidate.operator_name,
                    "family": candidate.family.value,
                    "description": candidate.description,
                    "compatible_domains": candidate.compatible_domains,
                    "compatible_archetypes": candidate.compatible_archetypes,
                    "tool_capabilities": [t.value for t in candidate.tool_capabilities],
                    "preconditions": [p.model_dump(mode="json") for p in candidate.preconditions],
                    "postconditions": [p.model_dump(mode="json") for p in candidate.postconditions],
                    "failure_modes": [f.value for f in candidate.failure_modes],
                    "repair_neighbors": [n.model_dump(mode="json") for n in candidate.repair_neighbors],
                    "tags": ["mined"],
                    "mined": True,
                    "metadata": {
                        **candidate.metadata,
                        "operator_purity": candidate.operator_purity,
                        "reliability": candidate.reliability,
                        "promotion_thresholds": candidate.promotion_thresholds,
                        "provenance_summary": candidate.provenance_summary,
                    },
                }
            )
        return rows

    def save_candidates(self, candidates: list[OperatorMiningCandidate], output_path: str) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.export_library_rows(candidates)
        if target.suffix == ".json":
            import json
            target.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(target, index=False)

    @staticmethod
    def _extract_structured_steps(trace: BranchTrace) -> list[ReasoningStep]:
        steps = list(getattr(trace, "steps", []) or [])
        return [step for step in steps if getattr(step, "operator_used", None)]

    @staticmethod
    def _coerce_trace(trace: BranchTrace | Mapping[str, Any] | Any) -> BranchTrace:
        if isinstance(trace, BranchTrace):
            return trace
        if isinstance(trace, Mapping):
            payload = dict(trace)
            if "steps" in payload and isinstance(payload["steps"], list):
                coerced_steps: list[ReasoningStep] = []
                for index, step in enumerate(payload["steps"], start=1):
                    if isinstance(step, ReasoningStep):
                        coerced_steps.append(step)
                        continue
                    if not isinstance(step, Mapping):
                        continue
                    coerced_steps.append(
                        ReasoningStep(
                            step_num=int(step.get("step_num", step.get("step_index", index)) or index),
                            description=str(step.get("description", step.get("text", "")) or ""),
                            operator_used=step.get("operator_used", step.get("operator_name")),
                            symbolic_expression=step.get("symbolic_expression"),
                            symbolic_valid=bool(step.get("symbolic_valid", True)),
                            python_code=step.get("python_code"),
                            python_result=step.get("python_result"),
                        )
                    )
                payload["steps"] = coerced_steps
            return BranchTrace(**payload)
        return BranchTrace(**dict(vars(trace)))

    def _record_precondition_votes(
        self,
        op_name: str,
        *,
        sequence: list[str],
        index: int,
    ) -> None:
        if index > 0:
            prev = sequence[index - 1]
            self._precondition_votes[op_name][("constraint_count", "gte", "1")] += 1
            self._precondition_votes[op_name][("previous_operator", "contains", prev)] += 1

        if "mod" in op_name or "parity" in op_name:
            self._precondition_votes[op_name][("has_integer_domain", "truthy", None)] += 2

        if op_name in {"double_counting", "bijection", "pigeonhole"}:
            self._precondition_votes[op_name][("has_counting_signal", "truthy", None)] += 2

        if op_name in {"coordinate_change", "inversion"}:
            self._precondition_votes[op_name][("has_geometry_entities", "truthy", None)] += 2

        if op_name in {"am_gm", "cauchy_schwarz", "bounding", "extremal_argument"}:
            self._precondition_votes[op_name][("has_optimization_signal", "truthy", None)] += 2

    def _record_postcondition_votes(
        self,
        op_name: str,
        *,
        symbolic_valid: bool,
        verifier_score: float,
    ) -> None:
        if symbolic_valid:
            self._postcondition_votes[op_name][(PostconditionKind.VERIFICATION_SIGNAL_ADDED.value, "symbolic_confirmation")] += 1
        if verifier_score > 0.60:
            self._postcondition_votes[op_name][(PostconditionKind.GOAL_REFINED.value, "high_verifier_support")] += 1

    def _build_preconditions(self, op_name: str) -> list[OperatorCondition]:
        votes = self._precondition_votes.get(op_name, {})
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
        out: list[OperatorCondition] = []
        for idx, ((field_name, operator, expected), count) in enumerate(ranked):
            kind = PreconditionKind.STATE
            if field_name == "has_integer_domain":
                kind = PreconditionKind.DOMAIN
            out.append(
                OperatorCondition(
                    condition_id=f"mined-pre::{op_name}::{idx}",
                    kind=kind,
                    field_name=field_name,
                    operator=operator,
                    expected_value=expected,
                    weight=max(0.5, min(2.0, 0.5 + 0.15 * count)),
                    description=f"mined_support={count}",
                )
            )
        return out

    def _build_postconditions(self, op_name: str) -> list[OperatorPostcondition]:
        votes = self._postcondition_votes.get(op_name, {})
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        out: list[OperatorPostcondition] = []
        for idx, ((kind, description), _count) in enumerate(ranked):
            out.append(
                OperatorPostcondition(
                    postcondition_id=f"mined-post::{op_name}::{idx}",
                    kind=PostconditionKind(kind),
                    description=description,
                    expected_effect=description,
                    confidence_hint=0.65,
                )
            )
        return out

    def _build_neighbors(self, op_name: str) -> list[OperatorNeighbor]:
        counts = self._neighbor_counts.get(op_name, {})
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        total = sum(v for _, v in ranked) or 1
        return [
            OperatorNeighbor(
                operator_name=name,
                reason="mined_sequence_neighbor",
                weight=round(count / total, 6),
            )
            for name, count in ranked
        ]

    @staticmethod
    def _tool_caps_for(op_name: str) -> list[ToolCapability]:
        if op_name in {"substitution", "symbolic_manipulation", "vieta", "am_gm", "cauchy_schwarz"}:
            return [ToolCapability.SYMPY]
        if op_name in {"modular_arithmetic", "parity_mod_reduction"}:
            return [ToolCapability.NUMBER_THEORY, ToolCapability.BRUTE_FORCE]
        if op_name == "brute_force_small":
            return [ToolCapability.BRUTE_FORCE, ToolCapability.SEARCH_ENUMERATOR]
        if op_name in {"coordinate_change", "inversion"}:
            return [ToolCapability.GEOMETRY]
        return [ToolCapability.NONE]

    @staticmethod
    def _description_for(op_name: str) -> str:
        return f"Mined operator candidate for {op_name.replace('_', ' ')}"

    @staticmethod
    def _normalize_domain(archetype_used: Any) -> str | None:
        if archetype_used is None:
            return None
        a = str(archetype_used).lower()
        if a in {"modular", "induction", "invariant"}:
            return "number_theory"
        if a in {"bijection", "pigeonhole"}:
            return "combinatorics"
        if a in {"coordinate_geometry"}:
            return "geometry"
        return None

    @staticmethod
    def _map_failure(raw: str) -> FailureMode:
        raw_l = raw.lower()
        if "domain" in raw_l:
            return FailureMode.DOMAIN_MISMATCH
        if "invariant" in raw_l:
            return FailureMode.INVARIANT_CONFLICT
        if "tool" in raw_l:
            return FailureMode.TOOL_UNAVAILABLE
        return FailureMode.UNKNOWN

    def _symbolic_success_rate(self, op_name: str) -> float:
        support = self._support.get(op_name, 0)
        if support <= 0:
            return 0.5
        return self._symbolic_successes.get(op_name, 0) / support

    def _mean_gain(self, op_name: str) -> float:
        values = self._verifier_gains.get(op_name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _operator_purity(self, op_name: str) -> float:
        support = self._support.get(op_name, 0)
        if support <= 0:
            return 0.0
        domain_peak = max(self._domain_votes.get(op_name, {}).values(), default=0) / support
        archetype_peak = max(self._archetype_votes.get(op_name, {}).values(), default=0) / support
        return max(0.0, min(1.0, 0.55 * domain_peak + 0.45 * archetype_peak))

    def _reliability(self, op_name: str) -> float:
        support = self._support.get(op_name, 0)
        support_term = min(1.0, support / max(self.min_support * 2, 1))
        symbolic = self._symbolic_success_rate(op_name)
        gain = max(0.0, min(1.0, 0.5 + self._mean_gain(op_name)))
        purity = self._operator_purity(op_name)
        return max(0.0, min(1.0, 0.30 * support_term + 0.30 * symbolic + 0.20 * gain + 0.20 * purity))

    @staticmethod
    def _top_votes(votes: dict[str, int], *, top_k: int) -> list[dict[str, int | str]]:
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{"value": name, "count": count} for name, count in ranked]


__all__ = ["OperatorMiner"]

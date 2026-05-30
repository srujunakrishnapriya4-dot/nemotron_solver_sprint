from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from src.common.logging import get_logger
from src.common.schemas import ParsedProblem, RetrievedTrace, RouteDecision
from src.retrieval.embedder import MathEmbedder
from src.retrieval.index_builder import RetrievalIndexBuilder
from src.retrieval.trace_prompting import build_hint_from_traces as _build_trace_hint_from_traces
from src.state_graph.node import ReasoningStateNode

if TYPE_CHECKING:
    from src.retrieval.retrieval_policy import RetrievalPolicy
    from src.retrieval.retrieval_policy import RetrievalHit


ReasoningState = ReasoningStateNode
logger = get_logger("imo3_hybrid.retrieval", module=__name__, component="query")


@dataclass(frozen=True)
class RetrievalQuery:
    problem_id: str
    query_text: str
    domain: str
    difficulty: str
    archetypes: tuple[str, ...] = ()
    operator_hints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    subproblem_snippets: tuple[str, ...] = ()
    target_goal_texts: tuple[str, ...] = ()
    open_obligation_claims: tuple[str, ...] = ()
    required_evidence_kinds: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    branch_phase: str = ""
    branch_operator_prefix: tuple[str, ...] = ()
    repair_mode: bool = False
    source: str = "problem_route_state"

    def lexical_text(self) -> str:
        parts = [
            f"domain: {self.domain}",
            f"difficulty: {self.difficulty}",
            f"branch_phase: {self.branch_phase}" if self.branch_phase else "",
            self.query_text,
            " ".join(self.archetypes),
            " ".join(self.operator_hints),
            " ".join(self.tags),
            " ".join(self.subproblem_snippets),
            " ".join(self.target_goal_texts),
            " ".join(self.open_obligation_claims),
            " ".join(self.required_evidence_kinds),
            " ".join(self.failure_modes),
            " ".join(self.branch_operator_prefix),
        ]
        return "\n".join(p for p in parts if p).strip()


def _ordered_top_keys(values: dict[str, float], limit: int) -> tuple[str, ...]:
    if not values:
        return ()
    ordered = sorted(values.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return tuple(key for key, _ in ordered[:limit] if key)


def _merge_unique(items: list[str], limit: int) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
        if len(ordered) >= limit:
            break
    return tuple(ordered)


def _trim_snippets(snippets: list[str], *, limit: int, per_snippet_chars: int) -> tuple[str, ...]:
    cleaned: list[str] = []
    for snippet in snippets:
        compact = " ".join(str(snippet or "").split()).strip()
        if not compact:
            continue
        cleaned.append(compact[:per_snippet_chars])
        if len(cleaned) >= limit:
            break
    return tuple(cleaned)


def _state_operator_hints(reasoning_state: Optional[ReasoningState]) -> tuple[str, ...]:
    if reasoning_state is None:
        return ()
    history = getattr(reasoning_state, "operator_history", ()) or ()
    names = [getattr(step, "operator_name", "") for step in history]
    return _merge_unique(list(reversed(names)), limit=4)


def _state_goal_texts(reasoning_state: Optional[ReasoningState]) -> tuple[str, ...]:
    if reasoning_state is None:
        return ()
    texts = [
        getattr(goal, "normalized_text", None) or getattr(goal, "text", None) or ""
        for goal in (getattr(reasoning_state, "goals", ()) or ())
        if getattr(goal, "status", "open") != "satisfied"
    ]
    return _trim_snippets(texts, limit=4, per_snippet_chars=140)


def _state_obligation_claims(reasoning_state: Optional[ReasoningState]) -> tuple[str, ...]:
    if reasoning_state is None:
        return ()
    claims = [
        getattr(item, "claim", "")
        for item in (getattr(reasoning_state, "proof_obligations", ()) or ())
        if getattr(item, "is_open", lambda: False)()
    ]
    return _trim_snippets(claims, limit=4, per_snippet_chars=140)


def _state_obligation_evidence_kinds(reasoning_state: Optional[ReasoningState]) -> tuple[str, ...]:
    if reasoning_state is None:
        return ()
    kinds = [
        getattr(getattr(item, "evidence_kind_required", None), "value", getattr(item, "evidence_kind_required", ""))
        for item in (getattr(reasoning_state, "proof_obligations", ()) or ())
        if getattr(item, "is_open", lambda: False)()
    ]
    return _merge_unique(list(kinds), limit=4)


def _state_failure_modes(reasoning_state: Optional[ReasoningState]) -> tuple[str, ...]:
    if reasoning_state is None:
        return ()
    metadata = dict(getattr(reasoning_state, "metadata", {}) or {})
    raw: list[str] = []
    for key in ("failure_type", "failure_mode", "latest_failure_type", "active_failure_mode"):
        value = metadata.get(key)
        if value:
            raw.append(str(getattr(value, "value", value)))
    repair_history = list(metadata.get("repair_history", []) or [])
    if repair_history:
        latest = repair_history[-1]
        if isinstance(latest, dict):
            value = latest.get("failure_type")
            if value:
                raw.append(str(value))
    return _merge_unique(raw, limit=4)


def _state_branch_phase(reasoning_state: Optional[ReasoningState]) -> str:
    if reasoning_state is None:
        return ""
    metadata = dict(getattr(reasoning_state, "metadata", {}) or {})
    return str(metadata.get("branch_phase") or metadata.get("phase") or "").strip()


def build_retrieval_query(
    problem: ParsedProblem,
    route: RouteDecision,
    *,
    reasoning_state: Optional[ReasoningState] = None,
    subproblem_text: Optional[str] = None,
    operator_sequence_hint: Optional[list[str]] = None,
    repair_mode: bool = False,
) -> RetrievalQuery:
    top_arch = _ordered_top_keys(route.archetypes, limit=4)
    route_ops = _ordered_top_keys(route.operator_prior, limit=5)
    state_ops = _state_operator_hints(reasoning_state)
    goal_texts = _state_goal_texts(reasoning_state)
    obligation_claims = _state_obligation_claims(reasoning_state)
    obligation_kinds = _state_obligation_evidence_kinds(reasoning_state)
    failure_modes = _state_failure_modes(reasoning_state)
    branch_phase = _state_branch_phase(reasoning_state)
    explicit_ops = tuple(operator_sequence_hint[:6]) if operator_sequence_hint else ()
    operator_hints = _merge_unique(list(explicit_ops) + list(state_ops) + list(route_ops), limit=6)

    tags = set(top_arch)
    tags.update(problem.likely_archetypes[:4])
    tags.update(getattr(route, "retrieval_tags", []) or [])
    tags.add(problem.domain.value)
    if problem.target:
        tags.add(problem.target.lower().replace(" ", "_"))
    if problem.parity_cues:
        tags.add("parity")
    if problem.symmetries:
        tags.add("symmetry")
    if repair_mode:
        tags.add("repair")
    for evidence_kind in obligation_kinds:
        tags.add(f"obl::{str(evidence_kind).lower()}")
    for failure_mode in failure_modes:
        tags.add(f"failure::{str(failure_mode).lower()}")

    snippets: list[str] = []

    # Put operator sequence first so it survives snippet trimming.
    if operator_sequence_hint:
        seq = [str(op).strip() for op in operator_sequence_hint if str(op).strip()]
        if seq:
            snippets.append("operator_sequence: " + " -> ".join(seq[:6]))
    elif operator_hints:
        snippets.append("operator_sequence: " + " -> ".join(operator_hints[:6]))

    snippets.extend(problem.constraint_texts()[:4])
    snippets.extend(goal_texts[:2])
    snippets.extend(obligation_claims[:2])

    if reasoning_state is not None:
        snippets.extend(reasoning_state.active_equations[:3])
        snippets.extend(reasoning_state.discovered_invariants[:2])
        if reasoning_state.partial_answer is not None:
            snippets.append(f"candidate_answer={reasoning_state.partial_answer}")

    if subproblem_text:
        snippets.append(subproblem_text)

    query_text_parts = [problem.raw_text]
    if subproblem_text:
        query_text_parts.append("SUBPROBLEM: " + " ".join(subproblem_text.split())[:320])
    if reasoning_state is not None:
        query_text_parts.append(reasoning_state.to_text_summary()[:420])
    if operator_hints:
        query_text_parts.append("OPERATOR HINTS: " + ", ".join(operator_hints[:6]))
    if goal_texts:
        query_text_parts.append("OPEN GOALS: " + " | ".join(goal_texts[:3]))
    if obligation_claims:
        query_text_parts.append("OPEN OBLIGATIONS: " + " | ".join(obligation_claims[:3]))
    if obligation_kinds:
        query_text_parts.append("REQUIRED EVIDENCE: " + ", ".join(obligation_kinds[:4]))
    if failure_modes:
        query_text_parts.append("FAILURE CONTEXT: " + ", ".join(failure_modes[:4]))

    return RetrievalQuery(
        problem_id=problem.problem_id,
        query_text="\n".join(p for p in query_text_parts if p).strip(),
        domain=problem.domain.value,
        difficulty=route.difficulty.value,
        archetypes=top_arch,
        operator_hints=operator_hints or route_ops,
        tags=tuple(sorted(tags)),
        subproblem_snippets=_trim_snippets(snippets, limit=10, per_snippet_chars=180),
        target_goal_texts=goal_texts,
        open_obligation_claims=obligation_claims,
        required_evidence_kinds=obligation_kinds,
        failure_modes=failure_modes,
        branch_phase=branch_phase,
        branch_operator_prefix=state_ops,
        repair_mode=repair_mode,
        source="repair" if repair_mode else "problem_route_state",
    ) 


class TraceRetriever:
    """
    Backward-compatible retrieval wrapper used by runtime code.
    """

    def __init__(
        self,
        embedder: MathEmbedder,
        index: RetrievalIndexBuilder,
        policy: Optional["RetrievalPolicy"] = None,
        top_k_easy: int = 3,
        top_k_hard: int = 5,
        min_similarity: float = 0.18,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self._policy_instance = policy
        self.top_k_easy = top_k_easy
        self.top_k_hard = top_k_hard
        self.min_similarity = min_similarity

    def build_query(
        self,
        problem: ParsedProblem,
        route: RouteDecision,
        reasoning_state: Optional[ReasoningState] = None,
        *,
        subproblem_text: Optional[str] = None,
        operator_sequence_hint: Optional[list[str]] = None,
        repair_mode: bool = False,
    ) -> RetrievalQuery:
        return build_retrieval_query(
            problem,
            route,
            reasoning_state=reasoning_state,
            subproblem_text=subproblem_text,
            operator_sequence_hint=operator_sequence_hint,
            repair_mode=repair_mode,
        )

    def retrieve_hits(
        self,
        query: RetrievalQuery,
        route: RouteDecision,
        *,
        top_k: Optional[int] = None,
    ) -> list["RetrievalHit"]:
        if not self.index.is_loaded or not route.use_retrieval:
            return []
        return self._policy().retrieve(
            query=query,
            route=route,
            index=self.index,
            embedder=self.embedder,
            top_k=top_k or self._resolve_top_k(route, query),
        )

    def retrieve_query(
        self,
        query: RetrievalQuery,
        route: RouteDecision,
        *,
        top_k: Optional[int] = None,
    ) -> list[RetrievedTrace]:
        hits = self.retrieve_hits(query, route, top_k=top_k)
        if not hits:
            return []
        threshold = min(self.min_similarity, 0.15) if query.repair_mode else self.min_similarity
        policy = self._policy()
        traces = [policy.hit_to_trace(hit) for hit in hits if hit.confidence >= threshold]
        if traces:
            logger.debug(
                "retrieval_query_hits",
                message=f"Retrieved {len(traces)} traces for {query.problem_id}",
                payload={
                    "problem_id": query.problem_id,
                    "trace_count": len(traces),
                    "best_confidence": round(hits[0].confidence, 6),
                    "source": query.source,
                    "repair_mode": query.repair_mode,
                },
            )
        return traces

    def retrieve(
        self,
        problem: ParsedProblem,
        route: RouteDecision,
        reasoning_state: Optional[ReasoningState] = None,
    ) -> list[RetrievedTrace]:
        query = self.build_query(problem, route, reasoning_state=reasoning_state)
        return self.retrieve_query(query, route)

    def retrieve_for_subproblem(
        self,
        problem: ParsedProblem,
        route: RouteDecision,
        subproblem_text: str,
        *,
        reasoning_state: Optional[ReasoningState] = None,
        operator_sequence_hint: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedTrace]:
        query = build_subproblem_query(
            problem,
            route,
            subproblem_text,
            reasoning_state=reasoning_state,
            operator_sequence_hint=operator_sequence_hint,
        )
        return self.retrieve_query(query, route, top_k=top_k)

    def retrieve_for_repair(
        self,
        problem: ParsedProblem,
        route: RouteDecision,
        failure_text: str,
        *,
        reasoning_state: Optional[ReasoningState] = None,
        operator_sequence_hint: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedTrace]:
        query = build_repair_query(
            problem,
            route,
            failure_text,
            reasoning_state=reasoning_state,
            operator_sequence_hint=operator_sequence_hint,
        )
        return self.retrieve_query(query, route, top_k=top_k)

    def _resolve_top_k(self, route: RouteDecision, query: RetrievalQuery) -> int:
        route_depth = max(0, int(getattr(route, "retrieval_depth", 0) or 0))
        difficulty_top_k = self.top_k_hard if route.difficulty_score >= 0.6 else self.top_k_easy
        if query.repair_mode:
            return min(max(route_depth, difficulty_top_k), 3) if route_depth > 0 else min(difficulty_top_k, 3)
        return max(route_depth, difficulty_top_k) if route_depth > 0 else difficulty_top_k

    def _policy(self) -> "RetrievalPolicy":
        if self._policy_instance is None:
            from src.retrieval.retrieval_policy import RetrievalPolicy

            self._policy_instance = RetrievalPolicy()
        return self._policy_instance


def build_subproblem_query(
    problem: ParsedProblem,
    route: RouteDecision,
    subproblem_text: str,
    *,
    reasoning_state: Optional[ReasoningState] = None,
    operator_sequence_hint: Optional[list[str]] = None,
) -> RetrievalQuery:
    return build_retrieval_query(
        problem,
        route,
        reasoning_state=reasoning_state,
        subproblem_text=subproblem_text,
        operator_sequence_hint=operator_sequence_hint,
        repair_mode=False,
    )


def build_repair_query(
    problem: ParsedProblem,
    route: RouteDecision,
    failure_text: str,
    *,
    reasoning_state: Optional[ReasoningState] = None,
    operator_sequence_hint: Optional[list[str]] = None,
) -> RetrievalQuery:
    return build_retrieval_query(
        problem,
        route,
        reasoning_state=reasoning_state,
        subproblem_text=failure_text,
        operator_sequence_hint=operator_sequence_hint,
        repair_mode=True,
    )


def build_hint_from_traces(traces: list[RetrievedTrace]) -> Optional[str]:
    return _build_trace_hint_from_traces(traces)

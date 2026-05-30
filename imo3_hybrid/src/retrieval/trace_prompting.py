from __future__ import annotations

from typing import Optional

from src.common.schemas import ParsedProblem, RetrievedTrace, RouteDecision


def _truncate(text: str, limit: int) -> str:
    clean = " ".join((text or "").strip().split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _dedupe_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        token = item.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _strategy_lines(trace: RetrievedTrace) -> list[str]:
    lines: list[str] = []
    operators = _dedupe_ordered(list(trace.operators_used))[:4]
    archetypes = _dedupe_ordered(list(trace.archetypes))[:3]
    compatibility_reasons = _dedupe_ordered(list(getattr(trace, "compatibility_reasons", []) or []))[:3]
    obligation_claims = _dedupe_ordered(list(getattr(trace, "relevant_obligation_claims", []) or []))[:2]
    evidence_kinds = _dedupe_ordered(list(getattr(trace, "relevant_evidence_kinds", []) or []))[:3]
    failure_modes = _dedupe_ordered(list(getattr(trace, "failure_mode_support", []) or []))[:2]
    repair_hints = _dedupe_ordered(list(getattr(trace, "repair_operator_hints", []) or []))[:3]
    if operators:
        lines.append(f"Likely operators: {', '.join(operators)}")
    if len(operators) >= 2:
        lines.append(f"Operator sequence hint: {' -> '.join(operators[:4])}")
    if archetypes:
        lines.append(f"Likely archetypes: {', '.join(archetypes)}")
    if compatibility_reasons:
        lines.append(f"Why retrieved: {', '.join(compatibility_reasons)}")
    if obligation_claims:
        lines.append(f"Relevant obligation: {', '.join(_truncate(item, 70) for item in obligation_claims)}")
    if evidence_kinds:
        lines.append(f"Needed evidence: {', '.join(evidence_kinds)}")
    if failure_modes:
        lines.append(f"Repair context: {', '.join(failure_modes)}")
    if repair_hints:
        lines.append(f"Repair operators: {', '.join(repair_hints)}")
    return lines


def _usable_traces(traces: list[RetrievedTrace], limit: int = 2) -> list[RetrievedTrace]:
    ranked = sorted(
        traces,
        key=lambda trace: (
            -(0.60 * float(trace.similarity_score) + 0.40 * float(getattr(trace, "compatibility_score", 0.0) or 0.0)),
            trace.trace_id,
        ),
    )
    return [trace for trace in ranked if float(trace.similarity_score) >= 0.12][:limit]


def build_hint_from_traces(traces: list[RetrievedTrace]) -> Optional[str]:
    if not traces:
        return None
    usable = _usable_traces(traces, limit=2)
    if not usable:
        usable = traces[:1]

    lines: list[str] = []
    operators = _dedupe_ordered(
        [operator for trace in usable for operator in list(trace.operators_used)[:3]]
    )[:4]
    archetypes = _dedupe_ordered(
        [archetype for trace in usable for archetype in list(trace.archetypes)[:2]]
    )[:3]
    if operators:
        lines.append(f"Likely operators: {', '.join(operators)}")
    if len(operators) >= 2:
        lines.append(f"Operator sequence hint: {' -> '.join(operators[:4])}")
    if archetypes:
        lines.append(f"Likely archetypes: {', '.join(archetypes)}")
    compatibility_reasons = _dedupe_ordered(
        [reason for trace in usable for reason in list(getattr(trace, "compatibility_reasons", []) or [])[:2]]
    )[:3]
    if compatibility_reasons:
        lines.append(f"Why retrieved: {', '.join(compatibility_reasons)}")
    repair_hints = _dedupe_ordered(
        [hint for trace in usable for hint in list(getattr(trace, "repair_operator_hints", []) or [])[:2]]
    )[:3]
    if repair_hints:
        lines.append(f"Repair operators: {', '.join(repair_hints)}")
    if lines:
        return "\n".join(lines)
    return None


def _support_block(trace: RetrievedTrace) -> str:
    lines = _strategy_lines(trace)
    if not lines:
        return ""
    compact = lines[:6]
    return "RETRIEVAL SUPPORT:\n" + "\n".join(compact) + "\n\n"


def build_trace_conditioned_prompt(
    problem: ParsedProblem,
    trace: RetrievedTrace,
    operator_hint: Optional[str] = None,
    *,
    max_example_problem_chars: int = 420,
    max_example_solution_chars: int = 720,
) -> str:
    hint_lines: list[str] = []
    if operator_hint:
        hint_lines.insert(0, f"Primary operator bias: {operator_hint}")
    hint_block = "\n".join(hint_lines)
    if hint_block:
        hint_block = f"TRANSFER HINTS:\n{hint_block}\n\n"
    support_block = _support_block(trace)

    reference_problem = _truncate(trace.problem, max_example_problem_chars)
    reference_solution = _truncate(trace.solution, max_example_solution_chars)

    return f"""Use the retrieved example only as a strategy reference.
Do not copy the derivation or the final answer.

REFERENCE PROBLEM:
{reference_problem}

REFERENCE STRATEGY SNIPPET:
{reference_solution}

{hint_block}{support_block}NEW PROBLEM:
{problem.raw_text}

Requirements:
- adapt the strategy to the new structure
- keep reasoning compact
- verify the final integer answer before committing

FINAL_ANSWER: [integer]"""


def build_hint_only_prompt(
    problem: ParsedProblem,
    traces: list[RetrievedTrace],
    *,
    failure_text: Optional[str] = None,
) -> str:
    hint_block = build_hint_from_traces(traces)
    usable = _usable_traces(traces, limit=1)
    failure_block = ""
    if failure_text:
        failure_block = f"REPAIR TARGET:\n{_truncate(failure_text, 260)}\n\n"
    if hint_block:
        hint_block = f"STRATEGY HINTS:\n{hint_block}\n\n"
    support_block = _support_block(usable[0]) if usable else ""

    return f"""Solve the olympiad problem with compact, verifiable reasoning.

{failure_block}{hint_block}{support_block}PROBLEM:
{problem.raw_text}

Requirements:
- choose operators that match the structure
- avoid copying any retrieved solution text
- end with a checked non-negative integer answer

FINAL_ANSWER: [integer]"""


def build_operator_hint_prompt(
    problem: ParsedProblem,
    traces: list[RetrievedTrace],
) -> str:
    return build_hint_only_prompt(problem, traces)


def build_repair_prompt(
    problem: ParsedProblem,
    failure_text: str,
    traces: list[RetrievedTrace],
    *,
    operator_hint: Optional[str] = None,
) -> str:
    hint_lines: list[str] = []
    if operator_hint:
        hint_lines.append(f"Primary operator bias: {operator_hint}")
    trace_hint = build_hint_from_traces(traces)
    if trace_hint:
        hint_lines.append(trace_hint)
    hint_block = "\n".join(hint_lines)
    if hint_block:
        hint_block = f"REPAIR HINTS:\n{hint_block}\n\n"
    support_block = _support_block((_usable_traces(traces, limit=1) or traces[:1])[0]) if traces else ""

    return f"""Repair the current solution locally.
Keep any verified prefix, replace only the failing idea, and avoid restarting from scratch.

FAILURE TO FIX:
{_truncate(failure_text, 320)}

{hint_block}{support_block}PROBLEM:
{problem.raw_text}

Requirements:
- preserve good prior steps when possible
- correct the failure with a smaller, safer argument
- finish with a checked non-negative integer answer

FINAL_ANSWER: [integer]"""


def select_conditioning_strategy(
    traces: list[RetrievedTrace],
    route: RouteDecision,
    branch_index: int,
    total_branches: int,
    *,
    repair_mode: bool = False,
) -> str:
    if not traces or not route.use_retrieval:
        return "none"

    usable = _usable_traces(traces, limit=1)
    if not usable:
        return "none"
    best_trace = usable[0]
    best_similarity = max(
        0.0,
        0.60 * float(best_trace.similarity_score) + 0.40 * float(getattr(best_trace, "compatibility_score", 0.0) or 0.0),
    )
    if repair_mode:
        return "hint_only" if best_similarity >= 0.12 else "none"

    branch_fraction = branch_index / max(total_branches, 1)
    if branch_fraction < 0.34 and best_similarity >= 0.55 and len(best_trace.solution or "") >= 120:
        return "full_trace"
    if branch_fraction < 0.67 and best_similarity >= 0.18:
        return "hint_only"
    return "none"


__all__ = [
    "build_hint_from_traces",
    "build_trace_conditioned_prompt",
    "build_hint_only_prompt",
    "build_operator_hint_prompt",
    "build_repair_prompt",
    "select_conditioning_strategy",
]

"""Ambiguity reporting for candidate programs."""

from __future__ import annotations

from nemotron_engine.core.schemas import AmbiguityReport, CandidateProgram


def target_outputs_agree(candidate_programs: list[CandidateProgram]) -> bool:
    outputs = _target_outputs(candidate_programs)
    return bool(outputs) and len(set(outputs)) == 1 and len(outputs) == len(candidate_programs)


def build_ambiguity_report(candidate_programs: list[CandidateProgram]) -> AmbiguityReport:
    outputs = tuple(_target_outputs(candidate_programs))
    if not candidate_programs:
        return AmbiguityReport(candidate_count=0, target_outputs=(), unique_target_output=False, ambiguous=True, reason="zero_candidates")
    if len(candidate_programs) == 1:
        unique = bool(outputs)
        return AmbiguityReport(
            candidate_count=1,
            target_outputs=outputs,
            unique_target_output=unique,
            ambiguous=not unique,
            reason="single_candidate" if unique else "missing_target_output",
        )
    unique = target_outputs_agree(candidate_programs)
    return AmbiguityReport(
        candidate_count=len(candidate_programs),
        target_outputs=outputs,
        unique_target_output=unique,
        ambiguous=not unique,
        reason="all_targets_agree" if unique else "target_outputs_disagree",
    )


def _target_outputs(candidate_programs: list[CandidateProgram]) -> list[str]:
    outputs: list[str] = []
    for candidate in candidate_programs:
        if candidate.target_execution is not None and candidate.target_execution.output_value not in {None, ""}:
            outputs.append(str(candidate.target_execution.output_value))
    return outputs


__all__ = ["build_ambiguity_report", "target_outputs_agree"]

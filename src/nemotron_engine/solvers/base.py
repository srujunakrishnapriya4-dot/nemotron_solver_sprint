"""Base solver contract for the Pass 3 small typed solver core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from nemotron_engine.core.schemas import AmbiguityReport, CanonicalProblem, ExecutableProof, Program, VerificationStatus
from nemotron_engine.programs.ambiguity import build_ambiguity_report
from nemotron_engine.programs.ranker import choose_unique_best
from nemotron_engine.programs.shadow_verifier import shadow_verify_proof
from nemotron_engine.programs.verifier import verify_program_on_problem


@dataclass(frozen=True)
class SolverAttempt:
    solver_name: str
    program: Program | None
    proof: ExecutableProof | None
    status: VerificationStatus
    reason: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.solver_name.strip():
            raise ValueError("solver_name must be non-empty.")
        if self.status is not VerificationStatus.PASS and not self.reason.strip():
            raise ValueError("Rejected solver attempts require a reason.")
        if self.status is VerificationStatus.PASS:
            if not self.reason.strip():
                raise ValueError("Accepted solver attempts require a reason.")
            if self.program is None or self.proof is None or self.proof.is_training_safe is not True:
                raise ValueError("Accepted solver attempts require a training-safe proof.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SolverResult:
    problem_id: str
    solver_name: str
    attempts: tuple[SolverAttempt, ...]
    accepted_attempts: tuple[SolverAttempt, ...]
    rejected_attempts: tuple[SolverAttempt, ...]
    ambiguity_report: AmbiguityReport
    best_attempt: SolverAttempt | None

    def __post_init__(self) -> None:
        if not self.problem_id.strip():
            raise ValueError("problem_id must be non-empty.")
        if not self.solver_name.strip():
            raise ValueError("solver_name must be non-empty.")
        attempts = tuple(self.attempts)
        accepted = tuple(self.accepted_attempts)
        rejected = tuple(self.rejected_attempts)
        attempt_ids = [id(attempt) for attempt in attempts]
        classified_ids = [id(attempt) for attempt in accepted + rejected]
        if sorted(attempt_ids) != sorted(classified_ids):
            raise ValueError("attempts must be exactly partitioned into accepted_attempts and rejected_attempts.")
        if len(set(classified_ids)) != len(classified_ids):
            raise ValueError("attempts cannot appear in both accepted_attempts and rejected_attempts.")
        for attempt in accepted:
            if attempt.status is not VerificationStatus.PASS or attempt.proof is None or attempt.proof.is_training_safe is not True:
                raise ValueError("accepted_attempts must all be PASS attempts with training-safe proofs.")
        for attempt in rejected:
            if attempt.status is VerificationStatus.PASS or not attempt.reason.strip():
                raise ValueError("rejected_attempts must all be non-PASS attempts with reasons.")
        if self.best_attempt is not None and all(id(self.best_attempt) != id(attempt) for attempt in accepted):
            raise ValueError("best_attempt must be present in accepted_attempts.")
        if self.ambiguity_report.ambiguous is True and self.best_attempt is not None:
            raise ValueError("ambiguous SolverResult cannot have best_attempt.")
        outputs = {
            str(attempt.proof.target_execution.output_value)
            for attempt in accepted
            if attempt.proof is not None
            and attempt.proof.target_execution is not None
            and attempt.proof.target_execution.output_value not in {None, ""}
        }
        if len(outputs) > 1 and self.best_attempt is not None:
            raise ValueError("SolverResult cannot have best_attempt when accepted target outputs disagree.")
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "accepted_attempts", accepted)
        object.__setattr__(self, "rejected_attempts", rejected)


def verify_candidate_program(
    problem: CanonicalProblem,
    program: Program,
    *,
    solver_name: str,
    expected_target_output: str | None = None,
) -> SolverAttempt:
    try:
        proof = verify_program_on_problem(problem, program, expected_target_output=expected_target_output)
        checked = shadow_verify_proof(proof)
    except Exception as exc:  # Fail closed around schema/executor/verifier errors.
        return SolverAttempt(
            solver_name=solver_name,
            program=program,
            proof=None,
            status=VerificationStatus.REJECTED,
            reason=f"verification_error:{exc}",
        )
    if checked.is_training_safe:
        return SolverAttempt(
            solver_name=solver_name,
            program=program,
            proof=checked,
            status=VerificationStatus.PASS,
            reason="verified",
            score=1.0,
        )
    reason = _proof_failure_reason(checked)
    return SolverAttempt(
        solver_name=solver_name,
        program=program,
        proof=checked,
        status=VerificationStatus.FAIL,
        reason=reason,
    )


def build_solver_result(problem: CanonicalProblem, solver_name: str, attempts: Sequence[SolverAttempt]) -> SolverResult:
    materialized = tuple(attempts)
    if not materialized:
        materialized = (
            SolverAttempt(
                solver_name=solver_name,
                program=None,
                proof=None,
                status=VerificationStatus.REJECTED,
                reason="no_solution",
                metadata={"no_solution": True},
            ),
        )
    accepted = tuple(attempt for attempt in materialized if attempt.status is VerificationStatus.PASS)
    rejected = tuple(attempt for attempt in materialized if attempt.status is not VerificationStatus.PASS)
    candidate_programs = [
        _candidate_from_attempt(attempt)
        for attempt in accepted
        if attempt.proof is not None and attempt.program is not None
    ]
    ambiguity = build_ambiguity_report(candidate_programs)
    best = choose_unique_best(accepted) if ambiguity.unique_target_output and not ambiguity.ambiguous else None
    return SolverResult(problem.problem_id, solver_name, materialized, accepted, rejected, ambiguity, best)


def rejected_attempt(solver_name: str, reason: str, *, metadata: dict[str, Any] | None = None) -> SolverAttempt:
    return SolverAttempt(
        solver_name=solver_name,
        program=None,
        proof=None,
        status=VerificationStatus.REJECTED,
        reason=reason,
        metadata=dict(metadata or {}),
    )


def _candidate_from_attempt(attempt: SolverAttempt):
    from nemotron_engine.core.schemas import CandidateProgram

    assert attempt.proof is not None
    assert attempt.program is not None
    return CandidateProgram(
        program=attempt.program,
        example_executions=attempt.proof.example_executions,
        target_execution=attempt.proof.target_execution,
        score=attempt.score,
    )


def _proof_failure_reason(proof: ExecutableProof) -> str:
    if proof.primary_verifier_pass is not True:
        return "primary_verifier_failed"
    if proof.shadow_verifier_pass is not True:
        return "shadow_verifier_failed"
    if proof.leakage_report.has_leakage:
        return "leakage_detected"
    if proof.ambiguity_report.ambiguous:
        return "ambiguous_candidate"
    if proof.format_report.valid is not True:
        return "invalid_format"
    return "proof_not_training_safe"


__all__ = [
    "SolverAttempt",
    "SolverResult",
    "build_solver_result",
    "rejected_attempt",
    "verify_candidate_program",
]

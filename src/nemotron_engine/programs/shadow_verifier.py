"""Independent shadow verifier for stored executable proofs."""

from __future__ import annotations

from dataclasses import replace

from nemotron_engine.core.schemas import ExecutableProof


def shadow_verify_proof(proof: ExecutableProof) -> ExecutableProof:
    target = proof.target_execution
    shadow_pass = (
        proof.primary_verifier_pass is True
        and bool(proof.example_executions)
        and all(trace.passed is True and trace.error is None for trace in proof.example_executions)
        and target is not None
        and target.output_value not in {None, ""}
        and target.error is None
        and proof.program_hash == proof.locked_program.program_hash
        and proof.ambiguity_report.candidate_count > 0
        and proof.ambiguity_report.unique_target_output is True
        and proof.ambiguity_report.ambiguous is False
        and proof.leakage_report.has_leakage is False
        and proof.format_report.valid is True
    )
    return replace(proof, shadow_verifier_pass=shadow_pass)


__all__ = ["shadow_verify_proof"]

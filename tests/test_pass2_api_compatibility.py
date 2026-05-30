from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_pass2_direct_imports() -> None:
    from nemotron_engine.core.manifest import create_manifest, read_manifest, stable_hash, validate_manifest_chain, write_manifest
    from nemotron_engine.core.registry import quarantine_problem, register_problem, split_allows_eval, split_allows_sft, validate_registry
    from nemotron_engine.core.schemas import (
        AmbiguityReport,
        CanonicalProblem,
        DomainSignature,
        ExamplePair,
        ExecutableProof,
        FormatReport,
        ImmutableProblemRecord,
        LeakageReport,
        Program,
        ProgramStep,
        SymbolTable,
        TargetQuery,
    )
    from nemotron_engine.parsing import build_symbol_table, canonicalize_prompt, infer_domain, reconstruct_prompt_skeleton, verify_round_trip
    from nemotron_engine.programs import (
        build_ambiguity_report,
        detect_target_leakage,
        execute_program,
        shadow_verify_proof,
        verify_program_on_problem,
    )

    for item in (
        ImmutableProblemRecord,
        CanonicalProblem,
        ExamplePair,
        TargetQuery,
        DomainSignature,
        SymbolTable,
        Program,
        ProgramStep,
        ExecutableProof,
        AmbiguityReport,
        LeakageReport,
        FormatReport,
        create_manifest,
        write_manifest,
        read_manifest,
        validate_manifest_chain,
        stable_hash,
        register_problem,
        validate_registry,
        quarantine_problem,
        split_allows_sft,
        split_allows_eval,
        canonicalize_prompt,
        infer_domain,
        build_symbol_table,
        reconstruct_prompt_skeleton,
        verify_round_trip,
        execute_program,
        verify_program_on_problem,
        shadow_verify_proof,
        build_ambiguity_report,
        detect_target_leakage,
    ):
        assert item is not None


def test_pass2_package_level_imports() -> None:
    from nemotron_engine.core import ImmutableProblemRecord, create_manifest, register_problem
    from nemotron_engine.parsing import canonicalize_prompt, verify_round_trip
    from nemotron_engine.programs import execute_program, shadow_verify_proof, verify_program_on_problem

    assert ImmutableProblemRecord is not None
    assert create_manifest is not None
    assert register_problem is not None
    assert canonicalize_prompt is not None
    assert verify_round_trip is not None
    assert execute_program is not None
    assert verify_program_on_problem is not None
    assert shadow_verify_proof is not None

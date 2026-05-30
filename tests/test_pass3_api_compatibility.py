from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_pass3_direct_and_package_imports() -> None:
    from nemotron_engine.programs.primitives import make_program
    from nemotron_engine.programs.ranker import choose_unique_best, rank_candidate_programs
    from nemotron_engine.programs.type_system import infer_value_domain, require_domain
    from nemotron_engine.solvers import solve_binary, solve_cipher_digit, solve_numeric, solve_symbol_digit, solve_symbolic
    from nemotron_engine.solvers.universal_synthesizer import synthesize

    for item in (
        solve_binary,
        solve_numeric,
        solve_symbol_digit,
        solve_cipher_digit,
        solve_symbolic,
        synthesize,
        rank_candidate_programs,
        choose_unique_best,
        infer_value_domain,
        require_domain,
        make_program,
    ):
        assert item is not None


def test_pass3_programs_package_exports() -> None:
    from nemotron_engine.programs import (
        build_ambiguity_report,
        choose_unique_best,
        execute_program,
        infer_value_domain,
        make_program,
        rank_candidate_programs,
        require_domain,
        shadow_verify_proof,
        validate_program,
        verify_program_on_problem,
    )

    assert build_ambiguity_report is not None
    assert choose_unique_best is not None
    assert execute_program is not None
    assert infer_value_domain is not None
    assert make_program is not None
    assert rank_candidate_programs is not None
    assert require_domain is not None
    assert shadow_verify_proof is not None
    assert validate_program is not None
    assert verify_program_on_problem is not None

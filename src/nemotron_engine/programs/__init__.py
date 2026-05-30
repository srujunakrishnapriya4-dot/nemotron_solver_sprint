"""Public program DSL, execution, and verification APIs."""

from .ambiguity import build_ambiguity_report, target_outputs_agree
from .dsl import ALLOWED_PRIMITIVES, Program, ProgramStep, ProgramStepKind, validate_program
from .executor import execute_program
from .leakage import detect_target_leakage
from .primitives import make_format_step, make_parse_step, make_program, make_select_identity_step, make_transform_step
from .ranker import ProgramRank, choose_unique_best, rank_candidate_programs
from .shadow_verifier import shadow_verify_proof
from .type_system import infer_value_domain, require_domain
from .verifier import verify_program_on_problem

__all__ = [
    "ALLOWED_PRIMITIVES",
    "Program",
    "ProgramStep",
    "ProgramStepKind",
    "ProgramRank",
    "build_ambiguity_report",
    "choose_unique_best",
    "detect_target_leakage",
    "execute_program",
    "infer_value_domain",
    "make_format_step",
    "make_parse_step",
    "make_program",
    "make_select_identity_step",
    "make_transform_step",
    "rank_candidate_programs",
    "require_domain",
    "shadow_verify_proof",
    "target_outputs_agree",
    "validate_program",
    "verify_program_on_problem",
]

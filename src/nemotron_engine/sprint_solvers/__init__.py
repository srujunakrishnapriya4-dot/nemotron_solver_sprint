"""Sprint 1 symbolic solver skeleton APIs."""

from .arithmetic_solver import enumerate_arithmetic_candidates, solve_arithmetic_problem
from .bit_solver import enumerate_bit_candidates, solve_bit_problem
from .dsl_synthesizer import enumerate_dsl_candidates, solve_dsl_problem
from .fallback_policy import FallbackDecision, decide_model_fallback
from .family_router import DSL_SOLVER, route_family
from .failure_mining import FailureMiningError, FailureMiningReport, mine_failures, validate_failure_mining_report
from .mapping_solver import enumerate_mapping_candidates, solve_mapping_problem
from .modular_solver import enumerate_modular_candidates, solve_modular_problem
from .prediction_runner import (
    AbstentionRecord,
    DisagreementRecord,
    PredictionRecord,
    PredictionRunReport,
    PredictionRunnerError,
    SolverAttemptRecord,
    run_predictions,
    validate_prediction_run_report,
)
from .problem_parser import FORBIDDEN_INFERENCE_FIELDS, forbidden_metadata_keys, parse_problem
from .sequence_solver import enumerate_sequence_candidates, solve_sequence_problem
from .solver_base import (
    ExamplePair,
    ParsedProblem,
    RouterDecision,
    SPRINT_STATUSES,
    SolverCandidate,
    SolverResult,
    SprintSolverError,
    choose_unique_verified_prediction,
    verify_candidate_on_examples,
)
from .string_solver import enumerate_string_candidates, solve_string_problem

__all__ = [
    "DSL_SOLVER",
    "FORBIDDEN_INFERENCE_FIELDS",
    "FallbackDecision",
    "FailureMiningError",
    "FailureMiningReport",
    "ExamplePair",
    "ParsedProblem",
    "AbstentionRecord",
    "DisagreementRecord",
    "PredictionRecord",
    "PredictionRunReport",
    "PredictionRunnerError",
    "RouterDecision",
    "SPRINT_STATUSES",
    "SolverCandidate",
    "SolverAttemptRecord",
    "SolverResult",
    "SprintSolverError",
    "choose_unique_verified_prediction",
    "decide_model_fallback",
    "enumerate_arithmetic_candidates",
    "enumerate_bit_candidates",
    "enumerate_dsl_candidates",
    "enumerate_mapping_candidates",
    "enumerate_modular_candidates",
    "enumerate_sequence_candidates",
    "enumerate_string_candidates",
    "forbidden_metadata_keys",
    "mine_failures",
    "parse_problem",
    "route_family",
    "run_predictions",
    "solve_arithmetic_problem",
    "solve_bit_problem",
    "solve_dsl_problem",
    "solve_mapping_problem",
    "solve_modular_problem",
    "solve_sequence_problem",
    "solve_string_problem",
    "validate_failure_mining_report",
    "validate_prediction_run_report",
    "verify_candidate_on_examples",
]

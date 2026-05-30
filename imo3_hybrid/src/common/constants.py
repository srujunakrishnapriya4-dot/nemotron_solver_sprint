"""Global constants for AIMO3 hybrid pipeline."""
from __future__ import annotations

from types import SimpleNamespace

COMPETITION_NAME = "aimo3"
NUM_PROBLEMS = 110
ANSWER_MIN = 0
ANSWER_MAX = 99999
TIME_LIMIT_PER_PROBLEM_SEC = 300

OFFLINE_ARTIFACT_VERSION = "v1"
OFFLINE_SPLIT_RATIOS = (0.80, 0.10, 0.10)
OFFLINE_SPLIT_SALT = "imo3_offline_split_v1"

W_VERIFIER = 0.45
W_TOOL_CONSISTENCY = 0.25
W_ANSWER_AGREEMENT = 0.15
W_BRANCH_NOVELTY = 0.10
W_SYMBOLIC_CHECK = 0.05

DEFAULT_BRANCH_BUDGETS = {
    "easy": 8,
    "medium": 16,
    "hard": 48,
    "very_hard": 96,
}

DEFAULT_RETRIEVAL_DEPTHS = {
    "easy": 2,
    "medium": 3,
    "hard": 4,
    "very_hard": 5,
}

DEFAULT_MAX_SEARCH_DEPTHS = {
    "easy": 4,
    "medium": 6,
    "hard": 7,
    "very_hard": 8,
}

DEFAULT_MAX_SEARCH_NODES = {
    "easy": 24,
    "medium": 64,
    "hard": 128,
    "very_hard": 192,
}

ROUTING_HIGH_ENTROPY_THRESHOLD = 0.72
ROUTING_MIXED_DOMAIN_MARGIN = 0.12
ROUTING_SHARPEN_TEMPERATURE = 0.80
ROUTING_FLATTEN_TEMPERATURE = 1.25
ROUTING_LOW_EVIDENCE_THRESHOLD = 1.20
ROUTING_ARCHETYPE_TEXT_BACKOFF_WEIGHT = 0.18
ROUTING_DOMAIN_TEXT_BACKOFF_WEIGHT = 0.20

SELF_CRITIQUE_TOP_K = 5
MAX_REPAIR_ATTEMPTS = 3
VERIFIER_PASS_THRESHOLD = 0.55
RETRIEVAL_TOP_K_EASY = 3
RETRIEVAL_TOP_K_HARD = 5

DEFAULT_OPERATOR_PRIOR = {
    "algebraic_manipulation": 0.12,
    "equation_solving": 0.08,
    "modular_reasoning": 0.08,
    "divisibility_reasoning": 0.07,
    "invariant_reasoning": 0.07,
    "symmetry_reduction": 0.06,
    "extremal_reasoning": 0.06,
    "counting_reasoning": 0.06,
    "bijection_reasoning": 0.04,
    "case_split": 0.05,
    "construction_reasoning": 0.05,
    "induction_reasoning": 0.04,
    "bruteforce_smallspace": 0.05,
    "symbolic_execution": 0.09,
    "geometry_transform": 0.04,
    "bounding_reasoning": 0.07,
    "contradiction_reasoning": 0.07,
}

DOMAIN_OPERATOR_PRIORS = {
    "algebra": {
        "algebraic_manipulation": 0.22,
        "equation_solving": 0.16,
        "bounding_reasoning": 0.10,
        "symbolic_execution": 0.10,
        "symmetry_reduction": 0.06,
    },
    "number_theory": {
        "modular_reasoning": 0.22,
        "divisibility_reasoning": 0.18,
        "invariant_reasoning": 0.08,
        "contradiction_reasoning": 0.08,
        "symbolic_execution": 0.08,
    },
    "combinatorics": {
        "counting_reasoning": 0.22,
        "bijection_reasoning": 0.14,
        "case_split": 0.12,
        "extremal_reasoning": 0.10,
    },
    "geometry": {
        "geometry_transform": 0.24,
        "construction_reasoning": 0.14,
        "symmetry_reduction": 0.10,
        "bounding_reasoning": 0.06,
        "symbolic_execution": 0.06,
    },
    "functional_equation": {
        "symbolic_execution": 0.16,
        "equation_solving": 0.12,
        "induction_reasoning": 0.08,
        "contradiction_reasoning": 0.06,
    },
    "mixed": {
        "symbolic_execution": 0.14,
        "case_split": 0.10,
        "invariant_reasoning": 0.08,
        "contradiction_reasoning": 0.08,
    },
}

ARCHETYPE_OPERATOR_PRIORS = {
    "parity": {"modular_reasoning": 0.18, "contradiction_reasoning": 0.10},
    "invariant": {"invariant_reasoning": 0.28, "contradiction_reasoning": 0.08},
    "extremal": {"extremal_reasoning": 0.26, "bounding_reasoning": 0.08},
    "symmetry": {"symmetry_reduction": 0.28, "algebraic_manipulation": 0.06},
    "pigeonhole": {"counting_reasoning": 0.22, "case_split": 0.08},
    "modular": {"modular_reasoning": 0.26, "divisibility_reasoning": 0.12},
    "generating_function": {"counting_reasoning": 0.18, "symbolic_execution": 0.10},
    "contradiction": {"contradiction_reasoning": 0.24, "case_split": 0.08},
    "construction": {"construction_reasoning": 0.24, "geometry_transform": 0.08},
    "induction": {"induction_reasoning": 0.24, "symbolic_execution": 0.08},
    "bijection": {"bijection_reasoning": 0.24, "counting_reasoning": 0.08},
    "case_work": {"case_split": 0.24, "contradiction_reasoning": 0.06},
    "brute_force_smallspace": {"bruteforce_smallspace": 0.26, "symbolic_execution": 0.06},
    "bounding": {"bounding_reasoning": 0.24, "algebraic_manipulation": 0.08},
    "symbolic_manipulation": {"algebraic_manipulation": 0.24, "symbolic_execution": 0.12},
}

PARSING = SimpleNamespace(
    DEFAULT_SCOPE_LABEL="root",
    DOMAIN_CANONICAL_ALIASES={
        "integer": "integers",
        "integers": "integers",
        "positive_integer": "positive_integers",
        "positive_integers": "positive_integers",
        "nonnegative_integer": "nonnegative_integers",
        "non_negative_integer": "nonnegative_integers",
        "nonnegative_integers": "nonnegative_integers",
        "non_negative_integers": "nonnegative_integers",
        "natural_number": "natural_numbers",
        "natural_numbers": "natural_numbers",
        "real": "real_numbers",
        "real_numbers": "real_numbers",
        "rational": "rational_numbers",
        "rational_numbers": "rational_numbers",
        "prime": "primes",
        "primes": "primes",
    },
    INDEXED_SYMBOL_PATTERN=r"^(?P<base>[A-Za-z][A-Za-z0-9]*)_\((?P<index>[^()]+)\)$",
    SIMPLE_SYMBOL_PATTERN=r"[A-Za-z][A-Za-z0-9]*",
    NORMALIZATION_MAX_PASSES=8,
    SENTENCE_SPLIT_PATTERN=r"[.?!;]",
    MAX_INPUT_CHARS=20000,
)

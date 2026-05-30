"""Pass 4 split-safe data generation and contamination control APIs."""

from .contamination import (
    ContaminationError,
    ContaminationReport,
    compute_answer_hash,
    compute_family_signature_hash,
    compute_normalized_prompt_hash,
    compute_prompt_hash,
    detect_cross_split_contamination,
    mark_contaminated_records,
)
from .forbidden_primitive_generator import ForbiddenPrimitiveGenerationError, generate_forbidden_primitive_records
from .llm_family_generator import LLMFamilyGenerationError, generate_llm_stress_records
from .mutation_generator import MutationGenerationError, mutate_record
from .split_builder import (
    SplitManifest,
    SplitManifestError,
    assign_splits,
    read_split_manifest,
    split_records_by_family,
    validate_split_manifest,
    write_split_manifest,
)
from .synthetic_dsl_generator import SyntheticGenerationError, generate_synthetic_dsl_records

__all__ = [
    "ContaminationError",
    "ContaminationReport",
    "ForbiddenPrimitiveGenerationError",
    "LLMFamilyGenerationError",
    "MutationGenerationError",
    "SplitManifest",
    "SplitManifestError",
    "SyntheticGenerationError",
    "assign_splits",
    "compute_answer_hash",
    "compute_family_signature_hash",
    "compute_normalized_prompt_hash",
    "compute_prompt_hash",
    "detect_cross_split_contamination",
    "generate_forbidden_primitive_records",
    "generate_llm_stress_records",
    "generate_synthetic_dsl_records",
    "mark_contaminated_records",
    "mutate_record",
    "read_split_manifest",
    "split_records_by_family",
    "validate_split_manifest",
    "write_split_manifest",
]

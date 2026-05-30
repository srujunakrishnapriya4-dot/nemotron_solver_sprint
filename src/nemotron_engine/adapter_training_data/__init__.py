"""Curated SFT dataset builders for adapter training."""

from .dataset_builder import AdapterDatasetBuildReport, AdapterTrainingDataError, build_adapter_training_datasets
from .family_labeler import FamilyLabel, label_competition_family
from .hard_example_miner import HardMiningConfig, mine_hard_examples, oversample_hard_examples
from .sft_formatter import SFTExample, format_direct_answer_example, format_family_tagged_example
from .solver_distillation import SolverDistillationRecord, distill_solver_predictions

__all__ = [
    "AdapterDatasetBuildReport",
    "AdapterTrainingDataError",
    "FamilyLabel",
    "HardMiningConfig",
    "SFTExample",
    "SolverDistillationRecord",
    "build_adapter_training_datasets",
    "distill_solver_predictions",
    "format_direct_answer_example",
    "format_family_tagged_example",
    "label_competition_family",
    "mine_hard_examples",
    "oversample_hard_examples",
]

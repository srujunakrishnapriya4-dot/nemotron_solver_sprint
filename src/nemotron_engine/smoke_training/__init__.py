"""Pass 12 tiny smoke-training execution boundary APIs."""

from .smoke_backend import (
    SmokeBackendError,
    SmokeBackendProtocol,
    SmokeBackendResult,
    normalize_smoke_backend_result,
    validate_smoke_backend_result,
)
from .smoke_config import SmokeTrainingConfig, SmokeTrainingConfigError, compute_smoke_config_hash, validate_smoke_training_config
from .smoke_dataset import (
    SmokeDataset,
    SmokeDatasetError,
    SmokeDatasetExample,
    build_tiny_dpo_smoke_dataset,
    build_tiny_sft_smoke_dataset,
    compute_smoke_dataset_hash,
    validate_smoke_dataset,
)
from .smoke_handoff import SmokeHandoffError, SmokeHandoffReport, build_smoke_handoff_to_pass11, validate_smoke_handoff_report
from .smoke_report import (
    SmokeTrainingReport,
    SmokeTrainingReportError,
    build_smoke_training_report,
    compute_smoke_training_report_hash,
    validate_smoke_training_report,
)
from .smoke_runner import SmokeRunConfig, SmokeRunReport, SmokeRunnerError, run_smoke_training, validate_smoke_run_report

__all__ = [
    "SmokeBackendError",
    "SmokeBackendProtocol",
    "SmokeBackendResult",
    "SmokeDataset",
    "SmokeDatasetError",
    "SmokeDatasetExample",
    "SmokeHandoffError",
    "SmokeHandoffReport",
    "SmokeRunConfig",
    "SmokeRunReport",
    "SmokeRunnerError",
    "SmokeTrainingConfig",
    "SmokeTrainingConfigError",
    "SmokeTrainingReport",
    "SmokeTrainingReportError",
    "build_smoke_handoff_to_pass11",
    "build_smoke_training_report",
    "build_tiny_dpo_smoke_dataset",
    "build_tiny_sft_smoke_dataset",
    "compute_smoke_config_hash",
    "compute_smoke_dataset_hash",
    "compute_smoke_training_report_hash",
    "normalize_smoke_backend_result",
    "run_smoke_training",
    "validate_smoke_backend_result",
    "validate_smoke_dataset",
    "validate_smoke_handoff_report",
    "validate_smoke_run_report",
    "validate_smoke_training_config",
    "validate_smoke_training_report",
]

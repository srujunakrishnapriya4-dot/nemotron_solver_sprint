"""Pass 14 real adapter package dry-run and final submission rehearsal APIs."""

from .adapter_evidence import (
    AdapterEvidenceBundle,
    AdapterEvidenceError,
    build_adapter_evidence_bundle,
    compute_adapter_evidence_hash,
    validate_adapter_evidence_bundle,
)
from .package_rehearsal import (
    PackageRehearsalConfig,
    PackageRehearsalError,
    PackageRehearsalReport,
    run_package_rehearsal,
    validate_package_rehearsal_report,
)
from .promotion_bridge import (
    PromotionBridgeConfig,
    PromotionBridgeError,
    PromotionBridgeReport,
    build_promotion_rehearsal_report,
    validate_promotion_rehearsal_report,
)
from .rehearsal_config import (
    RehearsalConfig,
    RehearsalConfigError,
    compute_rehearsal_config_hash,
    validate_rehearsal_config,
)
from .rehearsal_manifest import (
    FinalRehearsalManifest,
    FinalRehearsalManifestError,
    build_final_rehearsal_manifest,
    compute_final_rehearsal_manifest_hash,
    validate_final_rehearsal_manifest,
)
from .runtime_rehearsal import (
    RuntimeRehearsalConfig,
    RuntimeRehearsalError,
    RuntimeRehearsalReport,
    run_runtime_rehearsal,
    validate_runtime_rehearsal_report,
)
from .submission_rehearsal import (
    SubmissionRehearsalConfig,
    SubmissionRehearsalError,
    SubmissionRehearsalReport,
    run_submission_rehearsal,
    validate_submission_rehearsal_report,
)

__all__ = [
    "AdapterEvidenceBundle",
    "AdapterEvidenceError",
    "FinalRehearsalManifest",
    "FinalRehearsalManifestError",
    "PackageRehearsalConfig",
    "PackageRehearsalError",
    "PackageRehearsalReport",
    "PromotionBridgeConfig",
    "PromotionBridgeError",
    "PromotionBridgeReport",
    "RehearsalConfig",
    "RehearsalConfigError",
    "RuntimeRehearsalConfig",
    "RuntimeRehearsalError",
    "RuntimeRehearsalReport",
    "SubmissionRehearsalConfig",
    "SubmissionRehearsalError",
    "SubmissionRehearsalReport",
    "build_adapter_evidence_bundle",
    "build_final_rehearsal_manifest",
    "build_promotion_rehearsal_report",
    "compute_adapter_evidence_hash",
    "compute_final_rehearsal_manifest_hash",
    "compute_rehearsal_config_hash",
    "run_package_rehearsal",
    "run_runtime_rehearsal",
    "run_submission_rehearsal",
    "validate_adapter_evidence_bundle",
    "validate_final_rehearsal_manifest",
    "validate_package_rehearsal_report",
    "validate_promotion_rehearsal_report",
    "validate_rehearsal_config",
    "validate_runtime_rehearsal_report",
    "validate_submission_rehearsal_report",
]

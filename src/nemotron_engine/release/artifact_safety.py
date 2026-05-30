"""Release-tree artifact safety scanning for final candidates."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import fnmatch
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from nemotron_engine.core.schemas import stable_hash


class ArtifactSafetyError(ValueError):
    """Raised when release artifact safety reports are inconsistent."""


FORBIDDEN_FILE_PATTERNS = (
    "*.bin",
    "*.safetensors",
    "pytorch_model*",
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state*.pth",
    "*.pyc",
    ".env",
    "kaggle.json",
    "credentials*",
    "token*",
    "api_key*",
)
DATASET_PATTERNS = ("sft*.jsonl", "dpo*.jsonl", "train*.jsonl")
CACHE_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".ipynb_checkpoints"})
CRITICAL_SOURCE_DIRS = frozenset(
    {
        "scoring",
        "core",
        "parsing",
        "programs",
        "solvers",
        "data",
        "traces",
        "training",
        "evaluation",
        "packaging",
        "release",
        "runtime",
        "submission",
    }
)


@dataclass(frozen=True)
class ArtifactSafetyFinding:
    path: str
    artifact_type: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("path", "artifact_type", "reason"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ArtifactSafetyError(f"finding {name} must be non-empty.")


@dataclass(frozen=True)
class ArtifactSafetyConfig:
    root_path: str | Path
    ignore_paths: tuple[str | Path, ...] = ()
    allow_submission_zip: bool = False
    allow_dataset_exports: bool = False
    allow_notebooks: bool = False
    allow_symlinks: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_path", str(self.root_path))
        object.__setattr__(self, "ignore_paths", tuple(str(item) for item in self.ignore_paths))
        if not all(
            isinstance(value, bool)
            for value in (self.allow_submission_zip, self.allow_dataset_exports, self.allow_notebooks, self.allow_symlinks)
        ):
            raise ArtifactSafetyError("artifact safety allow flags must be booleans.")
        _reject_critical_ignores(Path(self.root_path), self.ignore_paths)
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise ArtifactSafetyError("config_hash does not match artifact safety config payload.")


@dataclass(frozen=True)
class ArtifactSafetyReport:
    passed: bool
    scanned_paths: tuple[str, ...]
    forbidden_artifacts: tuple[ArtifactSafetyFinding, ...] = ()
    symlink_paths: tuple[str, ...] = ()
    path_errors: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        scanned = tuple(str(item) for item in self.scanned_paths)
        forbidden = tuple(sorted(self.forbidden_artifacts, key=lambda item: (item.path, item.artifact_type, item.reason)))
        symlinks = tuple(str(item) for item in self.symlink_paths)
        path_errors = tuple(str(item) for item in self.path_errors)
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and (forbidden or symlinks or path_errors or errors):
            raise ArtifactSafetyError("passed=True cannot include forbidden artifacts, symlinks, path errors, or errors.")
        object.__setattr__(self, "scanned_paths", scanned)
        object.__setattr__(self, "forbidden_artifacts", forbidden)
        object.__setattr__(self, "symlink_paths", symlinks)
        object.__setattr__(self, "path_errors", path_errors)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise ArtifactSafetyError("report_hash does not match artifact safety report payload.")


def scan_release_artifacts(config: ArtifactSafetyConfig) -> ArtifactSafetyReport:
    if not isinstance(config, ArtifactSafetyConfig):
        raise ArtifactSafetyError("config must be an ArtifactSafetyConfig.")
    root = Path(config.root_path)
    if not root.exists() or not root.is_dir():
        return ArtifactSafetyReport(
            passed=False,
            scanned_paths=(),
            errors=(f"release root missing or not a directory: {root}",),
        )
    ignore_roots = tuple(_resolve_ignore(root, item) for item in config.ignore_paths)
    paths = tuple(path for path in _iter_release_paths(root) if not _is_ignored(path, ignore_roots))
    scanned = tuple(str(_relative(path, root)) for path in paths)
    forbidden = detect_release_artifacts(
        paths,
        root_path=root,
        allow_submission_zip=config.allow_submission_zip,
        allow_dataset_exports=config.allow_dataset_exports,
        allow_notebooks=config.allow_notebooks,
    )
    symlinks = tuple(str(_relative(path, root)) for path in paths if (path.is_symlink() or _is_reparse_point(path)) and not config.allow_symlinks)
    passed = not forbidden and not symlinks
    return ArtifactSafetyReport(
        passed=passed,
        scanned_paths=scanned,
        forbidden_artifacts=forbidden,
        symlink_paths=tuple(sorted(symlinks)),
        path_errors=(),
        errors=(),
        warnings=(),
    )


def detect_release_artifacts(
    paths: Iterable[str | Path],
    *,
    root_path: str | Path | None = None,
    allow_submission_zip: bool = False,
    allow_dataset_exports: bool = False,
    allow_notebooks: bool = False,
) -> tuple[ArtifactSafetyFinding, ...]:
    root = None if root_path is None else Path(root_path)
    findings: list[ArtifactSafetyFinding] = []
    for item in paths:
        path = Path(item)
        rel = str(_relative(path, root)) if root is not None else str(path)
        name = path.name.lower()
        parts = tuple(part.lower() for part in Path(rel).parts)
        if path.is_dir():
            if name in CACHE_DIR_NAMES or name.startswith("pytest-cache-files-") or any(part in CACHE_DIR_NAMES for part in parts):
                findings.append(ArtifactSafetyFinding(path=rel, artifact_type="cache_dir", reason="forbidden cache/temp directory"))
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in FORBIDDEN_FILE_PATTERNS):
            findings.append(ArtifactSafetyFinding(path=rel, artifact_type="forbidden_file", reason="forbidden release artifact"))
        if not allow_dataset_exports and any(fnmatch.fnmatch(name, pattern) for pattern in DATASET_PATTERNS):
            findings.append(ArtifactSafetyFinding(path=rel, artifact_type="dataset_export", reason="dataset exports are not allowed"))
        if not allow_submission_zip and name == "submission.zip":
            findings.append(ArtifactSafetyFinding(path=rel, artifact_type="submission_package", reason="submission.zip is not allowed"))
        if not allow_notebooks and name.endswith(".ipynb"):
            findings.append(ArtifactSafetyFinding(path=rel, artifact_type="notebook", reason="notebooks are not allowed"))
    return tuple(sorted(set(findings), key=lambda item: (item.path, item.artifact_type, item.reason)))


def validate_release_artifact_safety(report: ArtifactSafetyReport) -> ArtifactSafetyReport:
    if not isinstance(report, ArtifactSafetyReport):
        raise ArtifactSafetyError("report must be an ArtifactSafetyReport.")
    expected = _payload_hash(report, "report_hash")
    if report.report_hash != expected:
        raise ArtifactSafetyError("report_hash does not match artifact safety report payload.")
    if report.passed and (report.forbidden_artifacts or report.symlink_paths or report.path_errors or report.errors):
        raise ArtifactSafetyError("passed report contains artifact safety failures.")
    return report


def _iter_release_paths(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*"), key=lambda item: item.as_posix()))


def _resolve_ignore(root: Path, item: str | Path) -> Path:
    path = Path(item)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _is_ignored(path: Path, ignore_roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in ignore_roots)


def _reject_critical_ignores(root: Path, ignore_paths: tuple[str, ...]) -> None:
    for item in ignore_paths:
        path = Path(item)
        if path.is_absolute():
            try:
                rel = path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
        else:
            rel = path
        parts = tuple(part.lower() for part in rel.parts)
        if not parts:
            continue
        if parts[0] == "tests":
            raise ArtifactSafetyError("ignore_paths cannot hide tests.")
        if parts[0] == "src":
            if len(parts) == 1:
                raise ArtifactSafetyError("ignore_paths cannot hide src.")
            if len(parts) >= 2 and parts[:2] == ("src", "nemotron_engine"):
                if len(parts) == 2 or parts[2] in CRITICAL_SOURCE_DIRS:
                    raise ArtifactSafetyError("ignore_paths cannot hide critical locked source modules.")


def _relative(path: Path, root: Path | None) -> Path:
    if root is None:
        return path
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _is_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(path.lstat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "ArtifactSafetyConfig",
    "ArtifactSafetyError",
    "ArtifactSafetyFinding",
    "ArtifactSafetyReport",
    "detect_release_artifacts",
    "scan_release_artifacts",
    "validate_release_artifact_safety",
]

"""Static import and side-effect safety scanning for release candidates."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from nemotron_engine.core.schemas import stable_hash


class ImportSafetyError(ValueError):
    """Raised when import safety reports are inconsistent."""


FORBIDDEN_IMPORTS = frozenset({"torch", "transformers", "peft", "accelerate", "datasets", "trl", "kaggle", "openai", "wandb", "mlflow"})


@dataclass(frozen=True)
class ImportSafetyFinding:
    path: str
    line_number: int
    pattern: str
    severity: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ImportSafetyError("finding path must be non-empty.")
        if not isinstance(self.line_number, int) or isinstance(self.line_number, bool) or self.line_number < 1:
            raise ImportSafetyError("finding line_number must be a positive integer.")
        if self.severity not in {"error", "warning"}:
            raise ImportSafetyError("finding severity must be error or warning.")
        for name in ("pattern", "message"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ImportSafetyError(f"finding {name} must be non-empty.")


@dataclass(frozen=True)
class ImportSafetyConfig:
    scan_roots: tuple[str | Path, ...] = ("src/nemotron_engine",)
    repository_root: str | Path | None = None
    allow_test_files: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.scan_roots, tuple) or not self.scan_roots:
            raise ImportSafetyError("scan_roots must be a non-empty tuple.")
        if not isinstance(self.allow_test_files, bool):
            raise ImportSafetyError("allow_test_files must be a boolean.")
        object.__setattr__(self, "scan_roots", tuple(str(item) for item in self.scan_roots))
        if self.repository_root is not None:
            object.__setattr__(self, "repository_root", str(self.repository_root))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise ImportSafetyError("config_hash does not match import safety config payload.")


@dataclass(frozen=True)
class ImportSafetyReport:
    passed: bool
    scanned_files: tuple[str, ...]
    findings: tuple[ImportSafetyFinding, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        scanned = tuple(str(item) for item in self.scanned_files)
        findings = tuple(sorted(self.findings, key=lambda item: (item.path, item.line_number, item.pattern)))
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and (errors or any(item.severity == "error" for item in findings)):
            raise ImportSafetyError("passed=True cannot include import safety errors.")
        object.__setattr__(self, "scanned_files", scanned)
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise ImportSafetyError("report_hash does not match import safety report payload.")


def scan_for_forbidden_imports(paths: Iterable[str | Path], *, repository_root: str | Path | None = None) -> tuple[ImportSafetyFinding, ...]:
    findings: list[ImportSafetyFinding] = []
    root = Path(repository_root).resolve() if repository_root is not None else None
    for path in _python_files(paths):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    if top_level in FORBIDDEN_IMPORTS:
                        findings.append(_finding(path, root, node.lineno, f"import {top_level}", f"forbidden import: {alias.name}"))
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level = node.module.split(".")[0]
                if top_level in FORBIDDEN_IMPORTS:
                    findings.append(_finding(path, root, node.lineno, f"from {top_level}", f"forbidden import: {node.module}"))
    return tuple(sorted(findings, key=lambda item: (item.path, item.line_number, item.pattern)))


def scan_for_forbidden_calls(paths: Iterable[str | Path], *, repository_root: str | Path | None = None) -> tuple[ImportSafetyFinding, ...]:
    findings: list[ImportSafetyFinding] = []
    root = Path(repository_root).resolve() if repository_root is not None else None
    for path in _python_files(paths):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                dotted = _dotted_name(node)
                if dotted.startswith("openai."):
                    findings.append(_finding(path, root, node.lineno, "openai.", "forbidden OpenAI API access pattern"))
                if dotted.startswith("kaggle.api"):
                    findings.append(_finding(path, root, node.lineno, "kaggle.api", "forbidden Kaggle API access pattern"))
            if isinstance(node, ast.Call):
                dotted = _dotted_name(node.func)
                base = dotted.split(".")[-1] if dotted else ""
                pattern = _forbidden_call_pattern(dotted, base)
                if pattern is not None:
                    findings.append(_finding(path, root, node.lineno, pattern, f"forbidden call: {pattern}"))
                if _writes_submission_zip(node, dotted):
                    findings.append(_finding(path, root, node.lineno, "submission.zip write", "forbidden automatic submission.zip write"))
    unique = {(item.path, item.line_number, item.pattern, item.message): item for item in findings}
    return tuple(sorted(unique.values(), key=lambda item: (item.path, item.line_number, item.pattern)))


def audit_import_safety(config: ImportSafetyConfig | None = None) -> ImportSafetyReport:
    cfg = config or ImportSafetyConfig()
    root = Path(cfg.repository_root).resolve() if cfg.repository_root is not None else Path.cwd().resolve()
    scan_paths = tuple((root / item).resolve() if not Path(item).is_absolute() else Path(item).resolve() for item in cfg.scan_roots)
    python_files = _filter_test_files(_python_files(scan_paths), root=root, allow_test_files=cfg.allow_test_files)
    files = tuple(str(_relative(path, root)) for path in python_files)
    findings = scan_for_forbidden_imports(python_files, repository_root=root) + scan_for_forbidden_calls(python_files, repository_root=root)
    findings = tuple(sorted(findings, key=lambda item: (item.path, item.line_number, item.pattern)))
    errors = tuple(f"{item.path}:{item.line_number}:{item.pattern}" for item in findings if item.severity == "error")
    return ImportSafetyReport(
        passed=not errors,
        scanned_files=files,
        findings=findings,
        errors=errors,
        warnings=(),
    )


def _python_files(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_file() and path.suffix == ".py":
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(child for child in path.rglob("*.py") if child.is_file()))
    return tuple(sorted(set(path.resolve() for path in result)))


def _filter_test_files(paths: Iterable[Path], *, root: Path, allow_test_files: bool) -> tuple[Path, ...]:
    if allow_test_files:
        return tuple(paths)
    result: list[Path] = []
    for path in paths:
        rel = _relative(path, root)
        if "tests" not in tuple(part.lower() for part in rel.parts):
            result.append(path)
    return tuple(result)


def _parse(path: Path) -> ast.AST:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise ImportSafetyError(f"could not parse Python source: {path}") from exc


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _forbidden_call_pattern(dotted: str, base: str) -> str | None:
    if dotted in {"subprocess.run", "subprocess.Popen", "os.system", "requests.post", "torch.save"}:
        return dotted
    if dotted.endswith(".save_pretrained"):
        return "model.save_pretrained"
    if dotted.endswith(".train") and dotted.split(".")[-2:] == ["trainer", "train"]:
        return "trainer.train"
    if base in {"eval", "exec", "submit", "push_to_hub"}:
        return base
    return None


def _writes_submission_zip(node: ast.Call, dotted: str) -> bool:
    if _path_write_to_submission_zip(node, dotted):
        return True
    if _zipfile_write_to_submission_zip(node, dotted):
        return True
    if not any(_string_contains(arg, "submission.zip") for arg in node.args):
        return False
    if dotted.endswith("write_text") or dotted.endswith("write_bytes") or dotted.endswith("writestr"):
        return True
    if dotted == "open":
        if len(node.args) < 2:
            return False
        mode = _literal_string(node.args[1])
        return mode is not None and any(flag in mode for flag in ("w", "a", "x", "+"))
    return False


def _path_write_to_submission_zip(node: ast.Call, dotted: str) -> bool:
    if not dotted.endswith("write_text") and not dotted.endswith("write_bytes"):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Call):
        return False
    constructor = _dotted_name(func.value.func)
    if constructor not in {"Path", "pathlib.Path"}:
        return False
    return bool(func.value.args) and _string_contains(func.value.args[0], "submission.zip")


def _zipfile_write_to_submission_zip(node: ast.Call, dotted: str) -> bool:
    if dotted != "zipfile.ZipFile":
        return False
    if not node.args or not _string_contains(node.args[0], "submission.zip"):
        return False
    if len(node.args) < 2:
        return False
    mode = _literal_string(node.args[1])
    return mode is not None and any(flag in mode for flag in ("w", "a", "x", "+"))


def _string_contains(node: ast.AST, needle: str) -> bool:
    value = _literal_string(node)
    return value is not None and needle in value


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _finding(path: Path, root: Path | None, line_number: int, pattern: str, message: str) -> ImportSafetyFinding:
    return ImportSafetyFinding(
        path=str(_relative(path, root)) if root is not None else str(path),
        line_number=int(line_number),
        pattern=pattern,
        severity="error",
        message=message,
    )


def _relative(path: Path, root: Path | None) -> Path:
    if root is None:
        return path
    try:
        return path.resolve().relative_to(root)
    except ValueError:
        return path


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "ImportSafetyConfig",
    "ImportSafetyError",
    "ImportSafetyFinding",
    "ImportSafetyReport",
    "audit_import_safety",
    "scan_for_forbidden_calls",
    "scan_for_forbidden_imports",
]

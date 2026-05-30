from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


OUTPUT_PATH_FIELDS = {
    "token_output",
    "token_manifest_path",
    "train_manifest_path",
    "calibrated_eval_path",
    "parent_calibration_report_path",
    "candidate_decision_path",
    "submission_zip_path",
    "package_output_path",
}

OUTPUT_DIR_FIELDS = {
    "token_output_dir",
    "output_adapter_dir",
    "eval_output_dir",
    "stage_log_dir",
}

PACKAGE_FIELDS = {"submission_zip_path", "package_output_path"}


def is_under_path(path: Path, parent: Path) -> bool:
    try:
        resolved_path = Path(path).resolve(strict=False)
        resolved_parent = Path(parent).resolve(strict=False)
    except OSError:
        resolved_path = Path(str(path)).absolute()
        resolved_parent = Path(str(parent)).absolute()
    try:
        resolved_path.relative_to(resolved_parent)
        return True
    except ValueError:
        return False


def is_kaggle_input_path(path: Path) -> bool:
    return is_under_path(Path(path), Path("/kaggle/input"))


def is_kaggle_working_path(path: Path) -> bool:
    return is_under_path(Path(path), Path("/kaggle/working"))


def _looks_like_kaggle_absolute(path: Path) -> bool:
    return path.is_absolute() and str(path).replace("\\", "/").startswith("/kaggle/")


def require_not_kaggle_input_path(path: str | Path, *, field_name: str) -> Path:
    candidate = Path(path)
    if is_kaggle_input_path(candidate):
        raise SystemExit(f"{field_name} must not write under /kaggle/input: {candidate}")
    return candidate


def require_writable_output_path(
    path: str | Path,
    *,
    field_name: str,
    must_be_under_working_when_kaggle: bool = True,
    allow_tmp_runtime_patch: bool = False,
) -> Path:
    candidate = require_not_kaggle_input_path(path, field_name=field_name)
    normalized = str(candidate).replace("\\", "/")
    if (candidate.is_absolute() and is_under_path(candidate, Path("/tmp")) or normalized.startswith("/tmp/") or normalized == "/tmp") and not allow_tmp_runtime_patch:
        raise SystemExit(f"{field_name} must not use /tmp for training/eval outputs: {candidate}")
    if must_be_under_working_when_kaggle and _looks_like_kaggle_absolute(candidate) and not is_kaggle_working_path(candidate):
        raise SystemExit(f"{field_name} must be under /kaggle/working on Kaggle: {candidate}")
    return candidate


def require_writable_output_dir(
    path: str | Path,
    *,
    field_name: str,
    must_be_under_working_when_kaggle: bool = True,
    allow_tmp_runtime_patch: bool = False,
) -> Path:
    return require_writable_output_path(
        path,
        field_name=field_name,
        must_be_under_working_when_kaggle=must_be_under_working_when_kaggle,
        allow_tmp_runtime_patch=allow_tmp_runtime_patch,
    )


def require_safe_config_output_paths(
    config: dict,
    *,
    stage: str | None = None,
    allow_submission_artifact: bool = False,
) -> dict:
    for field in sorted(OUTPUT_PATH_FIELDS):
        value = config.get(field)
        if value in (None, ""):
            continue
        if field in PACKAGE_FIELDS and not allow_submission_artifact:
            raise SystemExit(f"{field} is forbidden in active Day 1.2 config: {value}")
        require_writable_output_path(value, field_name=field)
    for field in sorted(OUTPUT_DIR_FIELDS):
        value = config.get(field)
        if value in (None, ""):
            continue
        require_writable_output_dir(value, field_name=field)
    return config


def safe_mkdir_for_output(path: str | Path, *, field_name: str) -> Path:
    directory = require_writable_output_dir(path, field_name=field_name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_write_text(path: str | Path, content: str, *, field_name: str, encoding: str = "utf-8") -> Path:
    target = require_writable_output_path(path, field_name=field_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding=encoding)
    return target


@contextmanager
def safe_open_text_for_write(
    path: str | Path,
    *,
    field_name: str,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Iterator:
    target = require_writable_output_path(path, field_name=field_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding=encoding, newline=newline) as handle:
        yield handle

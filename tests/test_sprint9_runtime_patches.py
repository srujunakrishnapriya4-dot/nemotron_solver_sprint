from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

import kaggle_runtime_patches  # noqa: E402


def test_runtime_patch_functions_exist_and_name_cutlass_ptxas() -> None:
    text = (ROOT / "kaggle_anti086" / "kaggle_runtime_patches.py").read_text(encoding="utf-8")
    assert "nvidia_cutlass_dsl/python_packages" in text
    assert "ptxas-blackwell" in text
    assert hasattr(kaggle_runtime_patches, "apply_runtime_patches")


def test_auto_or_none_is_not_real_path() -> None:
    assert kaggle_runtime_patches.normalize_optional_path("auto_or_none") == "auto_or_none"
    with pytest.raises(SystemExit, match="auto_or_none"):
        kaggle_runtime_patches.require_real_optional_path("auto_or_none", field_name="parent_adapter_path")
    assert kaggle_runtime_patches.normalize_optional_path("none") is None

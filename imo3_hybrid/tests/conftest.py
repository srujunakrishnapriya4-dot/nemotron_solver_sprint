# tests/conftest.py
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import uuid

import pytest


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()


@pytest.fixture
def tmp_path() -> Path:
    root = Path(tempfile.gettempdir()) / "imo3_hybrid_pytest_local"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path

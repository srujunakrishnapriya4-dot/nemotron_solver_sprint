from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.backend_capability_matrix import build_backend_capability_matrix  # noqa: E402


def test_backend_matrix_does_not_assume_amd_can_train_cuda_stack(tmp_path: Path) -> None:
    matrix = build_backend_capability_matrix(tmp_path / "backend.json")

    amd = matrix["backends"]["amd_developer_cloud"]
    assert amd["can_train_Nemotron"] is False
    assert amd["can_use_Triton_CUTLASS"] is False
    assert matrix["verdict"]["amd_cloud_default"] == "data_generation_and_validation_only"


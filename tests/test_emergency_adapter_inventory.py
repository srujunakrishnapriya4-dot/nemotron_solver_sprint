from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.emergency_score_recovery.adapter_inventory import find_adapter_dirs, inspect_adapter_dir, scan_adapter_roots  # noqa: E402


def make_adapter(root: Path, *, rank: int = 32, size: int = 2048, extra: str | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text(json.dumps({"r": rank, "target_modules": ["q_proj"], "base_model_name_or_path": "base"}), encoding="utf-8")
    model = root / "adapter_model.safetensors"
    model.write_bytes(b"x")
    model.open("r+b").truncate(size)
    if extra:
        (root / extra).write_text("bad", encoding="utf-8")
    return root


def test_adapter_inventory_detects_valid_adapter_dirs(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path / "input" / "adapter")

    assert find_adapter_dirs(tmp_path) == (adapter,)
    records = scan_adapter_roots((tmp_path,))

    assert len(records) == 1
    assert records[0].packageable is True
    assert records[0].rank == 32


def test_rank_over_32_rejected(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path / "adapter", rank=64)
    record = inspect_adapter_dir(adapter)

    assert record.packageable is False
    assert "rank_gt_32" in record.package_rejection_reasons


def test_huge_custom_adapter_rejected_unless_known_good_parent(tmp_path: Path) -> None:
    custom = make_adapter(tmp_path / "custom_adapter", size=1600 * 1024 * 1024)
    public = make_adapter(tmp_path / "huikang" / "nemotron-adapter" / "transformers" / "default" / "20", size=1600 * 1024 * 1024)

    custom_record = inspect_adapter_dir(custom)
    public_record = inspect_adapter_dir(public)

    assert custom_record.huge_suspicious is True
    assert custom_record.packageable is False
    assert public_record.resembles_known_good_parent is True
    assert public_record.huge_suspicious is False


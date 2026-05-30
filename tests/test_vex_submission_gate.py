from __future__ import annotations

import json
import sys
from pathlib import Path
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.vex_eval_plan import vllm_eval_settings  # noqa: E402
from nemotron_engine.vex_progress_bridge.vex_submission_gate import evaluate_vex_submission_gate, package_vex_adapter  # noqa: E402


def make_adapter(path: Path, *, size: int = 2048) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")
    model = path / "adapter_model.safetensors"
    model.write_bytes(b"x")
    model.open("r+b").truncate(size)
    return path


def test_submission_gate_rejects_4gb_custom_adapter() -> None:
    adapter = make_adapter(Path(__file__).parent / "_tmp_vex_adapter", size=2048)
    try:
        decision = evaluate_vex_submission_gate(adapter, rank=32, adapter_size_limit_mb=0, eval_summary={"exact_answer_format_pass_rate": 1.0})
        assert decision["decision"] == "REJECT"
        assert "adapter_size_exceeds_limit" in decision["reasons"]
    finally:
        for child in adapter.iterdir():
            child.unlink()
        adapter.rmdir()


def test_submission_gate_rejects_full_prompt_loss_lineage(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path / "adapter")
    decision = evaluate_vex_submission_gate(adapter, rank=32, full_prompt_loss=True, eval_summary={"exact_answer_format_pass_rate": 1.0})

    assert decision["decision"] == "REJECT"
    assert "full_prompt_loss" in decision["reasons"]


def test_submission_gate_rejects_rank_over_32(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path / "adapter")
    decision = evaluate_vex_submission_gate(adapter, rank=64, eval_summary={"exact_answer_format_pass_rate": 1.0})

    assert "rank_gt_32" in decision["reasons"]


def test_vllm_eval_command_config_matches_expected_decoding() -> None:
    settings = vllm_eval_settings("smoke_16")

    assert settings["temperature"] == 0.0
    assert settings["top_p"] == 1.0
    assert settings["uses_lora_request"] is True
    assert settings["max_rows"] == 16


def test_submission_zip_contains_only_adapter_files(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path / "adapter")
    manifest = package_vex_adapter(adapter, tmp_path / "submission.zip", tmp_path / "manifest.json")

    assert manifest["contents"] == ["adapter_config.json", "adapter_model.safetensors"]
    with zipfile.ZipFile(tmp_path / "submission.zip") as archive:
        assert sorted(archive.namelist()) == ["adapter_config.json", "adapter_model.safetensors"]

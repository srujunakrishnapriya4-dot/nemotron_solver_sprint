from pathlib import Path

from kaggle_anti086.training.day2_lora_train_report import (
    adapter_dir_valid,
    build_matrix_report,
    validate_single_train_report,
)


def make_adapter_dir(tmp_path: Path, name: str) -> Path:
    p = tmp_path / "adapters" / name
    p.mkdir(parents=True)
    (p / "adapter_config.json").write_text("{}", encoding="utf-8")
    (p / "adapter_model.safetensors").write_bytes(b"fake")
    return p


def base_report(tmp_path: Path, name: str = "A1_direct_qv_r32") -> dict:
    adapter_path = make_adapter_dir(tmp_path, name)
    return {
        "adapter_name": name,
        "dataset_variant": "direct",
        "target_modules": ["q_proj", "v_proj"],
        "rank": 32,
        "learning_rate": 0.00002,
        "max_seq_len": 2048,
        "train_rows": 50000,
        "eval_rows": 5000,
        "train_loss_start": 12.0,
        "train_loss_end": 8.0,
        "loss_finite": True,
        "nan_count": 0,
        "inf_count": 0,
        "adapter_path": str(adapter_path),
        "safe_for_eval": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
    }


def test_adapter_dir_valid(tmp_path: Path):
    p = make_adapter_dir(tmp_path, "A1_direct_qv_r32")
    assert adapter_dir_valid(p)


def test_validate_good_report(tmp_path: Path):
    r = base_report(tmp_path)
    assert validate_single_train_report(r) == []


def test_validate_rejects_rank_over_32(tmp_path: Path):
    r = base_report(tmp_path)
    r["rank"] = 64
    reasons = validate_single_train_report(r)
    assert "rank_exceeds_32" in reasons


def test_validate_rejects_package_flag(tmp_path: Path):
    r = base_report(tmp_path)
    r["package_authorized"] = True
    reasons = validate_single_train_report(r)
    assert "package_authorized_must_be_false" in reasons


def test_validate_rejects_missing_adapter_files(tmp_path: Path):
    r = base_report(tmp_path)
    r["adapter_path"] = str(tmp_path / "missing")
    reasons = validate_single_train_report(r)
    assert "adapter_dir_invalid" in reasons


def test_matrix_pass_requires_three_good_adapters(tmp_path: Path):
    reports = [
        base_report(tmp_path, "A1_direct_qv_r32"),
        base_report(tmp_path, "A2_short_trace_qv_r32"),
        base_report(tmp_path, "A3_mixed_curriculum_qv_r32"),
    ]
    matrix = build_matrix_report(reports, out_dir=tmp_path)
    assert matrix["status"] == "PASS"
    assert matrix["passed_adapter_count"] == 3
    assert matrix["safe_for_phase10_adapter_eval"] is True
    assert matrix["package_authorized"] is False
    assert matrix["submission_authorized"] is False
    assert matrix["leaderboard_claim"] is False


def test_matrix_fails_with_only_two_good_adapters(tmp_path: Path):
    reports = [
        base_report(tmp_path, "A1_direct_qv_r32"),
        base_report(tmp_path, "A2_short_trace_qv_r32"),
    ]
    matrix = build_matrix_report(reports, out_dir=tmp_path)
    assert matrix["status"] == "FAIL"
    assert "fewer_than_3_adapters_passed:2" in matrix["blocked_reasons"]

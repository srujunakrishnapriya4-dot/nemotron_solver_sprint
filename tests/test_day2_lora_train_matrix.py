import json
from pathlib import Path

import pytest

from kaggle_anti086.training.day2_lora_train_matrix import (
    FIRST_WAVE_ORDER,
    VARIANTS,
    materialize_variant_dataset,
    phase8_gate,
    prepare_plan,
    validate_sft_row,
    validate_variant_config,
)


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")


def good_row(i: int):
    return {
        "prompt": f"Solve problem {i}",
        "target": f"Reason briefly. Therefore \\boxed{{{i}}}",
        "answer": str(i),
        "verification_status": "PASS",
        "ambiguity_count": 0,
    }


def install_phase8_pass_reports(out_dir: Path):
    write_json(
        out_dir / "day2_smoke_train_report.json",
        {
            "status": "PASS",
            "package_authorized": False,
            "submission_authorized": False,
            "leaderboard_claim": False,
        },
    )
    write_json(
        out_dir / "day2_adapter_delta_smoke_report.json",
        {
            "status": "PASS",
            "safe_for_full_training_matrix": True,
            "package_authorized": False,
            "submission_authorized": False,
            "leaderboard_claim": False,
        },
    )


def test_first_wave_contains_five_serious_variants():
    assert FIRST_WAVE_ORDER == [
        "A1_direct_qv_r32",
        "A3_mixed_curriculum_qv_r32",
        "A2_short_trace_qv_r32",
        "A4_format_heavy_qv_r32",
        "A5_composed_heavy_qv_r32",
    ]


def test_all_phase9_variants_are_safe_configs():
    for name, variant in VARIANTS.items():
        assert validate_variant_config(variant) == [], name
        assert variant.rank <= 32
        assert list(variant.target_modules) == ["q_proj", "v_proj"]
        assert variant.max_seq_len in {1024, 1536, 2048}
        assert 1e-6 <= variant.learning_rate <= 8e-5


def test_validate_sft_row_accepts_good_row():
    assert validate_sft_row(good_row(1), 1) is None


def test_validate_sft_row_rejects_multiple_boxed():
    row = good_row(1)
    row["target"] = "bad \\boxed{1} and \\boxed{2}"
    assert "target_must_have_exactly_one_boxed" in validate_sft_row(row, 1)


def test_validate_sft_row_rejects_abstain():
    row = good_row(1)
    row["target"] = "ABSTAIN \\boxed{1}"
    assert "target_contains_abstain" in validate_sft_row(row, 1)


def test_phase8_gate_blocks_missing_reports(tmp_path: Path):
    reasons = phase8_gate(tmp_path)
    assert "missing_phase8_smoke_train_report" in reasons
    assert "missing_phase8_adapter_delta_smoke_report" in reasons


def test_phase8_gate_passes_good_reports(tmp_path: Path):
    install_phase8_pass_reports(tmp_path)
    assert phase8_gate(tmp_path) == []


def test_prepare_plan_ready_when_phase8_passed(tmp_path: Path):
    install_phase8_pass_reports(tmp_path)
    plan = prepare_plan(tmp_path, ["A1_direct_qv_r32", "A3_mixed_curriculum_qv_r32"])
    assert plan["status"] == "READY_FOR_RUNTIME"
    assert plan["package_authorized"] is False
    assert plan["submission_authorized"] is False
    assert plan["leaderboard_claim"] is False


def test_materialize_variant_dataset(tmp_path: Path):
    install_phase8_pass_reports(tmp_path)

    v = VARIANTS["A1_direct_qv_r32"]
    rows = [good_row(i) for i in range(20)]
    write_jsonl(tmp_path / v.dataset_filename, rows)

    info = materialize_variant_dataset(
        tmp_path,
        v,
        seed=7,
        train_rows_override=10,
        eval_rows_override=5,
    )

    assert Path(info["train_path"]).exists()
    assert Path(info["eval_path"]).exists()
    assert info["train_rows"] == 10
    assert info["eval_rows"] == 5


def test_materialize_variant_dataset_rejects_bad_rows(tmp_path: Path):
    v = VARIANTS["A1_direct_qv_r32"]
    rows = [good_row(1), {"prompt": "x", "target": "no boxed", "answer": "1"}]
    write_jsonl(tmp_path / v.dataset_filename, rows)

    with pytest.raises(ValueError):
        materialize_variant_dataset(
            tmp_path,
            v,
            seed=7,
            train_rows_override=2,
            eval_rows_override=0,
        )

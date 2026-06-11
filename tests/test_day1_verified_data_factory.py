from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_anti086.training.day1_teacher_families import available_teachers
from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_row
from kaggle_anti086.training.day1_verified_data_factory import _check_preconditions, build_verified_data_factory
from kaggle_anti086.training.day1_verified_data_schema import read_jsonl


def test_small_build_creates_all_expected_files_and_trainable_rows(tmp_path: Path) -> None:
    result = build_verified_data_factory(
        tmp_path,
        train_per_family=5,
        eval_per_family=2,
        probe_per_family=2,
        hard_negatives_per_family=3,
        seed=123,
        require_preconditions=False,
    )
    files = result["output_files"]
    for key in (
        "train",
        "eval",
        "probe",
        "hard_negatives",
        "rejected",
        "manifest",
        "family_balance",
        "prompt_diversity",
        "dedup_leakage",
        "format_audit",
        "teacher_verification",
        "decision",
    ):
        assert Path(files[key]).exists(), key

    train = read_jsonl(Path(files["train"]))
    eval_rows = read_jsonl(Path(files["eval"]))
    probe = read_jsonl(Path(files["probe"]))
    pairs = read_jsonl(Path(files["hard_negatives"]))
    all_positive = [*train, *eval_rows, *probe]

    assert len(train) == 55
    assert len(eval_rows) == 22
    assert len(probe) == 22
    assert len(pairs) == 33
    assert {row["family"] for row in train} == set(available_teachers())
    assert all(verify_trainable_row(row).trainable for row in all_positive)
    prompts = [_key(row["prompt"]) for row in all_positive]
    assert len(prompts) == len(set(prompts))
    record_ids = [row["record_id"] for row in all_positive]
    assert len(record_ids) == len(set(record_ids))
    assert all(pair["chosen"] != pair["rejected"] for pair in pairs)
    assert all(pair["reason_rejected"] for pair in pairs)

    manifest = json.loads(Path(files["manifest"]).read_text(encoding="utf-8"))
    assert manifest["safe_to_train_lora"] is False
    assert manifest["safe_to_package"] is False
    assert manifest["safe_to_submit"] is False
    assert manifest["no_0_93_evidence"] is True
    assert manifest["no_0_95_evidence"] is True


def test_rejected_bad_rows_do_not_enter_positive_files(tmp_path: Path) -> None:
    result = build_verified_data_factory(
        tmp_path,
        train_per_family=3,
        eval_per_family=1,
        probe_per_family=1,
        hard_negatives_per_family=1,
        seed=456,
        require_preconditions=False,
    )
    positives = []
    for key in ("train", "eval", "probe"):
        positives.extend(read_jsonl(Path(result["output_files"][key])))
    rejected = read_jsonl(Path(result["output_files"]["rejected"]))
    rejected_prompts = {row["prompt"] for row in rejected if row.get("prompt")}
    positive_prompts = {row["prompt"] for row in positives}
    assert rejected_prompts.isdisjoint(positive_prompts)


def test_precondition_failure_blocks_factory(tmp_path: Path) -> None:
    bank = tmp_path / "bank.json"
    verifier = tmp_path / "verifier.json"
    bank.write_text(json.dumps({"teacher_bank_ready_for_verified_data_factory": False}), encoding="utf-8")
    verifier.write_text(json.dumps({"trainability_gate_ready_for_phase4": True, "no_0_93_evidence": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher_bank_ready"):
        _check_preconditions({"phase2b_teacher_bank": str(bank), "phase3_verifier": str(verifier)})


def _key(prompt: str) -> str:
    return " ".join(prompt.split())

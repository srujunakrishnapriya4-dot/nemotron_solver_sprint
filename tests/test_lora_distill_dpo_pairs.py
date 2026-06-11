from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_anti086.training.day1_dpo_pair_audit import validate_dpo_pair
from kaggle_anti086.training.day1_verified_data_factory import build_verified_data_factory
from kaggle_anti086.training.lora_distill_dpo_pairs import (
    DPOPair,
    _check_phase4_preconditions,
    build_dpo_pair_factory,
    read_dpo_jsonl,
    stable_dpo_pair_id,
    to_json_dict,
    write_jsonl,
)


def _pair(**overrides: object) -> DPOPair:
    chosen = "1 m = 100 cm.\n\\boxed{100}"
    rejected = "Use the wrong conversion factor.\nThis gives 10.\n\\boxed{10}"
    data = dict(
        schema_version=1,
        id=stable_dpo_pair_id("unit_conversion", "Convert 1 m to cm.", chosen, rejected, 123),
        family="unit_conversion",
        rule_id="unit_exact",
        difficulty="easy",
        prompt_style="direct",
        prompt="Convert 1 m to cm.",
        chosen=chosen,
        rejected=rejected,
        correct_answer="100",
        rejected_answer="10",
        reason_rejected="failed_unit_wrong_factor",
        chosen_verification_status="PASS",
        rejected_verification_status="FAIL",
        chosen_trainable=True,
        rejected_trainable=False,
        split="train",
        source="teacher_hard_negative",
        seed=123,
        metadata={},
    )
    data.update(overrides)
    return DPOPair(**data)


def test_dpo_pair_serializes_and_stable_id_is_deterministic() -> None:
    pair = _pair()
    assert to_json_dict(pair)["family"] == "unit_conversion"
    assert stable_dpo_pair_id("a", "b", "c", "d", 1) == stable_dpo_pair_id("a", "b", "c", "d", 1)
    assert stable_dpo_pair_id("a", "b", "c", "d", 1) != stable_dpo_pair_id("a", "b", "c", "d", 2)


def test_dpo_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [_pair()])
    rows = read_dpo_jsonl(path)
    assert rows[0]["reason_rejected"] == "failed_unit_wrong_factor"


def test_valid_pair_passes_validation() -> None:
    ok, failures = validate_dpo_pair(_pair())
    assert ok
    assert failures == []


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"rejected": "1 m = 100 cm.\n\\boxed{100}"}, "rejected_accidentally_trainable"),
        ({"chosen": "no box"}, "chosen_not_trainable"),
        ({"chosen": "1 m = 100 cm.\n\\boxed{100} trailing"}, "chosen_not_trainable"),
        ({"rejected": "1 m = 100 cm.\n\\boxed{100}", "rejected_answer": "100"}, "rejected_accidentally_trainable"),
        ({"rejected": "wrong\n\\boxed{100}", "rejected_answer": "100"}, "rejected_answer_matches_correct"),
        ({"reason_rejected": "not_a_reason"}, "unknown_rejection_reason"),
        ({"metadata": {"unsafe": True}}, "unsafe_metadata"),
        ({"source": "model_guess_unverified"}, "model_guess_unverified"),
        ({"metadata": {"local_recovery_claim": True}}, "unsafe_metadata"),
        ({"metadata": {"router_integration": True}}, "unsafe_metadata"),
        ({"split": "probe"}, "invalid_split"),
    ],
)
def test_invalid_pairs_fail_validation(updates: dict[str, object], reason: str) -> None:
    ok, failures = validate_dpo_pair(_pair(**updates))
    assert not ok
    assert reason in failures


def test_format_negative_may_box_correct_answer_but_still_fail_gate() -> None:
    pair = _pair(
        rejected="\\boxed{100} trailing",
        rejected_answer="100",
        reason_rejected="failed_format_text_after_box",
        source="format_negative",
    )
    ok, failures = validate_dpo_pair(pair)
    assert ok, failures


def test_small_dpo_build_creates_outputs_and_valid_pairs(tmp_path: Path) -> None:
    phase4 = tmp_path / "phase4"
    build_verified_data_factory(
        phase4,
        train_per_family=20,
        eval_per_family=5,
        probe_per_family=5,
        hard_negatives_per_family=10,
        seed=123,
        require_preconditions=False,
    )
    out = tmp_path / "phase5"
    result = build_dpo_pair_factory(
        out,
        phase4_dir=phase4,
        train_pairs_per_family=10,
        eval_pairs_per_family=3,
        seed=123,
        require_preconditions=False,
    )
    files = result["output_files"]
    for key in ("train_pairs", "eval_pairs", "rejected_pairs", "audit", "manifest", "decision"):
        assert Path(files[key]).exists()
    train = read_dpo_jsonl(Path(files["train_pairs"]))
    eval_rows = read_dpo_jsonl(Path(files["eval_pairs"]))
    assert len(train) == 110
    assert len(eval_rows) == 33
    assert {row["family"] for row in train} == {
        "symbol_mapping",
        "bit_manipulation",
        "char_cipher",
        "unit_conversion",
        "numeric_formula_safe",
        "word_cipher",
        "custom_numeral",
        "permutation_sorting",
        "gravity_numeric",
        "equation_operator",
        "sequence_pattern",
    }
    assert all(validate_dpo_pair(row)[0] for row in [*train, *eval_rows])
    train_prompts = {" ".join(row["prompt"].split()) for row in train}
    eval_prompts = {" ".join(row["prompt"].split()) for row in eval_rows}
    assert not (train_prompts & eval_prompts)
    manifest = json.loads(Path(files["manifest"]).read_text(encoding="utf-8"))
    assert manifest["safe_to_train_lora"] is False
    assert manifest["safe_to_package"] is False
    assert manifest["safe_to_submit"] is False
    assert manifest["no_0_93_evidence"] is True
    assert manifest["no_0_95_evidence"] is True


def test_precondition_failure_blocks_factory(tmp_path: Path) -> None:
    manifest = tmp_path / "phase4_verified_data_manifest.json"
    manifest.write_text(json.dumps({"gates": {"phase4_ready_for_phase5": False}}), encoding="utf-8")
    with pytest.raises(ValueError, match="phase4_ready_for_phase5"):
        _check_phase4_preconditions(manifest)

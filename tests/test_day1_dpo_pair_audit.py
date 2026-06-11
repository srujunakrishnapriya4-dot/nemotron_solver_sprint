from __future__ import annotations

from kaggle_anti086.training.day1_dpo_pair_audit import audit_dpo_pairs
from kaggle_anti086.training.lora_distill_dpo_pairs import DPOPair, stable_dpo_pair_id


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


def test_audit_status_pass_for_clean_small_build_like_pairs() -> None:
    train = [_pair(family="unit_conversion", id=f"t{i}", prompt=f"Convert {i + 1} m to cm.") for i in range(2)]
    eval_rows = [_pair(family="unit_conversion", id=f"e{i}", prompt=f"Convert {i + 10} m to cm.", split="eval") for i in range(2)]
    result = audit_dpo_pairs(train, eval_rows, [])
    assert result["validation_fail_count"] == 0
    assert result["duplicate_pair_count"] == 0
    assert result["train_eval_prompt_overlap_count"] == 0


def test_audit_catches_duplicate_id_and_prompt_overlap() -> None:
    train = [_pair(id="dup"), _pair(id="dup", prompt="Convert 2 m to cm.")]
    eval_rows = [_pair(id="eval", split="eval")]
    result = audit_dpo_pairs(train, eval_rows, [])
    assert result["status"] == "FAIL"
    assert result["duplicate_pair_count"] == 1
    assert result["train_eval_prompt_overlap_count"] == 1


def test_audit_catches_rejected_accidentally_trainable_and_chosen_equals_rejected() -> None:
    accepted_rejected = _pair(rejected="1 m = 100 cm.\n\\boxed{100}", rejected_answer="100")
    same = _pair(id="same", rejected="1 m = 100 cm.\n\\boxed{100}", chosen="1 m = 100 cm.\n\\boxed{100}")
    result = audit_dpo_pairs([accepted_rejected, same], [], [])
    assert result["status"] == "FAIL"
    assert result["rejected_accidentally_trainable_count"] >= 1
    assert result["chosen_equals_rejected_count"] >= 1


def test_audit_catches_unknown_reason_and_missing_family() -> None:
    bad_reason = _pair(reason_rejected="bad_reason")
    missing_family = _pair(id="missing", family="unknown_family")
    result = audit_dpo_pairs([bad_reason, missing_family], [], [])
    assert result["status"] == "FAIL"
    assert result["unknown_reason_count"] >= 1
    assert any("zero_train_pairs" in failure for failure in result["failures"])

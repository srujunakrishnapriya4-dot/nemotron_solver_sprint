from __future__ import annotations

from pathlib import Path

from kaggle_anti086.training.day1_verified_data_schema import (
    HardNegativePair,
    VerifiedSFTRecord,
    read_jsonl,
    stable_pair_id,
    stable_record_id,
    to_json_dict,
    write_jsonl,
)


def _record() -> VerifiedSFTRecord:
    return VerifiedSFTRecord(
        record_id="r1",
        family="unit_conversion",
        rule_id="unit_exact",
        difficulty="easy",
        prompt_style="direct",
        prompt="Convert 1 m to cm.",
        answer="100",
        trace="1 m = 100 cm.\n\\boxed{100}",
        target_text="1 m = 100 cm.\n\\boxed{100}",
        source="deterministic_teacher",
        verification_status="PASS",
        ambiguity_count=0,
        trainable=True,
        split="train",
        seed=123,
        teacher_version="test",
        metadata={},
    )


def test_verified_sft_record_serializes_to_json_dict() -> None:
    data = to_json_dict(_record())
    assert data["record_id"] == "r1"
    assert data["messages"][0] == {"role": "user", "content": "Convert 1 m to cm."}
    assert data["messages"][1]["content"] == data["target_text"]


def test_hard_negative_pair_serializes_to_json_dict() -> None:
    pair = HardNegativePair(
        pair_id="p1",
        family="unit_conversion",
        rule_id="unit_exact",
        difficulty="easy",
        prompt_style="direct",
        prompt="Convert 1 m to cm.",
        chosen="\\boxed{100}",
        rejected="\\boxed{10}",
        correct_answer="100",
        rejected_answer="10",
        reason_rejected="wrong_factor",
        verifier_status="FAIL",
        split="hard_negative",
        seed=123,
        metadata={},
    )
    assert to_json_dict(pair)["reason_rejected"] == "wrong_factor"


def test_stable_ids_are_deterministic() -> None:
    assert stable_record_id("a", "b", "c", "d", 1) == stable_record_id("a", "b", "c", "d", 1)
    assert stable_pair_id("a", "b", "c", "d", 1) == stable_pair_id("a", "b", "c", "d", 1)
    assert stable_record_id("a", "b", "c", "d", 1) != stable_record_id("a", "b", "c", "d", 2)


def test_jsonl_write_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [_record()])
    rows = read_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["record_id"] == "r1"
    assert rows[0]["messages"][1]["content"].endswith("\\boxed{100}")

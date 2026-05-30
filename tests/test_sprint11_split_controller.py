from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_anti086.data.split_controller import SplitConfig, build_splits, check_split_leakage, write_split_report


def row(idx: int, *, split: str = "train", rule_id: str | None = None, leakage_group: str | None = None, status: str = "verified", family: str = "bit_manipulation") -> dict:
    return {
        "id": f"r{idx}",
        "family": family,
        "subfamily": "fixture",
        "rule_id": rule_id or f"rule{idx}",
        "prompt": f"prompt {idx}",
        "answer": "1",
        "source": "unit",
        "solver_name": "fixture",
        "verification_status": status,
        "difficulty": 0.5,
        "split": split,
        "leakage_group": leakage_group or f"lg{idx}",
    }


def test_clean_split_passes_and_report_writes(tmp_path: Path) -> None:
    report = check_split_leakage([row(1)], {"rule_holdout_eval": [row(2, split="rule_holdout_eval")], "anti_leak_eval": [row(3, split="anti_leak_eval")]})
    assert report.status == "PASS"
    out = tmp_path / "report.json"
    write_split_report(report, out)
    assert out.exists()


def test_rule_holdout_overlap_fails() -> None:
    report = check_split_leakage([row(1, rule_id="same")], {"rule_holdout_eval": [row(2, split="rule_holdout_eval", rule_id="same")]})
    assert report.status == "FAIL"
    assert report.rule_overlap_count == 1


def test_anti_leak_group_overlap_fails() -> None:
    report = check_split_leakage([row(1, leakage_group="same")], {"anti_leak_eval": [row(2, split="anti_leak_eval", leakage_group="same")]})
    assert report.status == "FAIL"
    assert report.leakage_group_overlap_count == 1


def test_row_id_overlap_fails() -> None:
    report = check_split_leakage([row(1)], {"private_like_eval": [row(1, split="private_like_eval")]})
    assert report.status == "FAIL"
    assert report.row_id_overlap_count == 1


def test_unverified_and_unsupported_equation_train_fail() -> None:
    report = check_split_leakage(
        [row(1, status="unverified"), row(2, family="equation_operator", status="unsupported")],
        {},
    )
    assert report.status == "FAIL"
    assert report.unverified_train_rows == ["r1"]
    assert report.unsupported_equation_train_rows == ["r2"]


def test_build_splits_quarantines_unverified_train_and_is_deterministic() -> None:
    rows = [row(1), row(2, status="unverified"), row(3, split="private_like_eval")]
    one = build_splits(rows, SplitConfig(allow_failed_report=True, seed=123))
    two = build_splits(rows, SplitConfig(allow_failed_report=True, seed=123))
    assert [r["id"] for r in one.train] == [r["id"] for r in two.train]
    assert any(r["id"] == "r2" for r in one.quarantine)
    assert not any(r["id"] == "r2" for r in one.train)


def test_build_splits_fails_on_leakage_by_default() -> None:
    rows = [row(1, rule_id="same"), row(2, split="rule_holdout_eval", rule_id="same")]
    with pytest.raises(SystemExit):
        build_splits(rows)

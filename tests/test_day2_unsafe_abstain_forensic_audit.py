from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day2_unsafe_abstain_forensic_audit import build_unsafe_abstain_forensic_audit


TMP = Path("artifacts/test_tmp/day2_unsafe_abstain_forensic_audit")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _row(**kwargs) -> dict:
    row = {
        "row_id": "r0",
        "dataset": "core_eval",
        "family": "symbol_mapping",
        "subfamily": "expected_abstain",
        "prompt": "[x] !@ -> ab; #$ -> cd; query: !Z% output ?",
        "expected": "ABSTAIN",
        "gold_answer": "ABSTAIN",
        "raw_output": "",
        "extracted_answer": "",
        "normalized_answer": "",
        "correct": False,
        "valid": False,
        "failure_reason": "all_solvers_abstained",
        "route_reason": "solver_failed",
    }
    row.update(kwargs)
    return row


def _mining(root: Path, *, current=1383, maybe=150, unsafe=259, total=1792, target=1667) -> Path:
    return _write_json(
        root / "mining.json",
        {
            "schema_version": 1,
            "score_target": {
                "total_rows": total,
                "current_correct": current,
                "target_correct_needed": target,
                "additional_correct_needed": target - current,
                "abstain_rows": maybe + unsafe,
            },
            "recoverability_classes": {
                "maybe_recoverable_requires_new_solver": {"rows": maybe},
                "unsafe_ambiguous_do_not_answer": {"rows": unsafe},
            },
        },
    )


def _inputs(root: Path, rows_by_dataset: dict[str, list[dict]] | None = None) -> dict[str, Path]:
    rows_by_dataset = rows_by_dataset or {
        "core_eval": [_row()],
        "family_eval": [],
        "rule_holdout": [],
        "anti_leak": [],
    }
    return {
        dataset: _write_jsonl(root / f"{dataset}.jsonl", rows)
        for dataset, rows in rows_by_dataset.items()
    }


def _report(root: Path, rows: list[dict], **mining_kwargs) -> dict:
    return build_unsafe_abstain_forensic_audit(
        mining_report_path=_mining(root, **mining_kwargs),
        prediction_files=_inputs(root, {"core_eval": rows, "family_eval": [], "rule_holdout": [], "anti_leak": []}),
    )


def test_unsupported_unique_custom_numeral_is_not_truly_unsafe():
    root = _clean()
    report = _report(
        root,
        [
            _row(
                family="custom_numeral",
                prompt="[x] custom glyphs 0: 1 -> @; 2 -> &&. target number: 3",
                failure_reason="custom_numeral_solver_not_implemented",
            )
        ],
    )

    assert report["forensic_classes"]["possibly_recoverable_with_new_verified_solver"]["rows"] == 1
    assert report["forensic_classes"]["truly_unsafe_unseen_symbols"]["rows"] == 0


def test_contradictory_mapping_is_truly_unsafe():
    root = _clean()
    report = _report(root, [_row(family="char_cipher", prompt="[x] abc -> def; abc -> xyz; query: abc")])

    assert report["forensic_classes"]["truly_unsafe_contradiction"]["rows"] == 1


def test_unseen_symbol_query_is_truly_unsafe():
    root = _clean()
    report = _report(root, [_row(family="symbol_mapping", prompt="[x] !@ -> ab; #$ -> cd; query: !Z% output ?")])

    assert report["forensic_classes"]["truly_unsafe_unseen_symbols"]["rows"] == 1


def test_missing_known_bit_operation_coverage_is_possibly_recoverable():
    root = _clean()
    report = _report(
        root,
        [
            _row(
                family="bit_manipulation",
                prompt="[x] reverse bits: 0011 -> 1100. Input: 0101",
            )
        ],
    )

    assert report["forensic_classes"]["possibly_recoverable_with_new_verified_solver"]["rows"] == 1
    assert report["forensic_classes"]["truly_unsafe_multifit"]["rows"] == 0


def test_multifit_transform_remains_unsafe():
    root = _clean()
    report = _report(
        root,
        [
            _row(
                family="bit_manipulation",
                prompt="[x] 0000 -> 0000; 1111 -> 1111. Input: 0101",
            )
        ],
    )

    assert report["forensic_classes"]["truly_unsafe_multifit"]["rows"] == 1


def test_score_projection_math_is_correct():
    root = _clean()
    report = _report(
        root,
        [
            _row(
                family="custom_numeral",
                prompt="[x] custom glyphs 0: 1 -> @; 2 -> &&. target number: 3",
                failure_reason="custom_numeral_solver_not_implemented",
            )
        ],
    )

    projection = report["score_projection"]
    assert projection["if_only_maybe_rows_solved"] == pytest.approx((1383 + 150) / 1792)
    assert projection["if_medium_risk_rows_solved"] == pytest.approx((1383 + 150 + 1) / 1792)


def test_cannot_authorize_package_submission_or_training():
    root = _clean()
    report = _report(root, [_row()])

    assert report["v2a_150_authorized"] is False
    assert report["package_authorized"] is False
    assert report["submission_authorized"] is False
    assert report["leaderboard_claim"] is False


def test_cannot_reach_0_93_without_guessing_when_projected_recovery_low():
    root = _clean()
    report = _report(root, [_row(family="custom_numeral", prompt="[x] custom glyphs 0: 1 -> @; 2 -> &&. target number: 3", failure_reason="custom_numeral_solver_not_implemented")])

    assert report["score_projection"]["can_reach_0_93_without_guessing"] is False


def test_missing_mining_report_or_prediction_file_raises_loudly():
    root = _clean()
    inputs = _inputs(root)
    with pytest.raises(FileNotFoundError):
        build_unsafe_abstain_forensic_audit(mining_report_path=root / "missing.json", prediction_files=inputs)

    with pytest.raises(FileNotFoundError):
        build_unsafe_abstain_forensic_audit(
            mining_report_path=_mining(root),
            prediction_files={"core_eval": root / "missing.jsonl", "family_eval": inputs["family_eval"], "rule_holdout": inputs["rule_holdout"], "anti_leak": inputs["anti_leak"]},
        )


def test_output_is_deterministic():
    root = _clean()
    report_a = _report(root, [_row()])
    report_b = _report(root, [_row()])

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)

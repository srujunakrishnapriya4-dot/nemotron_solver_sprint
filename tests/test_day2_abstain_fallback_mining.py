from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day2_abstain_fallback_mining import build_abstain_fallback_mining_report


TMP = Path("artifacts/test_tmp/day2_abstain_fallback_mining")


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
        "prompt": "[core #0] ab -> cd; ef -> gh; query: ae",
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


def _final(root: Path, *, rows=1792, raw=1383 / 1792) -> Path:
    return _write_json(
        root / "final.json",
        {
            "schema_version": 1,
            "overall": {
                "rows": rows,
                "raw_exact_match": raw,
                "answerable_accuracy": 1.0,
                "abstain_rows": 409,
                "correct_abstain_rate": 1.0,
                "unsafe_abstain_answer_rate": 0.0,
            },
        },
    )


def _optional(root: Path) -> tuple[Path, Path]:
    return (
        _write_json(root / "resolution.json", {"policy_decision": {"decision": "ABSTAIN_NOT_ALLOWED_NEEDS_FALLBACK"}}),
        _write_json(root / "v2.json", {"schema_version": 2}),
    )


def _report(root: Path, rows_by_dataset: dict[str, list[dict]] | None = None) -> dict:
    resolution, v2 = _optional(root)
    return build_abstain_fallback_mining_report(
        prediction_files=_inputs(root, rows_by_dataset),
        resolution_report_path=resolution,
        final_solver_report_path=_final(root),
        failure_report_v2_path=v2,
    )


def test_score_target_math_for_current_day2_numbers():
    root = _clean()
    report = _report(root)

    target = report["score_target"]
    assert target["total_rows"] == 1792
    assert target["current_correct"] == 1383
    assert target["target_correct_needed"] == 1667
    assert target["additional_correct_needed"] == 284
    assert target["abstain_rows"] == 409
    assert target["required_abstain_recovery_rate"] == pytest.approx(284 / 409)


def test_abstain_rows_grouped_by_family_and_subfamily():
    root = _clean()
    report = _report(
        root,
        {
            "core_eval": [
                _row(row_id="a", family="symbol_mapping", subfamily="expected_abstain"),
                _row(row_id="b", family="custom_numeral", subfamily="unsupported"),
            ],
            "family_eval": [_row(row_id="c", family="custom_numeral", subfamily="unsupported")],
            "rule_holdout": [],
            "anti_leak": [],
        },
    )

    assert report["abstain_pool"]["by_family"] == {"custom_numeral": 2, "symbol_mapping": 1}
    assert report["abstain_pool"]["by_subfamily"] == {"expected_abstain": 1, "unsupported": 2}


def test_contradictory_mapping_rows_classified_unsafe():
    root = _clean()
    report = _report(root, {"core_eval": [_row(prompt="[x] abc -> def; abc -> xyz; query: abc")], "family_eval": [], "rule_holdout": [], "anti_leak": []})

    assert report["recoverability_classes"]["unsafe_ambiguous_do_not_answer"]["rows"] == 1
    assert report["decision"]["fallback_implementation_authorized"] is False


def test_unseen_symbol_rows_classified_unsafe_unless_inferable():
    root = _clean()
    report = _report(root, {"core_eval": [_row(prompt="[x] !@ -> ab; #$ -> cd; query: !Z% output ?")], "family_eval": [], "rule_holdout": [], "anti_leak": []})

    assert report["recoverability_classes"]["unsafe_ambiguous_do_not_answer"]["rows"] == 1
    assert report["candidate_fallback_families"][0]["recoverability"] == "unsafe"


def test_deterministic_unique_fallback_rows_classified_deterministic():
    root = _clean()
    report = _report(root, {"core_eval": [_row(prompt="[x] ab -> cd; ef -> gh; query: ae")], "family_eval": [], "rule_holdout": [], "anti_leak": []})

    assert report["recoverability_classes"]["deterministically_recoverable"]["rows"] == 1
    assert report["decision"]["fallback_implementation_authorized"] is True
    assert report["decision"]["authorized_families"][0]["family"] == "symbol_mapping"


def test_model_only_rows_are_model_fallback_and_not_authorized():
    root = _clean()
    report = _report(
        root,
        {
            "core_eval": [
                _row(
                    family="format_only",
                    prompt="[x] Requires language model judgment; query: decide",
                    failure_reason="model_fallback_required",
                )
            ],
            "family_eval": [],
            "rule_holdout": [],
            "anti_leak": [],
        },
    )

    assert report["recoverability_classes"]["requires_model_fallback_but_unverified"]["rows"] == 1
    assert report["decision"]["fallback_implementation_authorized"] is False


def test_no_low_risk_deterministic_rows_keeps_fallback_unauthorized():
    root = _clean()
    report = _report(root, {"core_eval": [_row(family="custom_numeral", failure_reason="custom_numeral_solver_not_implemented")], "family_eval": [], "rule_holdout": [], "anti_leak": []})

    assert report["recoverability_classes"]["maybe_recoverable_requires_new_solver"]["rows"] == 1
    assert report["decision"]["fallback_implementation_authorized"] is False


def test_scale_package_and_submission_are_always_false():
    root = _clean()
    report = _report(root)

    assert report["v2a_150_authorized"] is False
    assert report["package_authorized"] is False
    assert report["submission_authorized"] is False
    assert report["leaderboard_claim"] is False


def test_missing_prediction_file_raises_loudly():
    root = _clean()
    resolution, v2 = _optional(root)
    inputs = _inputs(root)
    inputs["core_eval"] = root / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        build_abstain_fallback_mining_report(
            prediction_files=inputs,
            resolution_report_path=resolution,
            final_solver_report_path=_final(root),
            failure_report_v2_path=v2,
        )


def test_output_is_deterministic():
    root = _clean()
    report_a = _report(root)
    report_b = _report(root)

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day2_abstain_resolution_report import build_abstain_resolution_report


TMP = Path("artifacts/test_tmp/day2_abstain_resolution_report")


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


def _final_report(root: Path, *, abstain_rows=2, raw=0.75, answerable=1.0, abstain_rate=1.0) -> Path:
    return _write_json(
        root / "final.json",
        {
            "schema_version": 1,
            "overall": {
                "raw_exact_match": raw,
                "answerable_accuracy": answerable,
                "abstain_rows": abstain_rows,
                "correct_abstain_rate": abstain_rate,
                "unsafe_abstain_answer_rate": 0.0,
            },
        },
    )


def _gate(root: Path) -> Path:
    return _write_json(root / "gate.json", {"schema_version": 1, "status": "WARN"})


def _row(**kwargs) -> dict:
    row = {
        "row_id": "r0",
        "dataset": "core_eval",
        "family": "format_only",
        "subfamily": "unsupported",
        "prompt": "Return ABSTAIN if unsupported.",
        "expected": "ABSTAIN",
        "gold_answer": "ABSTAIN",
        "raw_output": "",
        "extracted_answer": "",
        "normalized_answer": "",
        "correct": False,
        "valid": True,
        "failure_reason": "all_solvers_abstained",
        "route_reason": "solver_abstained",
    }
    row.update(kwargs)
    return row


def _predictions(root: Path, rows_by_dataset: dict[str, list[dict]] | None = None) -> dict[str, Path]:
    rows_by_dataset = rows_by_dataset or {
        "core_eval": [_row(row_id="c0", dataset="core_eval", family="custom_numeral")],
        "family_eval": [_row(row_id="f0", dataset="family_eval", family="sequence_pattern")],
        "rule_holdout": [],
        "anti_leak": [],
    }
    return {
        dataset: _write_jsonl(root / f"{dataset}.jsonl", rows)
        for dataset, rows in rows_by_dataset.items()
    }


def _repo(root: Path, *, policy: str) -> Path:
    repo = root / "repo"
    if policy == "unknown":
        repo.mkdir(parents=True, exist_ok=True)
        return repo
    if policy == "abstain_allowed":
        path = repo / "src/nemotron_engine/scoring/local_scorer.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ABSTAIN_ALLOWED = True\nallow_abstain = True\nallow_blank_answer = True\n", encoding="utf-8")
        return repo
    if policy == "concrete_only":
        answer = repo / "src/nemotron_engine/scoring/answer_extractor.py"
        validator = repo / "src/nemotron_engine/submission/submission_validator.py"
        answer.parent.mkdir(parents=True, exist_ok=True)
        validator.parent.mkdir(parents=True, exist_ok=True)
        answer.write_text("def extract_boxed_answer(x):\n    raise ValueError('Boxed answer cannot be empty')\n", encoding="utf-8")
        validator.write_text("from x import normalize_answer_candidate\nclass SubmissionRow:\n    answer: int\n", encoding="utf-8")
        return repo
    raise AssertionError(policy)


def _report(root: Path, *, policy: str = "unknown", rows_by_dataset: dict[str, list[dict]] | None = None) -> dict:
    return build_abstain_resolution_report(
        final_solver_report_path=_final_report(root),
        abstain_policy_gate_path=_gate(root),
        prediction_files=_predictions(root, rows_by_dataset),
        repo_root=_repo(root, policy=policy),
    )


def test_unknown_policy_with_abstain_rows_blocks_submission():
    root = _clean()
    report = _report(root, policy="unknown")

    assert report["policy_decision"]["decision"] == "ABSTAIN_POLICY_UNKNOWN_BLOCK_SUBMISSION"
    assert report["policy_decision"]["final_submission_policy_known"] is False
    assert report["decision"]["submission_authorized"] is False
    assert report["status"] == "WARN"


def test_explicit_abstain_support_yields_allowed_without_submission_authorization():
    root = _clean()
    report = _report(root, policy="abstain_allowed")

    assert report["policy_decision"]["decision"] == "ABSTAIN_ALLOWED"
    assert report["decision"]["submission_authorized"] is False
    assert report["decision"]["package_authorized"] is False


def test_concrete_answer_only_policy_requires_fallback():
    root = _clean()
    report = _report(root, policy="concrete_only")

    assert report["policy_decision"]["decision"] == "ABSTAIN_NOT_ALLOWED_NEEDS_FALLBACK"
    assert report["fallback_need"]["fallback_required"] is True
    assert report["fallback_need"]["fallback_allowed_now"] is False


def test_missing_final_solver_score_report_raises_loudly():
    root = _clean()
    with pytest.raises(FileNotFoundError):
        build_abstain_resolution_report(
            final_solver_report_path=root / "missing.json",
            abstain_policy_gate_path=_gate(root),
            prediction_files=_predictions(root),
            repo_root=_repo(root, policy="unknown"),
        )


def test_abstain_counts_are_deterministic_by_dataset_and_family():
    root = _clean()
    rows = {
        "core_eval": [
            _row(row_id="a", dataset="core_eval", family="custom_numeral"),
            _row(row_id="b", dataset="core_eval", family="custom_numeral"),
            _row(row_id="c", dataset="core_eval", family="numeric_formula", expected="17", gold_answer="17"),
        ],
        "family_eval": [_row(row_id="d", dataset="family_eval", family="sequence_pattern")],
        "rule_holdout": [],
        "anti_leak": [_row(row_id="e", dataset="anti_leak", family="sequence_pattern")],
    }
    report = _report(root, policy="concrete_only", rows_by_dataset=rows)

    assert report["abstain_rows"]["total"] == 4
    assert report["abstain_rows"]["by_dataset"] == {"anti_leak": 1, "core_eval": 2, "family_eval": 1}
    assert report["abstain_rows"]["by_family"] == {"custom_numeral": 2, "sequence_pattern": 2}


def test_scale_package_and_submission_are_never_authorized():
    root = _clean()
    report = _report(root, policy="abstain_allowed")

    assert report["decision"]["v2a_150_authorized"] is False
    assert report["decision"]["package_authorized"] is False
    assert report["decision"]["submission_authorized"] is False


def test_no_0_93_and_no_0_95_evidence_remain_true():
    root = _clean()
    report = _report(root, policy="abstain_allowed")

    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True
    assert report["leaderboard_claim"] is False


def test_report_contains_example_abstain_rows():
    root = _clean()
    report = _report(root, policy="concrete_only")

    examples = report["abstain_rows"]["example_rows"]
    assert examples
    assert examples[0]["gold_answer"] == "ABSTAIN"
    assert {"dataset", "row_id", "family", "prompt", "raw_output"}.issubset(examples[0])

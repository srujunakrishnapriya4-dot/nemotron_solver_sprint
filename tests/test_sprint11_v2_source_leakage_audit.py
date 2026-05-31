from kaggle_anti086.data.v2_source_leakage_audit import build_source_leakage_audit
from kaggle_anti086.data.v2_corpus_leakage import prompt_hash
from kaggle_anti086.data.v2_corpus_io import write_jsonl_checked


def _direct(**overrides):
    row = {
        "id": "d1",
        "source_row_id": "train-src-1",
        "rule_id": "train-rule",
        "leakage_group": "train-lg",
        "prompt": "train prompt",
        "answer": "x",
        "family": "format_only",
        "subfamily": "normalization",
        "verification_status": "verified",
        "solver_name": "solver_ensemble",
        "metadata": {
            "source_row_id": "train-src-1",
            "source_rule_id": "train-source-rule",
            "source_leakage_group": "train-source-lg",
            "source_prompt_hash": prompt_hash("train prompt"),
            "expected_solver_behavior": "answer",
        },
    }
    row.update(overrides)
    return row


def _run(tmp_path, direct, evals):
    paths = {}
    for name, rows in {"direct": direct, "solver_corrected": [], "abstain_safety": [], "hard_negative": []}.items():
        path = tmp_path / f"{name}.jsonl"
        write_jsonl_checked(path, rows, field_name=name)
        paths[name] = path
    eval_paths = {}
    for name, rows in evals.items():
        path = tmp_path / f"{name}.jsonl"
        write_jsonl_checked(path, rows, field_name=name)
        eval_paths[name] = path
    return build_source_leakage_audit(paths, eval_paths)


def test_clean_audit_passes(tmp_path):
    report = _run(tmp_path, [_direct()], {"rule_holdout": [{"rule_id": "h"}], "anti_leak": [{"leakage_group": "a"}], "private_like": [], "family_hard": []})
    assert report["status"] == "PASS"


def test_source_rule_id_overlap_with_rule_holdout_fails(tmp_path):
    row = _direct(metadata={**_direct()["metadata"], "source_rule_id": "h"})
    report = _run(tmp_path, [row], {"rule_holdout": [{"rule_id": "h"}], "anti_leak": [], "private_like": [], "family_hard": []})
    assert report["status"] == "FAIL"
    assert report["source_rule_id_overlap_with_rule_holdout"] == 1


def test_source_leakage_group_overlap_with_anti_leak_fails(tmp_path):
    row = _direct(metadata={**_direct()["metadata"], "source_leakage_group": "a"})
    report = _run(tmp_path, [row], {"rule_holdout": [], "anti_leak": [{"leakage_group": "a"}], "private_like": [], "family_hard": []})
    assert report["status"] == "FAIL"


def test_source_prompt_hash_overlap_with_eval_fails(tmp_path):
    row = _direct(metadata={**_direct()["metadata"], "source_prompt_hash": prompt_hash("eval prompt")})
    report = _run(tmp_path, [row], {"rule_holdout": [], "anti_leak": [], "private_like": [{"prompt": "eval prompt"}], "family_hard": []})
    assert report["source_prompt_hash_overlap_with_eval"] == 1


def test_duplicate_source_row_id_and_abstain_source_fail(tmp_path):
    rows = [_direct(id="a"), _direct(id="b")]
    report = _run(tmp_path, rows, {"rule_holdout": [], "anti_leak": [], "private_like": [], "family_hard": []})
    assert report["direct_source_row_id_duplicates"] == 1
    abstain = _direct(metadata={**_direct()["metadata"], "expected_solver_behavior": "abstain"})
    report = _run(tmp_path, [abstain], {"rule_holdout": [], "anti_leak": [], "private_like": [], "family_hard": []})
    assert report["direct_rows_from_expected_abstain"] == 1


def test_unsupported_direct_family_fails(tmp_path):
    report = _run(tmp_path, [_direct(family="equation_operator")], {"rule_holdout": [], "anti_leak": [], "private_like": [], "family_hard": []})
    assert report["unsupported_direct_family_rows"] == 1

import json

from kaggle_anti086.eval.day5_readiness_summary import build_readiness_summary


def _write(tmp_path, name: str, payload: dict):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _paths(tmp_path, *, answerable=0.95, unsafe=0.0, semantic=True):
    base_report = {"quality_status": "PASS", "unsafe_answer_rate": unsafe}
    paths = {
        "private_like": _write(tmp_path, "private", base_report),
        "rule_holdout": _write(tmp_path, "holdout", base_report),
        "family_hard": _write(tmp_path, "hard", base_report),
        "anti_leak": _write(tmp_path, "leak", base_report),
        "private_like_answerable": _write(tmp_path, "private_answer", {"answerable_exact_match": answerable, "unsafe_answer_rate": unsafe}),
        "rule_holdout_answerable": _write(tmp_path, "holdout_answer", {"answerable_exact_match": answerable, "unsafe_answer_rate": unsafe}),
        "family_hard_answerable": _write(tmp_path, "hard_answer", {"answerable_exact_match": answerable, "unsafe_answer_rate": unsafe}),
        "manifest": _write(tmp_path, "manifest", {"status": "PASS"}),
        "weak_subfamily": _write(tmp_path, "weak", {"subfamilies": [{"family": "format_only", "day6_action": "eligible_for_v2_direct_answer"}]}),
        "v2_eligibility": _write(tmp_path, "v2", {"eligible_verified_answer_count": 1000, "eligible_abstain_safety_count": 200, "status": "PASS"}),
        "semantic_holdout": _write(tmp_path, "semantic", {"semantic_holdout_pass": semantic, "status": "PASS" if semantic else "FAIL"}),
    }
    return paths


def test_readiness_passes_when_all_gates_satisfied(tmp_path) -> None:
    summary = build_readiness_summary(_paths(tmp_path))
    assert summary["status"] == "PASS"
    assert summary["decision"] == "ALLOW_DAY6_CORPUS_BUILD"
    assert summary["training_allowed"] is False
    assert summary["packaging_allowed"] is False
    assert summary["submission_allowed"] is False


def test_readiness_missing_artifact_blocks(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths["semantic_holdout"] = tmp_path / "missing.json"
    summary = build_readiness_summary(paths)
    assert summary["status"] == "FAIL"
    assert summary["gates"]["semantic_holdout_pass"]["status"] == "MISSING"


def test_readiness_fails_low_answerable_exact_match(tmp_path) -> None:
    summary = build_readiness_summary(_paths(tmp_path, answerable=0.89))
    assert summary["status"] == "FAIL"
    assert summary["decision"] == "BLOCK_DAY6_CORPUS_BUILD"


def test_readiness_fails_unsafe_answer_rate(tmp_path) -> None:
    summary = build_readiness_summary(_paths(tmp_path, unsafe=0.001))
    assert summary["status"] == "FAIL"
    assert summary["gates"]["unsafe_answer_rate_max"]["status"] == "FAIL"


def test_readiness_fails_semantic_holdout_false(tmp_path) -> None:
    summary = build_readiness_summary(_paths(tmp_path, semantic=False))
    assert summary["status"] == "FAIL"
    assert summary["gates"]["semantic_holdout_pass"]["status"] == "FAIL"

import json

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.v2_eligibility import build_v2_eligibility_report


def _write_jsonl(tmp_path, name: str, rows: list[dict]):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_v2_eligibility_classifies_rows_exactly_once(tmp_path) -> None:
    rows = build_eval_rows("private_like", 96, 1105)
    preds = []
    for row in rows:
        behavior = row["metadata"]["expected_solver_behavior"]
        can_answer = row["family"] in {"roman_numeral", "unit_conversion", "numeric_formula", "gravity_numeric"} and behavior == "answer"
        preds.append(
            {
                "id": row["id"],
                "prediction": row["answer"] if can_answer else "",
                "correct": can_answer,
                "abstained": not can_answer,
                "behavior_correct": can_answer or behavior == "abstain",
            }
        )
    report = build_v2_eligibility_report({"private_like": (_write_jsonl(tmp_path, "eval", rows), _write_jsonl(tmp_path, "pred", preds))})
    assert report["total_rows_seen"] == len(rows)
    assert sum(report["classification_counts"].values()) == len(rows)
    assert report["eligible_verified_answer_count"] > 0
    assert "blocked_solver_failure" in report["classification_counts"]
    assert report["day6_decision"] in {"ALLOW_CORPUS_BUILD", "BLOCK_CORPUS_BUILD"}


def test_v2_eligibility_blocks_duplicate_semantic_signature(tmp_path) -> None:
    rows = build_eval_rows("private_like", 16, 1)
    rows[1]["metadata"]["rule_signature"] = rows[0]["metadata"]["rule_signature"]
    preds = [{"id": row["id"], "correct": True, "abstained": False, "prediction": row["answer"]} for row in rows]
    report = build_v2_eligibility_report({"private_like": (_write_jsonl(tmp_path, "eval", rows), _write_jsonl(tmp_path, "pred", preds))})
    assert report["status"] == "FAIL"
    assert report["blocked_counts"]["blocked_leakage_risk"] >= 2

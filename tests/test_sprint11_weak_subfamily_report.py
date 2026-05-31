import json

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.weak_subfamily_report import build_weak_subfamily_report


def _write_jsonl(tmp_path, name: str, rows: list[dict]):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_weak_subfamily_report_uses_predictions_not_static_names(tmp_path) -> None:
    rows = [row for row in build_eval_rows("private_like", 96, 5) if row["family"] in {"roman_numeral", "symbol_mapping", "format_only"}]
    preds = []
    for row in rows:
        behavior = row["metadata"]["expected_solver_behavior"]
        correct = row["family"] == "roman_numeral" and behavior == "answer"
        abstained = row["family"] != "roman_numeral"
        preds.append(
            {
                "id": row["id"],
                "prediction": row["answer"] if correct else "",
                "correct": correct,
                "abstained": abstained,
                "behavior_correct": correct or behavior == "abstain",
                "reason": "all_solvers_abstained" if abstained else "",
            }
        )
    report = build_weak_subfamily_report({"private_like": (_write_jsonl(tmp_path, "eval", rows), _write_jsonl(tmp_path, "pred", preds))})
    roman = [row for row in report["subfamilies"] if row["family"] == "roman_numeral"][0]
    assert roman["day6_action"] == "eligible_for_v2_direct_answer"
    format_rows = [row for row in report["subfamilies"] if row["family"] == "format_only"]
    assert not format_rows or format_rows[0]["day6_action"] in {"eligible_for_abstain_safety", "blocked_unsupported_family"}
    assert report["status"] == "PASS"

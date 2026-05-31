import json

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.v2_eligibility import PER_FAMILY_THRESHOLDS, build_v2_eligibility_report


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
    assert report["schema_version"] == 2
    assert report["training_allowed"] is False
    assert report["day6_decision"] == "BLOCK_CORPUS_BUILD"
    assert "eligible_verified_answer_count_below_threshold" in report["remaining_blockers"]


def test_v2_eligibility_blocks_duplicate_semantic_signature(tmp_path) -> None:
    rows = build_eval_rows("private_like", 16, 1)
    rows[1]["metadata"]["rule_signature"] = rows[0]["metadata"]["rule_signature"]
    preds = [{"id": row["id"], "correct": True, "abstained": False, "prediction": row["answer"]} for row in rows]
    report = build_v2_eligibility_report({"private_like": (_write_jsonl(tmp_path, "eval", rows), _write_jsonl(tmp_path, "pred", preds))})
    assert report["status"] == "FAIL"
    assert report["blocked_counts"]["blocked_leakage_risk"] >= 2


def test_v2_eligibility_passes_with_family_thresholds(tmp_path) -> None:
    rows = []
    preds = []
    idx = 0
    for family, count in PER_FAMILY_THRESHOLDS.items():
        for n in range(count):
            row = {
                "id": f"{family}_{n}",
                "family": family,
                "subfamily": "sub",
                "rule_id": f"rule_{family}_{n}",
                "prompt": f"prompt {family} {n}",
                "answer": "42" if family in {"unit_conversion", "numeric_formula", "gravity_numeric"} else "ok",
                "source": "test",
                "solver_name": "solver_ensemble",
                "verification_status": "verified",
                "difficulty": 1,
                "split": "private_like_eval",
                "leakage_group": f"lg_{family}_{n}",
                "metadata": {"expected_solver_behavior": "answer", "rule_signature": f"sig_{family}_{n}", "generator_id": "test"},
            }
            rows.append(row)
            preds.append({"id": row["id"], "prediction": row["answer"], "correct": True, "abstained": False})
            idx += 1
    for n in range(300):
        row = {
            "id": f"extra_bit_{n}",
            "family": "bit_manipulation",
            "subfamily": "xor_mask",
            "rule_id": f"extra_rule_bit_{n}",
            "prompt": f"extra bit prompt {n}",
            "answer": "0101",
            "source": "test",
            "solver_name": "solver_ensemble",
            "verification_status": "verified",
            "difficulty": 1,
            "split": "private_like_eval",
            "leakage_group": f"extra_lg_bit_{n}",
            "metadata": {"expected_solver_behavior": "answer", "rule_signature": f"extra_sig_bit_{n}", "generator_id": "test"},
        }
        rows.append(row)
        preds.append({"id": row["id"], "prediction": row["answer"], "correct": True, "abstained": False})
    for n in range(200):
        row = {
            "id": f"abstain_{n}",
            "family": "equation_operator",
            "subfamily": "unsupported",
            "rule_id": f"abstain_rule_{n}",
            "prompt": f"unsupported {n}",
            "answer": "ABSTAIN",
            "source": "test",
            "solver_name": "solver_ensemble",
            "verification_status": "verified",
            "difficulty": 3,
            "split": "anti_leak_eval",
            "leakage_group": f"abstain_lg_{n}",
            "metadata": {"expected_solver_behavior": "abstain", "rule_signature": f"abstain_sig_{n}", "generator_id": "test"},
        }
        rows.append(row)
        preds.append({"id": row["id"], "prediction": "", "correct": False, "abstained": True})
    report = build_v2_eligibility_report({"private_like": (_write_jsonl(tmp_path, "eval_pass", rows), _write_jsonl(tmp_path, "pred_pass", preds))})
    assert report["status"] == "PASS"
    assert report["day6_decision"] == "ALLOW_CORPUS_BUILD"


def test_v2_eligibility_blocks_missing_metadata(tmp_path) -> None:
    rows = build_eval_rows("private_like", 16, 5)
    del rows[0]["metadata"]["generator_id"]
    preds = [{"id": row["id"], "correct": True, "abstained": False, "prediction": row["answer"]} for row in rows]
    report = build_v2_eligibility_report({"private_like": (_write_jsonl(tmp_path, "eval_missing", rows), _write_jsonl(tmp_path, "pred_missing", preds))})
    assert report["status"] == "FAIL"
    assert report["blocked_counts"]["blocked_missing_metadata"] >= 1

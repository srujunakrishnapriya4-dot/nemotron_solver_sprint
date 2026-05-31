from pathlib import Path

from kaggle_anti086.data.build_train_v2_corpus import build_train_v2_corpus
from kaggle_anti086.data.v2_corpus_io import read_json, read_jsonl


ROOT = Path("artifacts/sprint11")


def test_builder_writes_all_corpus_files_from_day5_artifacts(tmp_path):
    summary = build_train_v2_corpus(
        eligibility_report_path=ROOT / "day5_v2_eligibility_report.json",
        readiness_summary_path=ROOT / "day5_2_readiness_summary.json",
        eval_paths={
            "private_like": ROOT / "day5_private_like_eval_512.jsonl",
            "rule_holdout": ROOT / "day5_rule_holdout_eval_512.jsonl",
            "family_hard": ROOT / "day5_family_hard_eval_512.jsonl",
            "anti_leak": ROOT / "day5_anti_leak_eval_256.jsonl",
        },
        prediction_paths={
            "private_like": ROOT / "day5_private_like_solver_predictions.jsonl",
            "rule_holdout": ROOT / "day5_rule_holdout_solver_predictions.jsonl",
            "family_hard": ROOT / "day5_family_hard_solver_predictions.jsonl",
            "anti_leak": ROOT / "day5_anti_leak_solver_predictions.jsonl",
        },
        out_dir=tmp_path,
    )
    assert summary["status"] == "PASS"
    direct = read_jsonl(tmp_path / "train_v2_verified_direct_answer.jsonl")
    abstain = read_jsonl(tmp_path / "train_v2_abstain_safety.jsonl")
    hard_negative = read_jsonl(tmp_path / "train_v2_hard_negative.jsonl")
    assert len(direct) >= 1000
    assert len(abstain) >= 200
    assert len(hard_negative) >= 20
    assert (tmp_path / "train_v2_solver_corrected.jsonl").exists()
    assert (tmp_path / "train_v2_manifest.json").exists()


def test_direct_answer_corpus_is_eligible_only(tmp_path):
    build_train_v2_corpus(
        eligibility_report_path=ROOT / "day5_v2_eligibility_report.json",
        readiness_summary_path=ROOT / "day5_2_readiness_summary.json",
        eval_paths={
            "private_like": ROOT / "day5_private_like_eval_512.jsonl",
            "rule_holdout": ROOT / "day5_rule_holdout_eval_512.jsonl",
            "family_hard": ROOT / "day5_family_hard_eval_512.jsonl",
            "anti_leak": ROOT / "day5_anti_leak_eval_256.jsonl",
        },
        prediction_paths={
            "private_like": ROOT / "day5_private_like_solver_predictions.jsonl",
            "rule_holdout": ROOT / "day5_rule_holdout_solver_predictions.jsonl",
            "family_hard": ROOT / "day5_family_hard_solver_predictions.jsonl",
            "anti_leak": ROOT / "day5_anti_leak_solver_predictions.jsonl",
        },
        out_dir=tmp_path,
    )
    direct = read_jsonl(tmp_path / "train_v2_verified_direct_answer.jsonl")
    hard_negative = read_jsonl(tmp_path / "train_v2_hard_negative.jsonl")
    assert all(row["metadata"]["source_classification"] == "eligible_verified_answer" for row in direct)
    assert all(row["family"] not in {"equation_operator", "sequence_pattern", "permutation_sorting", "custom_numeral"} for row in direct)
    assert all(row["training_allowed"] is True for row in direct)
    assert all(row["loss_scope"] == "assistant_only" for row in direct)
    assert all(row.get("do_not_use_as_sft") is True for row in hard_negative)
    manifest = read_json(tmp_path / "train_v2_manifest.json")
    assert manifest["training_allowed"] is False

from __future__ import annotations

import json

from scripts.replay_selective_runtime import run_replay


def test_replay_selective_runtime_emits_deterministic_artifacts(tmp_path) -> None:
    replay_path = tmp_path / "replay.jsonl"
    output_dir = tmp_path / "out"
    replay_rows = [
        {
            "problem_id": "p1",
            "problem": "Find n.",
            "expected_answer": "42",
            "route": {
                "difficulty": "easy",
                "difficulty_score": 0.20,
                "route_uncertainty": 0.10,
                "problem_type_probs": {"number_theory": 0.8, "algebra": 0.2},
                "compute_signals": {"proof_burden": 0.10, "retrieval_need": 0.05, "repair_need": 0.05},
            },
            "attempt_batch_stopped_reason": "early_stop_consensus",
            "attempts": [
                {"attempt_id": "a0", "seed": 0, "raw_text": "\\boxed{42}", "extracted_answer": "42", "valid_answer": True, "mean_token_entropy": 0.20, "entropy_source": "true_logprobs", "runtime_sec": 0.1},
                {"attempt_id": "a1", "seed": 1, "raw_text": "\\boxed{42}", "extracted_answer": "42", "valid_answer": True, "mean_token_entropy": 0.18, "entropy_source": "true_logprobs", "runtime_sec": 0.1},
                {"attempt_id": "a2", "seed": 2, "raw_text": "\\boxed{42}", "extracted_answer": "42", "valid_answer": True, "mean_token_entropy": 0.19, "entropy_source": "true_logprobs", "runtime_sec": 0.1},
                {"attempt_id": "a3", "seed": 3, "raw_text": "\\boxed{42}", "extracted_answer": "42", "valid_answer": True, "mean_token_entropy": 0.17, "entropy_source": "true_logprobs", "runtime_sec": 0.1},
            ],
            "full_prediction": {"final_answer": 42, "confidence": 0.9},
        },
        {
            "problem_id": "p2",
            "problem": "Find m.",
            "expected_answer": "17",
            "route": {
                "difficulty": "easy",
                "difficulty_score": 0.25,
                "route_uncertainty": 0.12,
                "problem_type_probs": {"number_theory": 0.7, "algebra": 0.3},
                "compute_signals": {"proof_burden": 0.12, "retrieval_need": 0.08, "repair_need": 0.06},
            },
            "attempt_batch_stopped_reason": "attempt_budget_exhausted",
            "attempts": [
                {"attempt_id": "b0", "seed": 0, "raw_text": "\\boxed{42}", "extracted_answer": "42", "valid_answer": True, "mean_token_entropy": 0.70, "entropy_source": "true_logprobs", "runtime_sec": 0.1},
                {"attempt_id": "b1", "seed": 1, "raw_text": "\\boxed{17}", "extracted_answer": "17", "valid_answer": True, "mean_token_entropy": 1.30, "entropy_source": "true_logprobs", "runtime_sec": 0.1},
                {"attempt_id": "b2", "seed": 2, "raw_text": "\\boxed{19}", "extracted_answer": "19", "valid_answer": True, "mean_token_entropy": 1.35, "entropy_source": "true_logprobs", "runtime_sec": 0.1},
            ],
            "escalated_prediction": {"final_answer": 17, "confidence": 0.74},
            "full_prediction": {"final_answer": 17, "confidence": 0.78},
        },
    ]
    replay_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in replay_rows) + "\n",
        encoding="utf-8",
    )

    artifact_paths = run_replay(input_path=replay_path, output_dir=output_dir)

    replay_results = json.loads((output_dir / "replay_results.json").read_text(encoding="utf-8"))
    per_problem_rows = [
        json.loads(line)
        for line in (output_dir / "per_problem_modes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    escalation_summary = json.loads((output_dir / "escalation_helped_vs_hurt.json").read_text(encoding="utf-8"))

    assert artifact_paths["replay_results_path"].endswith("replay_results.json")
    assert replay_results["problem_count"] == 2
    assert replay_results["gate_thresholds"]["escalation_min_confidence"] == 0.66
    assert replay_results["modes"]["minimal_only"]["correct_count"] == 1
    assert replay_results["modes"]["selective_escalation"]["correct_count"] == 2
    assert escalation_summary["helped"] == 1
    assert escalation_summary["hurt"] == 0
    assert escalation_summary["reason_counts"]["low_confidence"] >= 1
    assert [row["problem_id"] for row in per_problem_rows] == ["p1", "p2"]
    assert per_problem_rows[1]["selective_escalation"]["escalation_used"] is True
    assert "high_true_entropy" in per_problem_rows[1]["selective_escalation"]["escalation_reasons"]


def test_replay_selective_runtime_accepts_minimal_prediction_rows(tmp_path) -> None:
    replay_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "out"
    replay_rows = [
        {
            "problem_id": "p1",
            "problem": "Find the non-negative integer answer.",
            "predicted_answer": "42",
            "gold_answer": "42",
            "metadata": {},
        },
        {
            "id": "p2",
            "question": "Find the non-negative integer answer.",
            "prediction": "17",
            "gold": "19",
            "metadata": {},
        },
    ]
    replay_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in replay_rows) + "\n",
        encoding="utf-8",
    )

    run_replay(input_path=replay_path, output_dir=output_dir)

    replay_results = json.loads((output_dir / "replay_results.json").read_text(encoding="utf-8"))
    per_problem_rows = [
        json.loads(line)
        for line in (output_dir / "per_problem_modes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert replay_results["problem_count"] == 2
    assert replay_results["modes"]["minimal_only"]["labeled_problem_count"] == 2
    assert per_problem_rows[0]["minimal_only"]["predicted_answer"] == "42"
    assert per_problem_rows[1]["minimal_only"]["predicted_answer"] == "17"
    assert per_problem_rows[0]["minimal_only"]["metadata"]["selection_metadata"]["valid_attempt_count"] == 1

from __future__ import annotations

from collections import Counter
import csv
import json
import re
from pathlib import Path
from typing import Any

from .bit_composition_solver import solve_bit_composition_problem
from .cipher_symbol_solver import solve_cipher_symbol_problem
from .competition_answer_policy import CompetitionAnswerPolicyError, validate_competition_answer
from .competition_prompt_adapter import CompetitionPromptAdapterError, parse_competition_prompt
from .competition_runner import _build_cipher_visible_vocabulary, _solve_bit_expression_synth, _solve_problem
from .hard_family_router import normalize_family_name
from .equation_symbolic_solver import solve_equation_symbolic_problem
from .gravity_unit_solver import solve_gravity_unit_problem


SOLVERS = {
    "bit_manipulation": solve_bit_composition_problem,
    "equation_symbolic": solve_equation_symbolic_problem,
    "unit_conversion": solve_gravity_unit_problem,
    "gravity_numeric": solve_gravity_unit_problem,
    "cipher_text": solve_cipher_symbol_problem,
}


def build_solver_coverage_report(
    train_csv_path: str | Path = "data/nemotron_competition/train.csv",
    output_dir: str | Path = "artifacts/win_system",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    correct: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    problems, parse_failures = _load_problems_with_diagnostics(Path(train_csv_path))
    vocab = _build_cipher_visible_vocabulary(Path(train_csv_path), include_sibling_train=False) if Path(train_csv_path).exists() else None
    routed_by_family = Counter()
    attempted_by_family = Counter()
    normalization_mismatches: Counter[str] = Counter()
    for problem in problems:
        family = normalize_family_name(problem.family)
        routed_by_family[family] += 1
        solver = SOLVERS.get(family)
        examples = tuple((ex.input, ex.output) for ex in problem.examples)
        hyp = None
        if not solver:
            prediction, reason, diagnostics = _solve_problem(problem, cipher_vocabulary=vocab)
            solver_name = "competition_runner"
        else:
            attempted_by_family[family] += 1
            hyp = solver(examples, problem.target_input)
            if hyp.verified and hyp.prediction is not None:
                prediction, reason, diagnostics, solver_name = hyp.prediction, hyp.rule_id, hyp.metadata, hyp.solver_name
            else:
                if family == "bit_manipulation":
                    prediction, reason = _solve_bit_expression_synth(problem, max_candidates=20000)
                    diagnostics = dict(hyp.metadata)
                    diagnostics.update({"coverage_budget": 20000, "primary_reason": hyp.reason, "fallback_reason": reason})
                else:
                    prediction, reason, diagnostics = _solve_problem(problem, cipher_vocabulary=vocab)
                solver_name = "competition_runner_fallback"
        if prediction is None:
            abstained.append(_row(problem, reason, None, solver_name=solver_name, diagnostics=diagnostics))
        elif _answers_match(prediction, problem.answer, family, normalization_mismatches):
            correct.append(_row(problem, reason, prediction, solver_name=solver_name, diagnostics=diagnostics))
        else:
            wrong.append(_row(problem, "answer_mismatch", prediction, solver_name=solver_name, diagnostics=diagnostics))
    _write_jsonl(out / "solver_verified_correct.jsonl", correct)
    _write_jsonl(out / "solver_verified_wrong.jsonl", wrong)
    _write_jsonl(out / "solver_abstained.jsonl", abstained)
    equation_failures = _failure_samples(problems, abstained + wrong, "equation_symbolic", limit=2000)
    _write_jsonl(out / "failure_samples_equation_symbolic.jsonl", equation_failures)
    _write_jsonl(out / "failure_samples_bit_budget.jsonl", [row for row in _failure_samples(problems, abstained, "bit_manipulation", limit=300) if row["failure_reason"] == "bit_candidate_budget_exhausted"])
    _write_jsonl(out / "failure_samples_numeric_precision.jsonl", [row for row in _failure_samples(problems, abstained + wrong, None, limit=400) if row["failure_reason"] in {"inconsistent_numeric_precision", "answer_mismatch"} and row["family"] in {"unit_conversion", "gravity_numeric"}])
    equation_taxonomy, equation_clusters = _equation_failure_taxonomy(equation_failures)
    (out / "equation_failure_taxonomy.json").write_text(json.dumps(equation_taxonomy, sort_keys=True, indent=2), encoding="utf-8")
    _write_jsonl(out / "equation_failure_clusters.jsonl", equation_clusters)
    _write_equation_cluster_examples(out / "equation_cluster_examples", equation_clusters)
    digit_symbol_taxonomy, digit_symbol_examples = _digit_symbol_failure_taxonomy(equation_clusters)
    (out / "digit_symbol_failure_taxonomy.json").write_text(json.dumps(digit_symbol_taxonomy, sort_keys=True, indent=2), encoding="utf-8")
    _write_jsonl(out / "digit_symbol_failure_examples.jsonl", digit_symbol_examples)
    equation_program_verified = [row for row in correct if row["family"] == "equation_symbolic"]
    equation_program_wrong = [row for row in wrong if row["family"] == "equation_symbolic"]
    equation_program_abstained = [row for row in abstained if row["family"] == "equation_symbolic"]
    _write_jsonl(out / "equation_program_verified.jsonl", [_equation_program_row(row, "verified") for row in equation_program_verified])
    _write_jsonl(out / "equation_program_wrong.jsonl", [_equation_program_row(row, "wrong") for row in equation_program_wrong])
    _write_jsonl(out / "equation_program_abstained.jsonl", [_equation_program_row(row, "abstained") for row in equation_program_abstained])
    eq_correct = Counter(row["family"] for row in correct).get("equation_symbolic", 0)
    bit_budget = Counter(row["reason"] for row in abstained).get("bit_candidate_budget_exhausted", 0)
    gravity_wrong = Counter(row["family"] for row in wrong).get("gravity_numeric", 0)
    unit_wrong = Counter(row["family"] for row in wrong).get("unit_conversion", 0)
    inconsistent_precision = Counter(row["reason"] for row in abstained).get("inconsistent_numeric_precision", 0)
    unsupported_equation = Counter(row["reason"] for row in abstained).get("unsupported_equation_transform", 0)
    equation_wrong = Counter(row["family"] for row in wrong).get("equation_symbolic", 0)
    equation_disagreement = Counter(row["reason"] for row in abstained).get("ambiguous_transform", 0) + Counter(row["reason"] for row in abstained).get("ambiguous_operator", 0)
    by_id = {problem.problem_id: problem for problem in problems}
    equation_solved_clusters = Counter(_equation_cluster_for_problem(by_id[row["id"]]) for row in equation_program_verified if row["id"] in by_id)
    ambiguity_distribution = Counter(str(row.get("diagnostics", {}).get("ambiguity_count", 0)) for row in equation_program_verified + equation_program_wrong)
    candidate_distribution = Counter(str(row.get("diagnostics", {}).get("candidate_count", "unknown")) for row in equation_program_verified + equation_program_wrong + equation_program_abstained)
    bottleneck_taxonomy = {}
    if eq_correct < 200:
        bottleneck_taxonomy["equation_symbolic"] = dict(Counter(row["reason"] for row in abstained + wrong if row["family"] == "equation_symbolic").most_common(10))
    if bit_budget > int(1096 * 0.6):
        bottleneck_taxonomy["bit_manipulation"] = {"bit_candidate_budget_exhausted": bit_budget, "note": "remaining rows require deeper canonical bit synthesis or more aggressive vector pruning"}
    if inconsistent_precision:
        bottleneck_taxonomy["numeric_precision"] = {"inconsistent_numeric_precision": inconsistent_precision, "note": "precision varies across examples; safe solver abstains instead of tolerating unknown rounding"}
    sprint74_success = (eq_correct >= 200 or (eq_correct - 45 >= 100 and "equation_symbolic" in bottleneck_taxonomy)) and equation_wrong <= 13 and unsupported_equation < 1481
    sprint75_success = (eq_correct >= 120 or eq_correct - 40 >= 75) and equation_wrong <= 12 and unsupported_equation < 1485 and equation_solved_clusters.get("digit_symbol_arithmetic", 0) > 0 and equation_solved_clusters.get("binary_operator_arithmetic", 0) > 0
    equation_program_report = {
        "equation_symbolic_verified_correct": eq_correct,
        "equation_symbolic_verified_wrong": equation_wrong,
        "equation_symbolic_disagreement": equation_disagreement,
        "unsupported_equation_transform": unsupported_equation,
        "digit_symbol_arithmetic_solved_count": equation_solved_clusters.get("digit_symbol_arithmetic", 0),
        "binary_operator_arithmetic_solved_count": equation_solved_clusters.get("binary_operator_arithmetic", 0),
        "ambiguity_count_distribution": dict(ambiguity_distribution.most_common()),
        "candidate_count_distribution": dict(candidate_distribution.most_common(20)),
        "verified_program_count": len(equation_program_verified),
        "wrong_program_count": len(equation_program_wrong),
        "abstained_program_count": len(equation_program_abstained),
        "cluster_counts": equation_taxonomy["cluster_counts"],
        "acceptance": {
            "baseline_equation_symbolic_verified_correct": 45,
            "equation_symbolic_increase": eq_correct - 45,
            "wrong_increase_limit": 13,
            "success": sprint74_success,
            "verdict": "SPRINT-7.4_SUCCESS_EQUATION_PROGRAM_SYNTHESIS" if sprint74_success else "SPRINT-7.4_FAIL_EQUATION_PROGRAM_SYNTHESIS",
        },
        "sprint75_acceptance": {
            "baseline_equation_symbolic_verified_correct": 40,
            "equation_symbolic_increase": eq_correct - 40,
            "digit_symbol_arithmetic_solved_count": equation_solved_clusters.get("digit_symbol_arithmetic", 0),
            "binary_operator_arithmetic_solved_count": equation_solved_clusters.get("binary_operator_arithmetic", 0),
            "success": sprint75_success,
            "verdict": "SPRINT-7.5_SUCCESS_DIGIT_SYMBOL_INDUCTION" if sprint75_success else "SPRINT-7.5_FAIL_DIGIT_SYMBOL_INDUCTION",
        },
    }
    (out / "equation_program_synthesis_report.json").write_text(json.dumps(equation_program_report, sort_keys=True, indent=2), encoding="utf-8")
    bit_program_verified = [row for row in correct if row["family"] == "bit_manipulation"]
    bit_program_wrong = [row for row in wrong if row["family"] == "bit_manipulation"]
    bit_program_abstained = [row for row in abstained if row["family"] == "bit_manipulation"]
    bit_failure_samples = _failure_samples(problems, bit_program_abstained + bit_program_wrong, "bit_manipulation", limit=2000)
    bit_budget_samples = [row for row in bit_failure_samples if row["failure_reason"] == "bit_candidate_budget_exhausted"]
    bit_taxonomy, bit_clusters = _bit_failure_taxonomy(bit_budget_samples)
    _write_jsonl(out / "bit_program_verified.jsonl", [_bit_program_row(row, "verified") for row in bit_program_verified])
    _write_jsonl(out / "bit_program_wrong.jsonl", [_bit_program_row(row, "wrong") for row in bit_program_wrong])
    _write_jsonl(out / "bit_program_abstained.jsonl", [_bit_program_row(row, "abstained") for row in bit_program_abstained])
    _write_jsonl(out / "bit_failure_clusters.jsonl", bit_clusters)
    _write_jsonl(out / "bit_budget_samples.jsonl", bit_budget_samples[:300])
    _write_jsonl(out / "bit_truth_table_mining_samples.jsonl", [row for row in bit_clusters if row["cluster"] in {"majority_choice_composition", "mask_or_threshold_like"}])
    (out / "bit_failure_taxonomy.json").write_text(json.dumps(bit_taxonomy, sort_keys=True, indent=2), encoding="utf-8")
    sprint72_success = (eq_correct >= 200 or (eq_correct >= 102 and "equation_symbolic" in bottleneck_taxonomy)) and (bit_budget <= int(1096 * 0.6) or "bit_manipulation" in bottleneck_taxonomy)
    bit_correct = Counter(row["family"] for row in correct).get("bit_manipulation", 0)
    bit_wrong = Counter(row["family"] for row in wrong).get("bit_manipulation", 0)
    bit_disagreement = sum(1 for row in abstained if row["family"] == "bit_manipulation" and row["reason"] in {"bit_expression_disagreement", "ambiguous_fits"})
    bit_depth_distribution: Counter[str] = Counter()
    bit_candidate_counts: Counter[str] = Counter()
    pruned_duplicate_total = 0
    pruned_failed_total = 0
    for row in bit_program_verified + bit_program_wrong + bit_program_abstained:
        diagnostics = row.get("diagnostics", {}) if isinstance(row.get("diagnostics"), dict) else {}
        bit_depth_distribution.update({str(key): int(value) for key, value in diagnostics.get("depth_solved_distribution", {}).items() if isinstance(value, int)})
        if "candidate_count" in diagnostics or "tested" in diagnostics:
            bit_candidate_counts[str(diagnostics.get("candidate_count", diagnostics.get("tested")))] += 1
        pruned_duplicate_total += int(diagnostics.get("pruned_duplicate_count", 0) or 0)
        pruned_failed_total += int(diagnostics.get("pruned_failed_count", 0) or 0)
    truth_table_solved = sum(1 for row in bit_program_verified if row.get("reason") == "truth_table_bit_mining")
    truth_table_wrong = sum(1 for row in bit_program_wrong if isinstance(row.get("diagnostics"), dict) and row.get("diagnostics", {}).get("expr_id") == "truth_table_bit_mining")
    sprint77_success = (
        (bit_budget <= 530 or bit_correct - 707 >= 250)
        and bit_wrong <= 11
        and bit_disagreement <= 10
        and bool(bit_depth_distribution or pruned_duplicate_total or pruned_failed_total)
    )
    sprint78_success = (
        (bit_budget <= 530 or bit_correct - 772 >= 200)
        and bit_wrong <= 14
        and bit_disagreement <= 10
        and bit_taxonomy["cluster_counts"].get("majority_choice_composition", 0) < 735
        and bool(bit_depth_distribution or pruned_duplicate_total or pruned_failed_total)
    )
    phase1_status = (
        "PHASE1_COMPLETE_GPU_MICRO_ALLOWED"
        if sprint78_success
        else "PHASE1_PARTIAL_GPU_MICRO_STACK_TEST_ONLY"
        if bit_correct > 772 and bit_wrong <= 14 and bit_disagreement <= 10
        else "PHASE1_INCOMPLETE_GPU_BLOCKED"
    )
    bit_program_report = {
        "baseline_bit_verified_correct": 707,
        "baseline_bit_candidate_budget_exhausted": 884,
        "baseline_bit_wrong": 6,
        "baseline_bit_expression_disagreement": 5,
        "bit_verified_correct": bit_correct,
        "bit_verified_correct_increase": bit_correct - 707,
        "bit_candidate_budget_exhausted": bit_budget,
        "bit_wrong": bit_wrong,
        "bit_expression_disagreement": bit_disagreement,
        "truth_table_miner_solved_count": truth_table_solved,
        "truth_table_miner_wrong_count": truth_table_wrong,
        "depth_solved_distribution": dict(bit_depth_distribution.most_common()),
        "candidate_count_distribution": dict(bit_candidate_counts.most_common(20)),
        "candidate_pruning_stats": {
            "pruned_duplicate_total": pruned_duplicate_total,
            "pruned_failed_total": pruned_failed_total,
        },
        "failure_taxonomy": bit_taxonomy,
        "acceptance": {
            "success": sprint77_success,
            "verdict": "SPRINT-7.7_SUCCESS_BIT_BREAKTHROUGH" if sprint77_success else "SPRINT-7.7_FAIL_BIT_BREAKTHROUGH",
            "budget_target": "<=530",
            "verified_increase_target": ">=250",
        },
        "sprint78_acceptance": {
            "baseline_bit_verified_correct": 772,
            "baseline_bit_candidate_budget_exhausted": 818,
            "bit_verified_correct_increase": bit_correct - 772,
            "success": sprint78_success,
            "verdict": "SPRINT-7.8_SUCCESS_BIT_TRUTH_TABLE_MINING" if sprint78_success else "SPRINT-7.8_FAIL_BIT_TRUTH_TABLE_MINING",
            "phase1_status": phase1_status,
        },
    }
    (out / "bit_program_synthesis_report.json").write_text(json.dumps(bit_program_report, sort_keys=True, indent=2), encoding="utf-8")
    truth_report = {
        "truth_table_miner_solved_count": truth_table_solved,
        "truth_table_miner_wrong_count": truth_table_wrong,
        "majority_choice_composition_remaining": bit_taxonomy["cluster_counts"].get("majority_choice_composition", 0),
        "mask_or_threshold_like_remaining": bit_taxonomy["cluster_counts"].get("mask_or_threshold_like", 0),
        "candidate_pruning_stats": bit_program_report["candidate_pruning_stats"],
        "acceptance": bit_program_report["sprint78_acceptance"],
    }
    (out / "bit_truth_table_mining_report.json").write_text(json.dumps(truth_report, sort_keys=True, indent=2), encoding="utf-8")
    (out / "bit_truth_table_cluster_report.json").write_text(json.dumps({"cluster_counts": bit_taxonomy["cluster_counts"], "target_clusters": {key: bit_taxonomy["cluster_counts"].get(key, 0) for key in ["majority_choice_composition", "mask_or_threshold_like"]}}, sort_keys=True, indent=2), encoding="utf-8")
    report = {
        "verified_correct_by_family": dict(Counter(row["family"] for row in correct)),
        "verified_wrong_by_family": dict(Counter(row["family"] for row in wrong)),
        "abstained_by_family": dict(Counter(row["family"] for row in abstained)),
        "parsed_count": len(problems),
        "parse_failure_count": len(parse_failures),
        "routed_by_family": dict(routed_by_family),
        "solver_attempted_by_family": dict(attempted_by_family),
        "solver_abstained_by_family": dict(Counter(row["family"] for row in abstained)),
        "solver_wrong_by_family": dict(Counter(row["family"] for row in wrong)),
        "solver_verified_correct_by_family": dict(Counter(row["family"] for row in correct)),
        "top_parse_failure_reasons": dict(Counter(row["reason"] for row in parse_failures).most_common(10)),
        "top_abstention_reasons": dict(Counter(row["reason"] for row in abstained).most_common(10)),
        "top_answer_normalization_mismatch_reasons": dict(normalization_mismatches.most_common(10)),
        "new_coverage_vs_sprint6": len(correct),
        "top_remaining_failure_reasons": dict(Counter(row["reason"] for row in abstained).most_common(10)),
        "hard_family_bottleneck_taxonomy": bottleneck_taxonomy,
        "equation_failure_taxonomy": equation_taxonomy,
        "sprint72_acceptance": {
            "equation_symbolic_verified_correct": eq_correct,
            "bit_candidate_budget_exhausted": bit_budget,
            "inconsistent_numeric_precision": inconsistent_precision,
            "gravity_wrong": gravity_wrong,
            "unit_wrong": unit_wrong,
            "success_or_documented_bottleneck": sprint72_success,
        },
        "sprint73_acceptance": {
            "baseline_equation_symbolic_verified_correct": 5,
            "equation_symbolic_verified_correct": eq_correct,
            "equation_symbolic_increase": eq_correct - 5,
            "unsupported_equation_transform": unsupported_equation,
            "equation_symbolic_wrong": equation_wrong,
            "equation_symbolic_disagreement": equation_disagreement,
            "success": eq_correct >= 200 or (eq_correct - 5 >= 100 and "equation_symbolic" in bottleneck_taxonomy),
            "verdict": "SPRINT-7.3_SUCCESS" if (eq_correct >= 200 or (eq_correct - 5 >= 100 and "equation_symbolic" in bottleneck_taxonomy)) else "SPRINT-7.3_FAIL_EQUATION_SYMBOLIC",
        },
        "sprint74_acceptance": equation_program_report["acceptance"],
        "sprint75_acceptance": equation_program_report["sprint75_acceptance"],
        "sprint77_acceptance": bit_program_report["acceptance"],
        "sprint78_acceptance": bit_program_report["sprint78_acceptance"],
        "phase1_status": phase1_status,
        "bit_program_synthesis": {
            "bit_verified_correct": bit_correct,
            "bit_candidate_budget_exhausted": bit_budget,
            "bit_wrong": bit_wrong,
            "bit_expression_disagreement": bit_disagreement,
            "depth_solved_distribution": bit_program_report["depth_solved_distribution"],
            "candidate_pruning_stats": bit_program_report["candidate_pruning_stats"],
            "truth_table_miner_solved_count": truth_table_solved,
        },
    }
    (out / "solver_coverage_report.json").write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def _load_problems_with_diagnostics(path: Path):
    problems = []
    failures = []
    if not path.exists():
        return problems, [{"id": None, "reason": "missing_train_csv"}]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or ()) or "prompt" not in (reader.fieldnames or ()):
            return problems, [{"id": None, "reason": "missing_id_or_prompt_column"}]
        for index, row in enumerate(reader, start=1):
            try:
                problems.append(parse_competition_prompt(str(row["id"]), str(row["prompt"]), row.get("answer")))
            except Exception as exc:
                failures.append({"id": row.get("id", str(index)), "reason": type(exc).__name__, "message": str(exc)})
    return problems, failures


def _answers_match(prediction: str, answer: str | None, family: str, mismatches: Counter[str]) -> bool:
    if answer is None:
        mismatches["missing_answer"] += 1
        return False
    try:
        pred = validate_competition_answer(prediction)
        gold = validate_competition_answer(answer)
    except CompetitionAnswerPolicyError:
        mismatches["invalid_answer_format"] += 1
        return False
    if family in {"unit_conversion", "gravity_numeric"}:
        try:
            from decimal import Decimal

            return Decimal(pred) == Decimal(gold)
        except Exception:
            pass
    if pred == gold:
        return True
    mismatches[f"{family}_normalized_mismatch"] += 1
    return False


def _row(problem, reason: str, prediction: str | None, *, solver_name: str = "", diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": problem.problem_id, "family": normalize_family_name(problem.family), "answer": problem.answer, "prediction": prediction, "reason": reason, "solver_name": solver_name, "diagnostics": diagnostics or {}}


def _failure_samples(problems, failures: list[dict[str, Any]], family: str | None, *, limit: int) -> list[dict[str, Any]]:
    by_id = {problem.problem_id: problem for problem in problems}
    out = []
    for row in failures:
        if family is not None and row["family"] != family:
            continue
        problem = by_id.get(row["id"])
        if problem is None:
            continue
        out.append(
            {
                "id": row["id"],
                "family": row["family"],
                "prompt_excerpt": problem.raw_prompt[:600],
                "answer": row["answer"],
                "route_family": normalize_family_name(problem.family),
                "parser_result": {"example_count": len(problem.examples), "target_input": problem.target_input},
                "solver_attempted": row.get("solver_name", ""),
                "failure_reason": row["reason"],
                "candidate_count": row.get("diagnostics", {}).get("tested") or row.get("diagnostics", {}).get("verified_count"),
                "diagnostics": row.get("diagnostics", {}) if isinstance(row.get("diagnostics"), dict) else {},
                "top_candidate_summaries": row.get("diagnostics", {}).get("verified", [])[:5] if isinstance(row.get("diagnostics"), dict) else [],
                "answer_normalization_detail": {"prediction": row.get("prediction"), "answer": row.get("answer")},
            }
        )
        if len(out) >= limit:
            break
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _equation_program_row(row: dict[str, Any], status: str) -> dict[str, Any]:
    diagnostics = row.get("diagnostics", {}) if isinstance(row.get("diagnostics"), dict) else {}
    return {
        "id": row["id"],
        "family": row["family"],
        "status": status,
        "solver_path": row.get("solver_name", ""),
        "rule_id": row.get("reason", ""),
        "candidate_count": diagnostics.get("candidate_count"),
        "verified_program_count": diagnostics.get("verified_count"),
        "ambiguity_count": diagnostics.get("ambiguity_count", 0),
        "target_prediction": row.get("prediction"),
        "normalized_answer_match": status == "verified",
        "failure_reason": None if status == "verified" else row.get("reason"),
    }


def _bit_program_row(row: dict[str, Any], status: str) -> dict[str, Any]:
    diagnostics = row.get("diagnostics", {}) if isinstance(row.get("diagnostics"), dict) else {}
    return {
        "id": row["id"],
        "family": row["family"],
        "status": status,
        "solver_path": row.get("solver_name", ""),
        "rule_id": row.get("reason", ""),
        "candidate_count": diagnostics.get("candidate_count", diagnostics.get("tested")),
        "verified_expression_count": diagnostics.get("verified_count"),
        "ambiguity_count": diagnostics.get("ambiguity_count", 0),
        "depth_solved_distribution": diagnostics.get("depth_solved_distribution", {}),
        "pruned_duplicate_count": diagnostics.get("pruned_duplicate_count", 0),
        "pruned_failed_count": diagnostics.get("pruned_failed_count", 0),
        "target_prediction": row.get("prediction"),
        "normalized_answer_match": status == "verified",
        "failure_reason": None if status == "verified" else row.get("reason"),
    }


def _bit_failure_taxonomy(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clusters: list[dict[str, Any]] = []
    cluster_counts: Counter[str] = Counter()
    widths: Counter[str] = Counter()
    example_counts: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    for row in samples:
        prompt = str(row.get("prompt_excerpt") or "")
        parser = row.get("parser_result", {}) if isinstance(row.get("parser_result"), dict) else {}
        target = str(parser.get("target_input") or "")
        examples = _extract_bit_examples(prompt)
        width = len(target) if target else _infer_bit_width_from_examples(examples)
        cluster = _classify_bit_budget_failure(prompt, examples, target)
        detected = _detect_bit_tokens(prompt)
        cluster_counts[cluster] += 1
        widths[str(width or "unknown")] += 1
        example_counts[str(parser.get("example_count", len(examples)))] += 1
        operators.update(detected)
        clusters.append(
            {
                "id": row["id"],
                "family": "bit_manipulation",
                "cluster": cluster,
                "failure_reason": row["failure_reason"],
                "prompt_excerpt": prompt[:240],
                "example_count": parser.get("example_count", len(examples)),
                "input_width": width,
                "output_width": width,
                "target_width": len(target) if target else None,
                "detected_operators": detected,
                "full_binary_examples": bool(examples) and all(set(left + right) <= {"0", "1"} and len(left) == len(right) for left, right in examples),
                "candidate_budget_before_exhaustion": row.get("candidate_count"),
                "depth_reached": _depth_reached(row),
                "candidate_families_attempted": ["primitive", "mask", "rotate_shift", "binary_boolean", "bit_position_boolean"],
                "likely_required_families": _likely_bit_families(cluster),
            }
        )
    taxonomy = {
        "total_budget_exhausted_samples": len(samples),
        "cluster_counts": dict(cluster_counts.most_common()),
        "example_count_distribution": dict(example_counts.most_common()),
        "width_distribution": dict(widths.most_common()),
        "detected_operator_counts": dict(operators.most_common()),
        "bottleneck_summary": _bit_bottleneck_summary(cluster_counts),
    }
    return taxonomy, clusters


def _extract_bit_examples(prompt: str) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for left, right in re.findall(r"\b([01]{4,32})\s*->\s*([01]{4,32})\b", prompt):
        examples.append((left, right))
    return examples


def _infer_bit_width_from_examples(examples: list[tuple[str, str]]) -> int | None:
    widths = {len(left) for left, _ in examples} | {len(right) for _, right in examples}
    return widths.pop() if len(widths) == 1 else None


def _detect_bit_tokens(prompt: str) -> list[str]:
    lower = prompt.lower()
    tokens = []
    for token in ("rotate", "rotation", "shift", "xor", "and", "or", "not", "majority", "choice", "mask", "reverse", "add", "subtract", "modulo"):
        if token in lower:
            tokens.append(token)
    return tokens


def _classify_bit_budget_failure(prompt: str, examples: list[tuple[str, str]], target: str) -> str:
    detected = set(_detect_bit_tokens(prompt))
    if examples and all(_looks_sparse(right) for _, right in examples):
        return "mask_or_threshold_like"
    if {"majority", "choice"} & detected:
        return "majority_choice_composition"
    if {"rotate", "rotation", "shift"} & detected:
        return "rotate_shift_mask_composition"
    if target and len(target) != 8:
        return "variable_width_or_format"
    if len(examples) < 8:
        return "low_example_count_high_ambiguity"
    return "unknown_deeper_boolean_program"


def _looks_sparse(bits: str) -> bool:
    ones = bits.count("1")
    return ones <= 2 or ones >= max(0, len(bits) - 2)


def _depth_reached(row: dict[str, Any]) -> str:
    detail = row.get("answer_normalization_detail", {})
    diagnostics = row.get("diagnostics", {}) if isinstance(row.get("diagnostics"), dict) else {}
    depths = diagnostics.get("depth_solved_distribution", {})
    if depths:
        return ",".join(sorted(str(key) for key in depths))
    if detail:
        return "fallback_unknown"
    return "unknown"


def _likely_bit_families(cluster: str) -> list[str]:
    mapping = {
        "mask_or_threshold_like": ["mask", "threshold", "choice", "majority", "bit_position_boolean"],
        "majority_choice_composition": ["majority", "choice", "mux", "composition depth 2", "composition depth 3"],
        "rotate_shift_mask_composition": ["rotate", "shift", "mask", "composition depth 2", "composition depth 3"],
        "variable_width_or_format": ["width inference", "format conversion", "endian transform"],
        "low_example_count_high_ambiguity": ["more examples required", "ambiguity-safe rejection"],
        "unknown_deeper_boolean_program": ["composition depth 3", "composition depth 4", "truth-table pruning"],
    }
    return mapping.get(cluster, ["unknown"])


def _bit_bottleneck_summary(counts: Counter[str]) -> dict[str, str]:
    return {
        "dominant_cluster": counts.most_common(1)[0][0] if counts else "none",
        "why_not_solved": "Remaining budget rows are mostly underconstrained deeper boolean programs; target-disagreeing ambiguity is preserved instead of guessed.",
        "next_safe_step": "Implement offline/vectorized truth-table mining over budget rows, then promote only rules that verify all examples and do not increase bit wrong count.",
    }


def _write_equation_cluster_examples(path: Path, clusters: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in clusters:
        grouped.setdefault(row["cluster"], []).append(row)
    candidate_families = {
        "deletion_insertion": ["fixed deletion", "constant insertion", "copy prefix/suffix then insert/delete"],
        "digit_symbol_arithmetic": ["symbol digit binding", "digit concat/reverse concat", "per-symbol arithmetic tables"],
        "binary_operator_arithmetic": ["op-specific arithmetic", "affine ax+by+c", "exact division/modulo"],
        "string_rewrite": ["char transduction", "interleave/deinterleave", "substring movement"],
        "modulo_exact_division": ["mod m inference", "exact quotient/remainder pair"],
    }
    for cluster, rows in grouped.items():
        output_rows: list[dict[str, Any]] = [
            {
                "record_type": "summary",
                "cluster": cluster,
                "count": len(rows),
                "example_count_distribution": _distribution(row.get("parser_result", {}).get("example_count") for row in rows),
                "input_output_length_distribution": _length_distribution(rows),
                "symbol_alphabet_statistics": _alphabet_stats(rows),
                "operator_token_statistics": _operator_stats(rows),
                "previous_solver_failed_because": "no verified bounded program matched every visible example without unsafe target/operator extrapolation",
                "candidate_program_families_needed": candidate_families.get(cluster, ["richer verified program synthesis"]),
            }
        ]
        for row in rows[:20]:
            output_rows.append(
                {
                    "record_type": "example",
                    "id": row["id"],
                    "cluster": cluster,
                    "failure_reason": row["failure_reason"],
                    "target_input": row.get("target_input"),
                    "prompt_excerpt": row.get("prompt_excerpt", "")[:600],
                }
            )
        _write_jsonl(path / f"{cluster}.jsonl", output_rows)


def _distribution(values) -> dict[str, int]:
    return dict(Counter(str(value) for value in values if value is not None).most_common())


def _length_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        excerpt = str(row.get("prompt_excerpt") or "")
        for line in excerpt.splitlines():
            if " = " in line:
                left, right = line.split(" = ", 1)
                counts[f"{len(left.strip(' `'))}->{len(right.strip(' `'))}"] += 1
    return dict(counts.most_common(20))


def _alphabet_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chars = Counter()
    for row in rows:
        target = str(row.get("target_input") or "")
        chars.update(ch for ch in target if not ch.isalnum())
    return {"unique_symbol_count": len(chars), "top_symbols": dict(chars.most_common(15))}


def _operator_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    ops = Counter()
    for row in rows:
        target = str(row.get("target_input") or "")
        if len(target) >= 3:
            ops[target[2]] += 1
    return dict(ops.most_common(20))


def _equation_failure_taxonomy(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    clusters: list[dict[str, Any]] = []
    for row in samples:
        cluster = _classify_equation_failure(row)
        counts[cluster] += 1
        reasons[row["failure_reason"]] += 1
        clusters.append(
            {
                "id": row["id"],
                "cluster": cluster,
                "failure_reason": row["failure_reason"],
                "target_input": row.get("parser_result", {}).get("target_input"),
                "parser_result": row.get("parser_result", {}),
                "answer": row.get("answer"),
                "prompt_excerpt": row.get("prompt_excerpt", "")[:240],
            }
        )
    taxonomy = {
        "total_clustered": len(samples),
        "cluster_counts": dict(counts.most_common()),
        "failure_reason_counts": dict(reasons.most_common()),
        "cluster_definitions": {
            "string_rewrite": "fixed position deletion/reordering or substring rewrite over symbolic expressions",
            "deletion_insertion": "output length differs from input by dropped/inserted characters",
            "reordering_permutation": "output is a permutation/projection of visible input characters",
            "char_substitution": "same-length character bijection or substitution",
            "token_substitution": "whitespace token-level substitution",
            "symbol_sequence_transduction": "mixed symbolic sequence mapping not captured by simple projection",
            "binary_operator_arithmetic": "two numeric operands around an operator",
            "digit_symbol_arithmetic": "numeric operands with nonstandard symbolic operator or digit concatenation",
            "modulo_exact_division": "numeric operator task where modulo or exact division is plausible",
            "expression_precedence": "multi-operator expression requiring precedence choice",
            "lookup_table": "exact input repeats or deterministic seen lookup",
            "unknown": "insufficient structure for safe symbolic induction",
        },
    }
    return taxonomy, clusters


def _classify_equation_failure(row: dict[str, Any]) -> str:
    parser = row.get("parser_result", {})
    target = str(parser.get("target_input") or "")
    excerpt = str(row.get("prompt_excerpt") or "")
    pairs = []
    for line in excerpt.splitlines():
        if " = " in line:
            left, right = line.split(" = ", 1)
            pairs.append((left.strip(" `"), right.strip(" `")))
    if target in {x for x, _ in pairs}:
        return "lookup_table"
    if re.fullmatch(r"-?\d+\s*[+\-*/%|\\{}@#$?&<>`]\s*-?\d+", target):
        if any(op in target for op in "/%"):
            return "modulo_exact_division"
        if any(not op.isdigit() and op not in "-+" for op in target):
            return "digit_symbol_arithmetic"
        return "binary_operator_arithmetic"
    if len(target) == 5 and pairs and all(len(x) == 5 for x, _ in pairs[: min(len(pairs), 3)]):
        lengths = {len(y) for _, y in pairs}
        if lengths and max(lengths) < 5:
            return "deletion_insertion"
        if lengths == {5}:
            return "char_substitution"
        return "symbol_sequence_transduction"
    if " " in target:
        return "token_substitution"
    if any(ch in target for ch in "+-*/%|\\{}@#$?&<>`"):
        return "string_rewrite"
    return "unknown"


def _equation_cluster_for_problem(problem) -> str:
    fake = {
        "parser_result": {"target_input": problem.target_input},
        "prompt_excerpt": problem.raw_prompt[:1200],
    }
    return _classify_equation_failure(fake)


def _digit_symbol_failure_taxonomy(clusters: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [row for row in clusters if row["cluster"] in {"digit_symbol_arithmetic", "binary_operator_arithmetic"}]
    rows: list[dict[str, Any]] = []
    output_kinds: Counter[str] = Counter()
    operator_tokens: Counter[str] = Counter()
    length_patterns: Counter[str] = Counter()
    constraint_counts: Counter[str] = Counter()
    for row in selected:
        excerpt = str(row.get("prompt_excerpt") or "")
        pairs = []
        for line in excerpt.splitlines():
            if " = " in line:
                left, right = line.split(" = ", 1)
                left = left.strip(" `")
                right = right.strip(" `")
                pairs.append((left, right))
                length_patterns[f"{len(left)}->{len(right)}"] += 1
                output_kinds[_answer_kind(right)] += 1
                if len(left) >= 3:
                    operator_tokens[left[2]] += 1
        target = str(row.get("target_input") or "")
        if len(target) >= 3:
            operator_tokens[target[2]] += 1
        target_op_examples = sum(1 for left, _ in pairs if len(left) >= 3 and len(target) >= 3 and left[2] == target[2])
        enough_constraints = target_op_examples >= 2
        constraint_counts["enough_constraints" if enough_constraints else "underdetermined_or_unseen_target_operator"] += 1
        digit_chars = sorted({ch for left, right in pairs for ch in left + right if ch.isdigit()} | {ch for ch in target if ch.isdigit()})
        symbol_chars = sorted({ch for left, right in pairs for ch in left + right if not ch.isalnum() and not ch.isspace()} | {ch for ch in target if not ch.isalnum() and not ch.isspace()})
        rows.append(
            {
                "id": row["id"],
                "cluster": row["cluster"],
                "target_input": target,
                "output_kind": _answer_kind(str(row.get("answer") or "")),
                "operator_tokens": sorted(set(left[2] for left, _ in pairs if len(left) >= 3)),
                "target_operator_seen_count": target_op_examples,
                "input_output_length_pattern": dict(Counter(f"{len(left)}->{len(right)}" for left, right in pairs).most_common()),
                "symbol_alphabet": symbol_chars,
                "digit_alphabet": digit_chars,
                "symbols_as_operands": any((len(left) >= 1 and not left[0].isdigit()) or (len(left) >= 4 and not left[3].isdigit()) for left, _ in pairs),
                "symbols_as_operators": any(len(left) >= 3 and not left[2].isalnum() for left, _ in pairs),
                "leading_zero_meaningful": any(right.startswith("0") and len(right) > 1 for _, right in pairs),
                "enough_constraints_for_unique_induction": enough_constraints,
            }
        )
    taxonomy = {
        "total": len(selected),
        "clusters": dict(Counter(row["cluster"] for row in selected).most_common()),
        "output_kinds": dict(output_kinds.most_common()),
        "operator_tokens": dict(operator_tokens.most_common()),
        "input_output_length_patterns": dict(length_patterns.most_common(20)),
        "constraint_summary": dict(constraint_counts.most_common()),
    }
    return taxonomy, rows


def _answer_kind(text: str) -> str:
    if re.fullmatch(r"-?\d+", text or ""):
        return "digit_string" if len(text.lstrip("-")) > 1 and text.lstrip("-").startswith("0") else "integer"
    if any(ch.isdigit() for ch in text) and any(not ch.isdigit() for ch in text):
        return "mixed"
    if text and all(not ch.isalnum() and not ch.isspace() for ch in text):
        return "symbolic"
    return "unknown"

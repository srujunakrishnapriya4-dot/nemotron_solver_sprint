from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nemotron_engine.sprint_solvers.mapping_solver as mapping_solver  # noqa: E402
from nemotron_engine.sprint_solvers import SolverCandidate, parse_problem, run_predictions, solve_mapping_problem  # noqa: E402


def test_char_bijection_solved() -> None:
    result = solve_mapping_problem(parse_problem("ab -> XY\nba -> YX\nTarget: aba -> ?"))

    assert result.status == "solved"
    assert result.prediction == "XYX"


def test_token_bijection_solved() -> None:
    result = solve_mapping_problem(parse_problem("red blue -> uno dos\nblue red -> dos uno\nTarget: red red blue -> ?"))

    assert result.status == "solved"
    assert result.prediction == "uno uno dos"


def test_digit_symbol_mapping_solved() -> None:
    result = solve_mapping_problem(parse_problem("@# -> 12\n#@ -> 21\nTarget: @@# -> ?"))

    assert result.status == "solved"
    assert result.prediction == "112"


def test_alphabet_shift_solved() -> None:
    result = solve_mapping_problem(parse_problem("abc -> bcd\nxyz -> yza\nTarget: cab -> ?"))

    assert result.status == "solved"
    assert result.prediction == "dbc"


def test_one_character_alphabet_shift_rejected_abstains() -> None:
    result = solve_mapping_problem(parse_problem("a -> b\nTarget: z -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "insufficient_shift_evidence" for candidate in result.rejected_candidates)


def test_alphabet_shift_with_two_character_evidence_solves() -> None:
    result = solve_mapping_problem(parse_problem("az -> ba\nTarget: by -> ?"))

    assert result.status == "solved"
    assert result.prediction == "cz"


def test_collision_rejected() -> None:
    result = solve_mapping_problem(parse_problem("ab -> cc\nTarget: a -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "mapping_collision" for candidate in result.rejected_candidates)


def test_unseen_target_symbol_abstains() -> None:
    result = solve_mapping_problem(parse_problem("a -> B\nTarget: z -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "unseen_target_symbol" for candidate in result.rejected_candidates)


def test_unseen_target_token_gives_rejected_diagnostic() -> None:
    result = solve_mapping_problem(parse_problem("red blue -> uno dos\nblue red -> dos uno\nTarget: red green -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "unseen_target_token" for candidate in result.rejected_candidates)


def test_token_bijection_does_not_activate_for_bare_strings() -> None:
    candidates = mapping_solver.enumerate_mapping_candidates(parse_problem("ab -> XY\nba -> YX\nTarget: ab -> ?"))

    assert all(candidate.metadata["rule_id"] != "token_bijection" for candidate in candidates)


def test_exact_lookup_solves_only_seen_target() -> None:
    result = solve_mapping_problem(parse_problem("cat -> dog\nbird -> fish\nTarget: cat -> ?"))

    assert result.status == "solved"
    assert result.prediction == "dog"
    assert any(candidate.metadata["rule_id"] == "exact_lookup" for candidate in result.verified_candidates)


def test_exact_lookup_rejects_unseen_target() -> None:
    candidates = mapping_solver.enumerate_mapping_candidates(parse_problem("cat -> dog\nbird -> fish\nTarget: horse -> ?"))

    assert all(candidate.metadata["rule_id"] != "exact_lookup" for candidate in candidates)


def test_disagreement_returns_disagreement(monkeypatch) -> None:
    parsed = parse_problem("x -> y\nTarget: z -> ?")
    first = SolverCandidate("mapping_solver", ("y",), "1", {"rule_id": "first"})
    second = SolverCandidate("mapping_solver", ("y",), "2", {"rule_id": "second"})
    monkeypatch.setattr(mapping_solver, "enumerate_mapping_candidates", lambda _: (first, second))

    result = mapping_solver.solve_mapping_problem(parsed)

    assert result.status == "disagreement"


def test_prediction_runner_rejects_non_integer_mapping_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    input_path.write_text(json.dumps({"id": "p", "prompt": "mapping\ncat -> dog\nbird -> fish\nTarget: cat -> ?"}) + "\n", encoding="utf-8")

    report = run_predictions(input_path, output_dir)
    predictions = (output_dir / "predictions.jsonl").read_text(encoding="utf-8")
    abstentions = [json.loads(line) for line in (output_dir / "abstentions.jsonl").read_text(encoding="utf-8").splitlines() if line]

    assert report.prediction_count == 0
    assert predictions == ""
    assert abstentions[0]["reason"] == "invalid_symbolic_prediction"

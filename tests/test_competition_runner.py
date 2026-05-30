from __future__ import annotations

from dataclasses import replace
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint import (  # noqa: E402
    CompetitionRunnerError,
    run_competition_test_predictions,
    run_competition_train_eval,
)
from nemotron_engine.competition_sprint.competition_prompt_adapter import parse_competition_prompt  # noqa: E402
from nemotron_engine.competition_sprint.competition_runner import (  # noqa: E402
    _bit_choice,
    _bit_majority,
    _format_bits,
    _rol8,
    _ror8,
    _solve_bit_expression_synth,
    _solve_cipher,
    _solve_gravity,
    _solve_unit,
)


BIT_PROMPT = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.

Here are some examples of input -> output:
00000000 -> 11111111
11111111 -> 00000000
01010101 -> 10101010

Now, determine the output for: 00110100"""

CIPHER_PROMPT = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
aaa bbb -> cat dog
bbb aaa -> dog cat
ccc aaa -> book cat
Now, decrypt the following text: aaa ccc"""

ROMAN_PROMPT = """In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:
11 -> XI
15 -> XV
94 -> XCIV
19 -> XIX
Now, write the number 38 in the Wonderland numeral system."""


def write_csv(path: Path, rows: list[dict[str, str]], *, answers: bool) -> None:
    fields = ["id", "prompt", "answer"] if answers else ["id", "prompt"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_train_eval_small_fixture_preserves_formats(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    out = tmp_path / "out"
    write_csv(
        train,
        [
            {"id": "bit", "prompt": BIT_PROMPT, "answer": "11001011"},
            {"id": "cipher", "prompt": CIPHER_PROMPT, "answer": "cat book"},
            {"id": "roman", "prompt": ROMAN_PROMPT, "answer": "XXXVIII"},
        ],
        answers=True,
    )

    report = run_competition_train_eval(train, out)
    predictions = {row["problem_id"]: row["answer"] for row in read_jsonl(out / "competition_predictions_train.jsonl")}

    assert report.prediction_count == 3
    assert predictions == {"bit": "11001011", "cipher": "cat book", "roman": "XXXVIII"}
    assert report.accuracy == 1.0


def test_train_answer_not_used_for_inference(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    out = tmp_path / "out"
    write_csv(train, [{"id": "roman", "prompt": ROMAN_PROMPT, "answer": "I"}], answers=True)

    report = run_competition_train_eval(train, out)
    row = read_jsonl(out / "competition_predictions_train.jsonl")[0]

    assert row["answer"] == "XXXVIII"
    assert row["correct"] is False
    assert report.accuracy == 0.0


def test_test_candidate_submission_written(tmp_path: Path) -> None:
    test = tmp_path / "test.csv"
    out = tmp_path / "out"
    write_csv(test, [{"id": "roman", "prompt": ROMAN_PROMPT}], answers=False)

    report = run_competition_test_predictions(test, out)
    submission = (out / "candidate_submission.csv").read_text(encoding="utf-8")

    assert report.prediction_count == 1
    assert submission.splitlines() == ["id,answer", "roman,XXXVIII"]


def test_report_hash_forgery_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    out = tmp_path / "out"
    write_csv(train, [{"id": "roman", "prompt": ROMAN_PROMPT, "answer": "XXXVIII"}], answers=True)
    report = run_competition_train_eval(train, out)

    with pytest.raises(CompetitionRunnerError):
        replace(report, report_hash="forged")


def bit_prompt(rule, target: str = "00110100") -> tuple[str, str]:
    inputs = ("00000000", "11111111", "01010101", "10010011", "00111100")
    lines = "\n".join(f"{item} -> {_format_bits(rule(int(item, 2)))}" for item in inputs)
    prompt = f"""In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.

Here are some examples of input -> output:
{lines}

Now, determine the output for: {target}"""
    return prompt, _format_bits(rule(int(target, 2)))


def test_bit_expression_majority_solves() -> None:
    prompt, expected = bit_prompt(lambda x: _bit_majority((x << 1) & 255, x >> 1, _rol8(x, 1)))
    problem = parse_competition_prompt("maj", prompt)

    assert _solve_bit_expression_synth(problem) == (expected, "solved")


def test_bit_expression_choice_solves() -> None:
    prompt, expected = bit_prompt(lambda x: _bit_choice((x << 1) & 255, x >> 1, _rol8(x, 2)))
    problem = parse_competition_prompt("choice", prompt)

    assert _solve_bit_expression_synth(problem) == (expected, "solved")


def test_bit_expression_composed_shift_xor_solves_and_stays_bitstring() -> None:
    prompt, expected = bit_prompt(lambda x: ((x << 1) & 255) ^ _ror8(x, 1))
    problem = parse_competition_prompt("xor", prompt)
    prediction, reason = _solve_bit_expression_synth(problem)

    assert (prediction, reason) == (expected, "solved")
    assert isinstance(prediction, str)
    assert len(prediction) == 8
    assert set(prediction) <= {"0", "1"}


def test_bit_expression_three_way_xor_with_not_solves() -> None:
    prompt, expected = bit_prompt(lambda x: ~(((x << 1) & 255) ^ (x >> 1) ^ _ror8(x, 1)))
    problem = parse_competition_prompt("xor3", prompt)

    assert _solve_bit_expression_synth(problem) == (expected, "solved")


def test_bit_expression_disagreement_abstains() -> None:
    prompt = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.

Here are some examples of input -> output:
00000000 -> 00000000

Now, determine the output for: 00000001"""
    problem = parse_competition_prompt("ambiguous-bit", prompt)

    assert _solve_bit_expression_synth(problem) == (None, "bit_expression_disagreement")


def test_bit_expression_candidate_budget_respected() -> None:
    prompt, _ = bit_prompt(lambda x: _bit_majority((x << 1) & 255, x >> 1, _rol8(x, 1)))
    problem = parse_competition_prompt("budget", prompt)

    assert _solve_bit_expression_synth(problem, max_candidates=0) == (None, "bit_candidate_budget_exhausted")


def test_bit_expression_does_not_bypass_example_verification() -> None:
    prompt = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.

Here are some examples of input -> output:
00000001 -> 00000000
00000001 -> 11111111

Now, determine the output for: 00000001"""
    problem = parse_competition_prompt("no-bypass", prompt)

    assert _solve_bit_expression_synth(problem)[0] is None


def test_cipher_prompt_vocabulary_completion_solves_unique_pattern() -> None:
    prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
trb wzrswvog hffk -> cat imagines book
Now, decrypt the following text: trb wzrswvog hffk"""
    problem = parse_competition_prompt("cipher-complete", prompt)

    assert _solve_cipher(problem) == ("cat imagines book", "solved")


def test_cipher_prompt_vocabulary_ambiguous_pattern_abstains() -> None:
    prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
aaa ccc -> cat dog
bbb ddd -> hat fog
Now, decrypt the following text: zzz ccc"""
    problem = parse_competition_prompt("cipher-ambiguous", prompt)

    assert _solve_cipher(problem)[0] is None


def test_cipher_completion_collision_abstains() -> None:
    prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
aaa bbb -> cat cat
Now, decrypt the following text: aaa"""
    problem = parse_competition_prompt("cipher-collision", prompt)

    assert _solve_cipher(problem) == (None, "mapping_collision")


def test_cipher_corpus_visible_vocabulary_completion_solves_unique_pattern() -> None:
    prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
gbcpovb tqorbog bxo zrswtrj pffq -> student creates the magical door
bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows
nqwvtogg qorpg bxo zegboqwfcg gotqob -> princess reads the mysterious secret
Now, decrypt the following text: trb wzrswvog hffk"""
    problem = parse_competition_prompt("cipher-corpus", prompt)

    assert _solve_cipher(problem, {"book": 3}) == ("cat imagines book", "solved")


def test_cipher_corpus_visible_completion_ambiguous_abstains() -> None:
    prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ff -> oo
Now, decrypt the following text: hffk"""
    problem = parse_competition_prompt("cipher-corpus-ambiguous", prompt)

    assert _solve_cipher(problem, {"book": 1, "door": 1}) == (None, "ambiguous_corpus_vocabulary_completion")


def test_cipher_corpus_visible_completion_collision_abstains() -> None:
    prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
pff -> boo
Now, decrypt the following text: hffk"""
    problem = parse_competition_prompt("cipher-corpus-collision", prompt)

    assert _solve_cipher(problem, {"book": 1}) == (None, "corpus_completion_collision")


def test_cipher_corpus_visible_completion_does_not_use_answer_column(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    out = tmp_path / "out"
    prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ff -> oo
Now, decrypt the following text: hffk"""
    write_csv(train, [{"id": "label-only", "prompt": prompt, "answer": "book"}], answers=True)
    write_csv(test, [{"id": "target", "prompt": prompt}], answers=False)

    report = run_competition_test_predictions(test, out)
    abstentions = read_jsonl(out / "competition_abstentions_test.jsonl")

    assert report.prediction_count == 0
    assert abstentions[0]["reason"] == "unseen_target_token"
    assert abstentions[0]["partial_decoded_phrase"] == "?oo?"


def test_cipher_corpus_visible_completion_uses_only_visible_prompt_examples(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    out = tmp_path / "out"
    target_prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ff -> oo
Now, decrypt the following text: hffk"""
    visible_vocab_prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
zzzz -> book
Now, decrypt the following text: zzzz"""
    write_csv(train, [{"id": "visible", "prompt": visible_vocab_prompt, "answer": "not used"}], answers=True)
    write_csv(test, [{"id": "target", "prompt": target_prompt}], answers=False)

    report = run_competition_test_predictions(test, out)
    predictions = read_jsonl(out / "competition_predictions_test.jsonl")

    assert report.prediction_count == 1
    assert predictions[0]["problem_id"] == "target"
    assert predictions[0]["answer"] == "book"


def test_cipher_corpus_visible_completion_not_hardcoded_to_test_id() -> None:
    prompt = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
gbcpovb tqorbog bxo zrswtrj pffq -> student creates the magical door
bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows
nqwvtogg qorpg bxo zegboqwfcg gotqob -> princess reads the mysterious secret
Now, decrypt the following text: trb wzrswvog hffk"""
    problem = parse_competition_prompt("arbitrary-id", prompt)

    assert _solve_cipher(problem, {"book": 1}) == ("cat imagines book", "solved")


def test_gravity_rounded_examples_solve() -> None:
    prompt = """In Alice's Wonderland, the gravitational constant has been secretly changed.
For t = 1.37s, distance = 14.92 m
For t = 4.27s, distance = 144.96 m
For t = 3.28s, distance = 85.54 m
Now, determine the falling distance for t = 4.41s"""
    problem = parse_competition_prompt("gravity", prompt)

    assert _solve_gravity(problem) == ("154.62", "solved")


def test_gravity_inconsistent_examples_abstain() -> None:
    prompt = """In Alice's Wonderland, the gravitational constant has been secretly changed.
For t = 1.00s, distance = 8.00 m
For t = 2.00s, distance = 99.00 m
Now, determine the falling distance for t = 3.00s"""
    problem = parse_competition_prompt("gravity-bad", prompt)

    assert _solve_gravity(problem) == (None, "no_verified_gravity_rule")


def test_unit_ratio_rounded_examples_solve() -> None:
    prompt = """In Alice's Wonderland, a secret unit conversion is applied to measurements.
13.34 m becomes 16.24
23.29 m becomes 28.35
7.03 m becomes 8.56
Now, convert the following measurement: 34.62 m"""
    problem = parse_competition_prompt("unit-ratio", prompt)

    assert _solve_unit(problem) == ("42.15", "solved")


def test_unit_affine_rounded_examples_solve() -> None:
    prompt = """In Alice's Wonderland, a secret unit conversion is applied to measurements.
1.00 m becomes 3.00
2.00 m becomes 5.00
3.00 m becomes 7.00
Now, convert the following measurement: 4.00 m"""
    problem = parse_competition_prompt("unit-affine", prompt)

    assert _solve_unit(problem) == ("9.00", "solved")


def test_unit_ratio_affine_disagreement_abstains() -> None:
    prompt = """In Alice's Wonderland, a secret unit conversion is applied to measurements.
1.00 m becomes 1.00
2.00 m becomes 1.99
Now, convert the following measurement: 0.00 m"""
    problem = parse_competition_prompt("unit-ambiguous", prompt)

    assert _solve_unit(problem) == (None, "numeric_model_disagreement")

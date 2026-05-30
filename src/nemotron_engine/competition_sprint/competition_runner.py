"""Competition sprint train evaluation and test prediction runner."""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
import csv
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, getcontext
import json
from pathlib import Path
import re
from typing import Any

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.sprint_solvers import ExamplePair, ParsedProblem, solve_bit_problem

from .competition_answer_policy import CompetitionAnswerPolicyError, infer_answer_kind, validate_competition_answer
from .competition_family_router import route_competition_problem
from .competition_prompt_adapter import CompetitionProblem, parse_competition_prompt

getcontext().prec = 50

_BIT_MASK = 0xFF
_BIT_BUDGET = 5000
_BIT_RE = re.compile(r"^[01]{8}$")
_ALPHA_WORD_RE = re.compile(r"[a-z]+")


class CompetitionRunnerError(ValueError):
    """Raised when a competition run fails validation."""


@dataclass(frozen=True)
class CompetitionPrediction:
    problem_id: str
    answer: str
    family: str
    source: str
    prediction_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _require_text(self.problem_id, "problem_id"))
        object.__setattr__(self, "answer", validate_competition_answer(self.answer))
        object.__setattr__(self, "family", _require_text(self.family, "family"))
        object.__setattr__(self, "source", _require_text(self.source, "source"))
        _set_or_check_hash(self, "prediction_hash")


@dataclass(frozen=True)
class CompetitionRunReport:
    mode: str
    total_rows: int
    parsed_count: int
    prediction_count: int
    abstention_count: int
    error_count: int
    accuracy: float | None
    accuracy_by_family: Mapping[str, float | None]
    prediction_count_by_family: Mapping[str, int]
    abstention_count_by_family: Mapping[str, int]
    report_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _require_text(self.mode, "mode"))
        for field_name in ("total_rows", "parsed_count", "prediction_count", "abstention_count", "error_count"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise CompetitionRunnerError(f"{field_name} cannot be negative.")
            object.__setattr__(self, field_name, value)
        if self.total_rows != self.prediction_count + self.abstention_count + self.error_count:
            raise CompetitionRunnerError("total rows must equal predictions plus abstentions plus errors.")
        object.__setattr__(self, "accuracy_by_family", dict(sorted(self.accuracy_by_family.items())))
        object.__setattr__(self, "prediction_count_by_family", _count_map(self.prediction_count_by_family))
        object.__setattr__(self, "abstention_count_by_family", _count_map(self.abstention_count_by_family))
        _set_or_check_hash(self, "report_hash")


def run_competition_train_eval(train_csv_path: str | Path, output_dir: str | Path) -> CompetitionRunReport:
    """Run conservative local competition solvers on labeled train CSV."""

    return _run_competition(train_csv_path, output_dir, mode="train", has_answers=True)


def run_competition_test_predictions(test_csv_path: str | Path, output_dir: str | Path) -> CompetitionRunReport:
    """Run conservative local competition solvers on unlabeled test CSV and write candidate CSV."""

    return _run_competition(test_csv_path, output_dir, mode="test", has_answers=False)


def validate_competition_run_report(report: CompetitionRunReport, output_dir: str | Path | None = None) -> CompetitionRunReport:
    """Validate report hash and optional artifact counts."""

    _validate_hash(report, "report_hash")
    if output_dir is None:
        return report
    out = Path(output_dir)
    suffix = report.mode
    predictions = _read_jsonl(out / f"competition_predictions_{suffix}.jsonl")
    abstentions = _read_jsonl(out / f"competition_abstentions_{suffix}.jsonl")
    errors = _read_jsonl(out / f"competition_errors_{suffix}.jsonl")
    metrics = _read_json(out / f"competition_family_metrics_{suffix}.json")
    if len(predictions) != report.prediction_count:
        raise CompetitionRunnerError("prediction count artifact mismatch.")
    if len(abstentions) != report.abstention_count:
        raise CompetitionRunnerError("abstention count artifact mismatch.")
    if len(errors) != report.error_count:
        raise CompetitionRunnerError("error count artifact mismatch.")
    if metrics.get("report_hash") != report.report_hash:
        raise CompetitionRunnerError("metrics hash mismatch.")
    for row in predictions:
        validate_competition_answer(row["answer"])
    if report.mode == "test":
        csv_path = out / "candidate_submission.csv"
        if not csv_path.exists():
            raise CompetitionRunnerError("candidate submission CSV missing.")
    return report


def _build_cipher_visible_vocabulary(path: Path, *, include_sibling_train: bool) -> Counter[str]:
    """Collect lowercase words from visible cipher example outputs, never labels."""

    paths = [path]
    sibling_train = path.with_name("train.csv")
    if include_sibling_train and sibling_train.exists() and sibling_train != path:
        paths.append(sibling_train)

    vocabulary: Counter[str] = Counter()
    for source_path in paths:
        if not source_path.exists():
            continue
        with source_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                problem_id = str(row.get("id") or row.get("problem_id") or f"row-{index}")
                prompt = str(row.get("prompt") or row.get("raw_prompt") or "")
                try:
                    problem = parse_competition_prompt(problem_id, prompt, None)
                except Exception:
                    continue
                if problem.family == "cipher_text":
                    vocabulary.update(_visible_plaintext_words(problem))
    return vocabulary


def _run_competition(path: str | Path, output_dir: str | Path, *, mode: str, has_answers: bool) -> CompetitionRunReport:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cipher_vocabulary = _build_cipher_visible_vocabulary(Path(path), include_sibling_train=mode == "test")
    predictions: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    family_total: dict[str, int] = {}
    family_pred: dict[str, int] = {}
    family_abstain: dict[str, int] = {}
    family_correct: dict[str, int] = {}
    parsed_count = 0
    total_rows = 0

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            total_rows += 1
            problem_id = str(row.get("id") or row.get("problem_id") or f"row-{index}")
            try:
                problem = parse_competition_prompt(problem_id, str(row.get("prompt") or row.get("raw_prompt") or ""), row.get("answer") if has_answers else None)
                parsed_count += 1
                family_total[problem.family] = family_total.get(problem.family, 0) + 1
                prediction, reason, diagnostics = _solve_problem(problem, cipher_vocabulary=cipher_vocabulary)
                if prediction is None:
                    family_abstain[problem.family] = family_abstain.get(problem.family, 0) + 1
                    row_out = {"problem_id": problem.problem_id, "family": problem.family, "reason": reason}
                    row_out.update(diagnostics)
                    abstentions.append(row_out)
                    continue
                expected_kind = problem.answer_kind if has_answers and problem.answer_kind else _expected_kind_for_family(problem.family)
                answer = validate_competition_answer(prediction, expected_kind=expected_kind)
                pred = CompetitionPrediction(problem.problem_id, answer, problem.family, "symbolic")
                row_out: dict[str, Any] = _record_to_dict(pred)
                if has_answers and problem.answer is not None:
                    row_out["expected_answer"] = problem.answer
                    row_out["correct"] = answer == problem.answer
                    if row_out["correct"]:
                        family_correct[problem.family] = family_correct.get(problem.family, 0) + 1
                predictions.append(row_out)
                family_pred[problem.family] = family_pred.get(problem.family, 0) + 1
            except Exception as exc:
                errors.append({"problem_id": problem_id, "reason": str(exc), "row_index": index})

    accuracy_by_family = {
        family: (family_correct.get(family, 0) / family_pred[family]) if has_answers and family_pred.get(family) else None
        for family in sorted(family_total)
    }
    accuracy = (sum(1 for row in predictions if row.get("correct")) / len(predictions)) if has_answers and predictions else None
    report = CompetitionRunReport(
        mode=mode,
        total_rows=total_rows,
        parsed_count=parsed_count,
        prediction_count=len(predictions),
        abstention_count=len(abstentions),
        error_count=len(errors),
        accuracy=accuracy,
        accuracy_by_family=accuracy_by_family,
        prediction_count_by_family=family_pred,
        abstention_count_by_family=family_abstain,
    )
    _write_jsonl(out / f"competition_predictions_{mode}.jsonl", predictions)
    _write_jsonl(out / f"competition_abstentions_{mode}.jsonl", abstentions)
    _write_jsonl(out / f"competition_errors_{mode}.jsonl", errors)
    _write_json(out / f"competition_family_metrics_{mode}.json", _record_to_dict(report))
    if mode == "test":
        _write_candidate_submission(out / "candidate_submission.csv", predictions)
    validate_competition_run_report(report, out)
    return report


def _solve_problem(problem: CompetitionProblem, *, cipher_vocabulary: Mapping[str, int] | None = None) -> tuple[str | None, str, dict[str, Any]]:
    route_competition_problem(problem)
    if problem.family == "bit_manipulation":
        prediction, reason = _solve_bit(problem)
        return prediction, reason, {}
    if problem.family == "cipher_text":
        return _solve_cipher_detailed(problem, corpus_vocabulary=cipher_vocabulary)
    if problem.family == "roman_numeral":
        prediction, reason = _solve_roman(problem)
        return prediction, reason, {}
    if problem.family == "unit_conversion":
        prediction, reason = _solve_unit(problem)
        return prediction, reason, {}
    if problem.family == "gravity_numeric":
        prediction, reason = _solve_gravity(problem)
        return prediction, reason, {}
    if problem.family == "equation_symbolic":
        prediction, reason = _solve_equation(problem)
        return prediction, reason, {}
    return None, "unknown_family", {}


def _solve_bit(problem: CompetitionProblem) -> tuple[str | None, str]:
    parsed = ParsedProblem(
        problem_id=problem.problem_id,
        raw_prompt=problem.raw_prompt,
        examples=tuple(ExamplePair(example.input, example.output) for example in problem.examples),
        target_input=problem.target_input,
    )
    result = solve_bit_problem(parsed)
    if result.status == "solved":
        raw_outputs = {candidate.metadata.get("raw_candidate_output") for candidate in result.verified_candidates if candidate.metadata.get("raw_candidate_output")}
        if len(raw_outputs) != 1:
            return None, "missing_unique_raw_bit_output"
        return str(next(iter(raw_outputs))), "solved"
    return _solve_bit_expression_synth(problem)


def _solve_bit_expression_synth(problem: CompetitionProblem, *, max_candidates: int = _BIT_BUDGET) -> tuple[str | None, str]:
    """Bounded 8-bit expression search over prompt-visible examples only."""

    if max_candidates <= 0:
        return None, "bit_candidate_budget_exhausted"
    if not _BIT_RE.fullmatch(problem.target_input):
        return None, "invalid_bit_input"
    if any(not _BIT_RE.fullmatch(example.input) or not _BIT_RE.fullmatch(example.output) for example in problem.examples):
        return None, "invalid_bit_input"

    xs = tuple(int(example.input, 2) for example in problem.examples)
    ys = tuple(int(example.output, 2) for example in problem.examples)
    target = int(problem.target_input, 2)
    primitives = _bit_primitives(xs, target)
    verified: list[str] = []
    tested = 0

    def consider(values: tuple[int, ...], target_value: int) -> bool:
        nonlocal tested
        if tested >= max_candidates:
            return False
        tested += 1
        if values == ys:
            verified.append(_format_bits(target_value))
        return True

    for _, values, target_value in primitives:
        if not consider(values, target_value):
            break

    if tested < max_candidates:
        for _, values, target_value in primitives:
            inverted = tuple((~value) & _BIT_MASK for value in values)
            if not consider(inverted, (~target_value) & _BIT_MASK):
                break

    if tested < max_candidates:
        binary_ops = (
            lambda left, right: left ^ right,
            lambda left, right: left & right,
            lambda left, right: left | right,
        )
        for _, left_values, left_target in primitives:
            for _, right_values, right_target in primitives:
                for op in binary_ops:
                    values = tuple(op(left, right) & _BIT_MASK for left, right in zip(left_values, right_values))
                    if not consider(values, op(left_target, right_target) & _BIT_MASK):
                        break
                if tested >= max_candidates:
                    break
            if tested >= max_candidates:
                break

    if tested < max_candidates:
        for _, first_values, first_target in primitives:
            for _, second_values, second_target in primitives:
                for _, third_values, third_target in primitives:
                    xor_values = tuple((a ^ b ^ c) & _BIT_MASK for a, b, c in zip(first_values, second_values, third_values))
                    xor_target = (first_target ^ second_target ^ third_target) & _BIT_MASK
                    if not consider(xor_values, xor_target):
                        break
                    inverted_xor = tuple((~value) & _BIT_MASK for value in xor_values)
                    if not consider(inverted_xor, (~xor_target) & _BIT_MASK):
                        break
                    majority_values = tuple(_bit_majority(a, b, c) for a, b, c in zip(first_values, second_values, third_values))
                    if not consider(majority_values, _bit_majority(first_target, second_target, third_target)):
                        break
                    choice_values = tuple(_bit_choice(a, b, c) for a, b, c in zip(first_values, second_values, third_values))
                    if not consider(choice_values, _bit_choice(first_target, second_target, third_target)):
                        break
                if tested >= max_candidates:
                    break
            if tested >= max_candidates:
                break

    unique = sorted(set(verified))
    if not unique:
        return None, "bit_candidate_budget_exhausted" if tested >= max_candidates else "abstain"
    if len(unique) > 1:
        return None, "bit_expression_disagreement"
    return unique[0], "solved"


def _bit_primitives(xs: tuple[int, ...], target: int) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    primitives: list[tuple[str, tuple[int, ...], int]] = []

    def add(name: str, fn: Any) -> None:
        primitives.append((name, tuple(fn(x) & _BIT_MASK for x in xs), fn(target) & _BIT_MASK))

    add("x", lambda x: x)
    add("not", lambda x: (~x) & _BIT_MASK)
    for amount in (1, 2, 3):
        add(f"shl{amount}", lambda x, amount=amount: x << amount)
    for amount in (1, 2, 3):
        add(f"shr{amount}", lambda x, amount=amount: x >> amount)
    for amount in (1, 2, 3, 4):
        add(f"rol{amount}", lambda x, amount=amount: _rol8(x, amount))
    for amount in (1, 2, 3, 4):
        add(f"ror{amount}", lambda x, amount=amount: _ror8(x, amount))
    primitives.append(("const_00", tuple(0 for _ in xs), 0))
    primitives.append(("const_ff", tuple(_BIT_MASK for _ in xs), _BIT_MASK))
    return tuple(primitives)


def _rol8(value: int, amount: int) -> int:
    return ((value << amount) & _BIT_MASK) | (value >> (8 - amount))


def _ror8(value: int, amount: int) -> int:
    return (value >> amount) | ((value << (8 - amount)) & _BIT_MASK)


def _bit_majority(first: int, second: int, third: int) -> int:
    return ((first & second) | (first & third) | (second & third)) & _BIT_MASK


def _bit_choice(first: int, second: int, third: int) -> int:
    return ((first & second) | (((~first) & _BIT_MASK) & third)) & _BIT_MASK


def _format_bits(value: int) -> str:
    return format(value & _BIT_MASK, "08b")


def _solve_cipher(problem: CompetitionProblem, corpus_vocabulary: Mapping[str, int] | None = None) -> tuple[str | None, str]:
    prediction, reason, _ = _solve_cipher_detailed(problem, corpus_vocabulary=corpus_vocabulary)
    return prediction, reason


def _solve_cipher_detailed(problem: CompetitionProblem, *, corpus_vocabulary: Mapping[str, int] | None = None) -> tuple[str | None, str, dict[str, Any]]:
    token_map: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for example in problem.examples:
        inputs = example.input.split()
        outputs = example.output.split()
        if len(inputs) != len(outputs):
            return None, "token_length_mismatch", {}
        for left, right in zip(inputs, outputs):
            if token_map.get(left, right) != right or reverse.get(right, left) != left:
                return None, "mapping_collision", {}
            token_map[left] = right
            reverse[right] = left
    target_tokens = problem.target_input.split()
    if target_tokens and all(token in token_map for token in target_tokens):
        return " ".join(token_map[token] for token in target_tokens), "solved", {}

    char_map: dict[str, str] = {}
    char_reverse: dict[str, str] = {}
    for example in problem.examples:
        if len(example.input) != len(example.output):
            return None, "unseen_target_token", {}
        for left, right in zip(example.input, example.output):
            if char_map.get(left, right) != right or char_reverse.get(right, left) != left:
                return None, "mapping_collision", {}
            char_map[left] = right
            char_reverse[right] = left
    if all(char in char_map for char in problem.target_input):
        return "".join(char_map[char] for char in problem.target_input), "solved", {}

    prompt_vocabulary = Counter(_visible_plaintext_words(problem))
    prediction, reason, diagnostics = _complete_cipher_from_vocabulary(
        problem,
        char_map,
        char_reverse,
        prompt_vocabulary,
        source="same_prompt_vocab",
        ambiguous_reason="ambiguous_cipher_completion",
        collision_reason="mapping_collision",
    )
    if prediction is not None or reason == "ambiguous_cipher_completion":
        return prediction, reason, diagnostics
    if corpus_vocabulary:
        prediction, reason, diagnostics = _complete_cipher_from_vocabulary(
            problem,
            char_map,
            char_reverse,
            Counter(corpus_vocabulary),
            source="corpus_visible_vocab",
            ambiguous_reason="ambiguous_corpus_vocabulary_completion",
            collision_reason="corpus_completion_collision",
        )
        return prediction, reason, diagnostics
    return None, "unseen_target_token", diagnostics


def _complete_cipher_from_vocabulary(
    problem: CompetitionProblem,
    char_map: dict[str, str],
    char_reverse: dict[str, str],
    vocabulary_counts: Mapping[str, int],
    *,
    source: str,
    ambiguous_reason: str,
    collision_reason: str,
) -> tuple[str | None, str, dict[str, Any]]:
    vocabulary = sorted(vocabulary_counts)
    completed_map = dict(char_map)
    completed_reverse = dict(char_reverse)
    decoded_tokens: list[str] = []
    candidate_matches: dict[str, list[str]] = {}
    unknown_patterns: list[str] = []
    diagnostics: dict[str, Any] = {
        "partial_decoded_phrase": _partial_decode_phrase(problem.target_input, completed_map),
        "unknown_token_patterns": unknown_patterns,
        "candidate_vocabulary_matches": candidate_matches,
        "completion_source": source,
    }

    for encrypted_token in problem.target_input.split():
        if all(char in completed_map for char in encrypted_token):
            decoded_tokens.append("".join(completed_map[char] for char in encrypted_token))
            continue
        matches: list[tuple[str, dict[str, str]]] = []
        pattern = _partial_decode_token(encrypted_token, completed_map)
        unknown_patterns.append(pattern)
        collision_seen = False
        for word in vocabulary:
            extension, blocked_reason = _cipher_word_extension(encrypted_token, word, completed_map, completed_reverse)
            if extension is not None:
                matches.append((word, extension))
            elif blocked_reason == "collision":
                collision_seen = True
        candidate_matches[pattern] = [word for word, _ in matches]
        if len(matches) != 1:
            if not matches and collision_seen:
                return None, collision_reason, diagnostics
            return None, ambiguous_reason if len(matches) > 1 else "unseen_target_token", diagnostics
        word, extension = matches[0]
        for left, right in extension.items():
            if completed_map.get(left, right) != right or completed_reverse.get(right, left) != left:
                return None, collision_reason, diagnostics
            completed_map[left] = right
            completed_reverse[right] = left
        decoded_tokens.append(word)

    for example in problem.examples:
        if "".join(completed_map.get(char, "?") for char in example.input) != example.output:
            return None, "cipher_completion_failed_verification", diagnostics
    return " ".join(decoded_tokens), "solved", diagnostics


def _cipher_word_extension(encrypted: str, plaintext: str, mapping: dict[str, str], reverse: dict[str, str]) -> tuple[dict[str, str] | None, str | None]:
    if len(encrypted) != len(plaintext):
        return None, "mismatch"
    extension: dict[str, str] = {}
    local_reverse = dict(reverse)
    for left, right in zip(encrypted, plaintext):
        known = mapping.get(left, extension.get(left))
        if known is not None and known != right:
            return None, "mismatch"
        known_left = local_reverse.get(right)
        if known_left is not None and known_left != left:
            return None, "collision"
        extension[left] = right
        local_reverse[right] = left
    return extension, None


def _visible_plaintext_words(problem: CompetitionProblem) -> tuple[str, ...]:
    return tuple(word for example in problem.examples for word in _ALPHA_WORD_RE.findall(example.output.lower()))


def _partial_decode_phrase(text: str, mapping: Mapping[str, str]) -> str:
    return " ".join(_partial_decode_token(token, mapping) for token in text.split())


def _partial_decode_token(token: str, mapping: Mapping[str, str]) -> str:
    return "".join(mapping.get(char, "?") for char in token)


def _solve_roman(problem: CompetitionProblem) -> tuple[str | None, str]:
    try:
        for example in problem.examples:
            if _to_roman(int(example.input)) != example.output:
                return None, "non_standard_roman_examples"
        return _to_roman(int(problem.target_input)), "solved"
    except Exception:
        return None, "roman_parse_error"


def _solve_unit(problem: CompetitionProblem) -> tuple[str | None, str]:
    pairs = [(_decimal_from_text(example.input), Decimal(example.output)) for example in problem.examples]
    target = _decimal_from_text(problem.target_input)
    precision = _consistent_precision(tuple(example.output for example in problem.examples))
    if precision is None:
        return None, "inconsistent_numeric_precision"

    targets: set[str] = set()
    ratio_target = _fit_ratio_interval(pairs, target, precision)
    if ratio_target is not None:
        if len(pairs) >= 3:
            return ratio_target, "solved"
        targets.add(ratio_target)
    targets.update(_fit_affine_candidates(pairs, target, precision))
    if not targets:
        return None, "no_verified_numeric_rule"
    if len(targets) > 1:
        return None, "numeric_model_disagreement"
    return next(iter(targets)), "solved"


def _solve_gravity(problem: CompetitionProblem) -> tuple[str | None, str]:
    pairs = [(Decimal(example.input), Decimal(example.output)) for example in problem.examples]
    target = Decimal(problem.target_input)
    precision = _consistent_precision(tuple(example.output for example in problem.examples))
    if precision is None:
        return None, "inconsistent_numeric_precision"
    transformed = tuple((t * t, d) for t, d in pairs)
    prediction = _fit_ratio_interval(transformed, target * target, precision)
    if prediction is not None:
        return prediction, "solved"
    return None, "no_verified_gravity_rule"


def _solve_equation(problem: CompetitionProblem) -> tuple[str | None, str]:
    lookup = {example.input: example.output for example in problem.examples}
    if problem.target_input in lookup:
        return lookup[problem.target_input], "solved"
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for example in problem.examples:
        if len(example.input) != len(example.output):
            return None, "unsupported_equation_transform"
        for left, right in zip(example.input, example.output):
            if mapping.get(left, right) != right or reverse.get(right, left) != left:
                return None, "mapping_collision"
            mapping[left] = right
            reverse[right] = left
    if all(char in mapping for char in problem.target_input):
        return "".join(mapping[char] for char in problem.target_input), "solved"
    return None, "unseen_target_symbol"


def _expected_kind_for_family(family: str) -> str | None:
    return {
        "bit_manipulation": "bitstring",
        "cipher_text": "lowercase_text",
        "roman_numeral": "roman",
        "unit_conversion": "decimal",
        "gravity_numeric": "decimal",
    }.get(family)


def _to_roman(value: int) -> str:
    if value <= 0 or value > 3999:
        raise ValueError("roman value out of range")
    parts: list[str] = []
    for amount, symbol in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while value >= amount:
            parts.append(symbol)
            value -= amount
    return "".join(parts)


def _decimal_from_text(text: str) -> Decimal:
    match = re_search_decimal(text)
    if match is None:
        raise CompetitionRunnerError("decimal input not found.")
    return Decimal(match)


def re_search_decimal(text: str) -> str | None:
    match = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


def _precision(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


def _consistent_precision(values: tuple[str, ...]) -> int | None:
    precisions = {_precision(value) for value in values}
    if len(precisions) != 1:
        return None
    return next(iter(precisions))


def _fit_ratio_interval(pairs: list[tuple[Decimal, Decimal]] | tuple[tuple[Decimal, Decimal], ...], target_x: Decimal, precision: int) -> str | None:
    interval: tuple[Decimal, Decimal] | None = None
    ratios: list[Decimal] = []
    for x, y in pairs:
        if x == 0:
            if _format_decimal(Decimal(0), precision) != _format_decimal(y, precision):
                return None
            continue
        ratios.append(y / x)
        low_y, high_y = _rounding_interval(y, precision)
        low_c = low_y / x
        high_c = high_y / x
        if low_c > high_c:
            low_c, high_c = high_c, low_c
        interval = (low_c, high_c) if interval is None else (max(interval[0], low_c), min(interval[1], high_c))
        if interval[0] > interval[1]:
            return None
    if interval is None:
        return None

    low, high = interval
    ratios = sorted(ratios)
    candidate = ratios[len(ratios) // 2]
    if candidate < low or candidate > high:
        candidate = (low + high) / Decimal(2)
    if not all(_format_decimal(x * candidate, precision) == _format_decimal(y, precision) for x, y in pairs):
        return None
    return _format_decimal(target_x * candidate, precision)


def _fit_affine_candidates(pairs: list[tuple[Decimal, Decimal]], target_x: Decimal, precision: int) -> set[str]:
    targets: set[str] = set()
    for left_index, (x1, y1) in enumerate(pairs):
        for x2, y2 in pairs[left_index + 1 :]:
            if x1 == x2:
                continue
            a = (y2 - y1) / (x2 - x1)
            b = y1 - (a * x1)
            if all(_format_decimal((a * x) + b, precision) == _format_decimal(y, precision) for x, y in pairs):
                targets.add(_format_decimal((a * target_x) + b, precision))
    return targets


def _rounding_interval(value: Decimal, precision: int) -> tuple[Decimal, Decimal]:
    unit = Decimal(1) if precision == 0 else Decimal(1).scaleb(-precision)
    half = unit / Decimal(2)
    return value - half, value + half


def _format_decimal(value: Decimal, places: int) -> str:
    quant = Decimal("1") if places == 0 else Decimal("1." + ("0" * places))
    try:
        return str(value.quantize(quant, rounding=ROUND_HALF_UP))
    except InvalidOperation as exc:
        raise CompetitionRunnerError("decimal formatting failed.") from exc


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")


def _write_candidate_submission(path: Path, predictions: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "answer"))
        writer.writeheader()
        for row in predictions:
            writer.writerow({"id": row["problem_id"], "answer": row["answer"]})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise CompetitionRunnerError("JSONL artifact row is not an object.")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CompetitionRunnerError("JSON artifact is not an object.")
    return payload


def _record_to_dict(record: Any) -> dict[str, Any]:
    return {item.name: getattr(record, item.name) for item in fields(record)}


def _count_map(values: Mapping[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(values.items())}


def _require_text(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise CompetitionRunnerError(f"{field_name} must be non-empty.")
    return text


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = _payload_hash(instance, hash_field)
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise CompetitionRunnerError(f"{hash_field} does not match payload.")


def _validate_hash(instance: object, hash_field: str) -> None:
    if getattr(instance, hash_field) != _payload_hash(instance, hash_field):
        raise CompetitionRunnerError(f"{hash_field} does not match payload.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "CompetitionPrediction",
    "CompetitionRunReport",
    "CompetitionRunnerError",
    "run_competition_test_predictions",
    "run_competition_train_eval",
    "validate_competition_run_report",
]

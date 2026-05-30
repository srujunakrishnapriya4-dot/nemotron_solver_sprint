"""Conservative parser for simple example-target prompt structures."""

from __future__ import annotations

import re

from nemotron_engine.core.schemas import (
    CanonicalProblem,
    CanonicalizationError,
    DomainKind,
    DomainSignature,
    ExamplePair,
    ParsePermission,
    SymbolTable,
    TargetQuery,
    stable_hash,
)


_IO_RE = re.compile(r"^\s*(?:Input\s*:\s*)?(?P<input>.+?)\s*->\s*(?:Output\s*:\s*)?(?P<output>.+?)\s*$", re.I)
_TARGET_RE = re.compile(r"^\s*(?:Target\s*:\s*)?(?:Input\s*:\s*)?(?P<input>.+?)\s*->\s*(?P<output>\?|.+?)\s*$", re.I)
_TARGET_PREFIX_RE = re.compile(r"^\s*Target\s*:", re.I)
_INT_RE = re.compile(r"^[+-]?(0|[1-9]\d*)$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\.\d+)$")
_FRACTION_RE = re.compile(r"^[+-]?\d+/[+-]?\d+$")
_SYMBOL_RE = re.compile(r"^[A-Za-z]+$")


def canonicalize_prompt(problem_id: str, raw_prompt: str, *, permissive: bool = False) -> CanonicalProblem:
    text = str(raw_prompt or "").strip()
    if not text:
        raise CanonicalizationError("Prompt is empty.")

    examples: list[ExamplePair] = []
    target: TargetQuery | None = None
    warnings: list[str] = []
    unclear_lines: list[str] = []
    in_target_block = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower() in {"examples:", "examples"}:
            in_target_block = False
            continue
        if line.lower() in {"target:", "target"}:
            in_target_block = True
            continue
        match = _TARGET_RE.match(line)
        if not match:
            unclear_lines.append(line)
            continue
        lhs = _clean_endpoint(match.group("input"))
        rhs = _clean_endpoint(match.group("output"))
        has_explicit_target_prefix = bool(_TARGET_PREFIX_RE.match(line))
        is_target = has_explicit_target_prefix or in_target_block or rhs == "?"
        if is_target:
            if rhs != "?":
                target = TargetQuery(input_value=lhs, expected_output=rhs, certain=True)
            else:
                target = TargetQuery(input_value=lhs, expected_output=None, certain=True)
            in_target_block = False
        else:
            examples.append(ExamplePair(input_value=lhs, output_value=rhs))

    if unclear_lines:
        warnings.append("unclear_lines:" + "|".join(unclear_lines))
    if not examples:
        raise CanonicalizationError("No examples found in prompt.")
    if target is None:
        if not permissive:
            raise CanonicalizationError("No target query found in prompt.")
        warnings.append("missing_target_permissive")
        target = TargetQuery(input_value="UNKNOWN", expected_output=None, certain=False)

    input_domain = infer_domain(examples[0].input_value)
    output_domain = infer_domain(examples[0].output_value)
    symbol_table = build_symbol_table(text)
    high_confidence = bool(examples and target and target.certain and not unclear_lines)
    parse_confidence = 0.96 if high_confidence else 0.55 if permissive else 0.80
    if not high_confidence and not permissive:
        raise CanonicalizationError("Prompt has uncertain structure.")
    parse_permission = (
        ParsePermission.SOLVER_ALLOWED
        if parse_confidence >= 0.92 and target.certain
        else ParsePermission.STRESS_EVAL_ONLY
        if permissive
        else ParsePermission.NO_SFT
    )
    round_trip_score = _initial_round_trip_score(examples=examples, target=target)
    parse_payload = {
        "problem_id": problem_id,
        "examples": examples,
        "target": target,
        "input_domain": input_domain,
        "output_domain": output_domain,
        "warnings": tuple(warnings),
    }
    return CanonicalProblem(
        problem_id=problem_id,
        raw_prompt=text,
        examples=tuple(examples),
        target=target,
        input_domain=input_domain,
        output_domain=output_domain,
        symbol_table=symbol_table,
        parse_confidence=parse_confidence,
        round_trip_score=round_trip_score,
        parse_hash=stable_hash(parse_payload),
        parser_warnings=tuple(warnings),
        parse_permission=parse_permission,
    )


def infer_domain(value: str) -> DomainSignature:
    raw = str(value).strip()
    normalized = raw
    if _FRACTION_RE.fullmatch(raw):
        kind = DomainKind.FRACTION
    elif _DECIMAL_RE.fullmatch(raw):
        kind = DomainKind.DECIMAL
    elif set(raw) <= {"0", "1"} and len(raw) > 1:
        kind = DomainKind.BITSTRING
    elif _INT_RE.fullmatch(raw):
        kind = DomainKind.INTEGER
        normalized = str(int(raw))
    elif raw.isdigit():
        kind = DomainKind.DIGIT_SEQUENCE
    elif _SYMBOL_RE.fullmatch(raw):
        kind = DomainKind.SYMBOL_SEQUENCE
    elif len(raw.split()) > 1:
        kind = DomainKind.TOKEN_SEQUENCE
    elif "=" in raw:
        kind = DomainKind.EQUATION
    else:
        kind = DomainKind.UNKNOWN
    return DomainSignature(kind=kind, raw_value=raw, normalized_value=normalized)


def build_symbol_table(text: str) -> SymbolTable:
    symbols = sorted({char for char in str(text) if not char.isalnum() and not char.isspace()})
    mapping = {symbol: symbol for symbol in symbols}
    return SymbolTable(symbols=mapping, normalized_symbols=mapping)


def _clean_endpoint(value: str) -> str:
    text = str(value).strip()
    for prefix in ("Input:", "Output:", "Target:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return text


def _initial_round_trip_score(*, examples: list[ExamplePair], target: TargetQuery) -> float:
    if not examples or not target.input_value:
        return 0.0
    return 0.98 if target.certain else 0.70


__all__ = [
    "build_symbol_table",
    "canonicalize_prompt",
    "infer_domain",
]

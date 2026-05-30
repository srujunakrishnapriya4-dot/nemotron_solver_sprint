"""Typed answer canonicalization for clustering and final selection.

This module intentionally separates strong equivalence-preserving
canonicalization from weak formatting/heuristic cleanup. Downstream
aggregation can therefore cluster exact mathematical matches without
silently overclaiming equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
import re
from typing import Iterable, Sequence

import sympy as sp
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from src.common.constants import ANSWER_MAX, ANSWER_MIN


_SYMPY_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FRACTION_RE = re.compile(r"^[+-]?\d+\s*/\s*[+-]?\d+$")
_DECIMAL_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)
_COMMON_WRAPPER_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|ans|output|result|therefore|thus|so)\s*(?:is|=|:)\s*(.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
_EXPLICIT_INTEGER_EQ_RE = re.compile(r"^\s*(?:[a-zA-Z][a-zA-Z0-9_]*|n|x|y|z)\s*=\s*([+-]?\d+)\s*$")
_TOP_LEVEL_RELATION_RE = re.compile(r"(?<![<>=!])(<=|>=|!=|=|<|>)(?![<>=])")
_TOP_LEVEL_OR_RE = re.compile(r"\b(?:or|and/or)\b", re.IGNORECASE)
_AMBIGUOUS_SNIPPETS = (
    (r"\\pm|\u00b1", "plus/minus notation is ambiguous."),
    (r"\u2248|~|\u223c", "approximate-answer notation is not exact."),
    (r"\.\.\.|\u2026", "ellipsis notation is ambiguous."),
)
_UNSUPPORTED_SNIPPETS = (
    (r"\\begin\{", "matrix/cases environments are not supported for canonical answer clustering."),
    (r"%", "percent notation is not canonicalized safely."),
)
_UNICODE_REPLACEMENTS = {
    "\u2212": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u00d7": "*",
    "\u00f7": "/",
    "\u2215": "/",
}
_LATEX_LITERAL_REPLACEMENTS = {
    r"\cdot": "*",
    r"\times": "*",
    r"\div": "/",
    r"\left": "",
    r"\right": "",
    r"\,": " ",
    r"\;": " ",
    r"\!": "",
    r"\{": "{",
    r"\}": "}",
    r"\[": "[",
    r"\]": "]",
    r"\(": "(",
    r"\)": ")",
}
_WRAPPING_COMMANDS = ("boxed", "text", "mathrm", "operatorname")
_LIST_HINTS = ("list", "sequence", "vector")
_TUPLE_HINTS = ("tuple", "pair", "ordered", "coordinate", "point")


class CanonicalizationStatus(str, Enum):
    SUCCESS = "success"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    EMPTY = "empty"


class CanonicalizationStrength(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class CanonicalValueKind(str, Enum):
    INTEGER = "integer"
    RATIONAL = "rational"
    EXPRESSION = "expression"
    SET = "set"
    TUPLE = "tuple"
    LIST = "list"
    TEXT = "text"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class CanonicalizationMethod(str, Enum):
    IDENTITY = "identity"
    WHITESPACE_NORMALIZATION = "whitespace_normalization"
    FORMAT_NORMALIZATION = "format_normalization"
    LATEX_NORMALIZATION = "latex_normalization"
    HEURISTIC_ANSWER_EXTRACTION = "heuristic_answer_extraction"
    INTEGER_NORMALIZATION = "integer_normalization"
    RATIONAL_NORMALIZATION = "rational_normalization"
    NUMERIC_EXPRESSION_SIMPLIFICATION = "numeric_expression_simplification"
    SAFE_EXPRESSION_SIMPLIFICATION = "safe_expression_simplification"
    STRUCTURAL_EXPRESSION_NORMALIZATION = "structural_expression_normalization"
    SET_NORMALIZATION = "set_normalization"
    TUPLE_NORMALIZATION = "tuple_normalization"
    LIST_NORMALIZATION = "list_normalization"
    WEAK_TEXT_NORMALIZATION = "weak_text_normalization"
    AMBIGUITY_DETECTED = "ambiguity_detected"
    UNSUPPORTED_FORM = "unsupported_form"


@dataclass(frozen=True)
class CanonicalizedAnswer:
    original_text: str
    normalized_text: str
    canonical_text: str | None
    status: CanonicalizationStatus
    strength: CanonicalizationStrength
    value_kind: CanonicalValueKind
    method: CanonicalizationMethod
    methods_applied: tuple[CanonicalizationMethod, ...] = field(default_factory=tuple)
    exact: bool = False
    ambiguous: bool = False
    reason: str | None = None
    components: tuple[str, ...] = field(default_factory=tuple)

    @property
    def supports_equivalence(self) -> bool:
        return (
            self.status is CanonicalizationStatus.SUCCESS
            and self.strength is CanonicalizationStrength.STRONG
            and bool(self.canonical_text)
        )

    def cluster_key(self, *, require_strong: bool = True) -> str | None:
        if self.status is not CanonicalizationStatus.SUCCESS:
            return None
        if require_strong and self.strength is not CanonicalizationStrength.STRONG:
            return None
        return self.canonical_text


@dataclass(frozen=True)
class _CollectionSpec:
    kind: CanonicalValueKind
    items: tuple[str, ...]
    ambiguous: bool = False
    reason: str | None = None



def canonicalize_answer(
    answer: str | None,
    *,
    answer_type: str | None = None,
    allow_heuristic: bool = True,
) -> CanonicalizedAnswer:
    """Canonicalize one answer while preserving provenance and strength."""

    original_text = "" if answer is None else str(answer)
    if not original_text.strip():
        return _result(
            original_text=original_text,
            normalized_text="",
            canonical_text=None,
            status=CanonicalizationStatus.EMPTY,
            strength=CanonicalizationStrength.NONE,
            value_kind=CanonicalValueKind.EMPTY,
            method=CanonicalizationMethod.IDENTITY,
            exact=False,
            ambiguous=False,
            reason="Answer is empty.",
        )

    methods: list[CanonicalizationMethod] = []
    normalized = _normalize_input_text(original_text, methods)
    if not normalized:
        return _result(
            original_text=original_text,
            normalized_text="",
            canonical_text=None,
            status=CanonicalizationStatus.EMPTY,
            strength=CanonicalizationStrength.NONE,
            value_kind=CanonicalValueKind.EMPTY,
            method=CanonicalizationMethod.IDENTITY,
            methods_applied=methods,
            exact=False,
            ambiguous=False,
            reason="Answer is empty after normalization.",
        )

    if allow_heuristic:
        extracted = _strip_common_answer_wrapper(normalized)
        if extracted != normalized:
            methods.append(CanonicalizationMethod.HEURISTIC_ANSWER_EXTRACTION)
            normalized = extracted

    explicit_integer = _extract_explicit_integer_binding(normalized, answer_type=answer_type)
    if explicit_integer is not None:
        methods.append(CanonicalizationMethod.INTEGER_NORMALIZATION)
        return _result(
            original_text=original_text,
            normalized_text=normalized,
            canonical_text=explicit_integer,
            status=CanonicalizationStatus.SUCCESS,
            strength=CanonicalizationStrength.STRONG,
            value_kind=CanonicalValueKind.INTEGER,
            method=CanonicalizationMethod.INTEGER_NORMALIZATION,
            methods_applied=methods,
            exact=True,
            ambiguous=False,
            reason="Explicit integer answer binding normalized safely.",
        )

    issue = _detect_explicit_issue(normalized)
    if issue is not None:
        return _result(
            original_text=original_text,
            normalized_text=normalized,
            canonical_text=None,
            status=issue[0],
            strength=CanonicalizationStrength.NONE,
            value_kind=CanonicalValueKind.UNKNOWN,
            method=issue[1],
            methods_applied=methods + [issue[1]],
            exact=False,
            ambiguous=issue[0] is CanonicalizationStatus.AMBIGUOUS,
            reason=issue[2],
        )

    collection = _parse_collection_literal(normalized, answer_type=answer_type)
    if collection is not None:
        return _canonicalize_collection(
            original_text=original_text,
            normalized_text=normalized,
            spec=collection,
            answer_type=answer_type,
            inherited_methods=methods,
        )

    numeric = _canonicalize_numeric_literal(
        original_text=original_text,
        normalized_text=normalized,
        inherited_methods=methods,
    )
    if numeric is not None:
        return numeric

    symbolic = _canonicalize_symbolic_expression(
        original_text=original_text,
        normalized_text=normalized,
        inherited_methods=methods,
    )
    if symbolic is not None:
        return symbolic

    return _result(
        original_text=original_text,
        normalized_text=normalized,
        canonical_text=normalized,
        status=CanonicalizationStatus.SUCCESS,
        strength=CanonicalizationStrength.WEAK,
        value_kind=CanonicalValueKind.TEXT,
        method=CanonicalizationMethod.WEAK_TEXT_NORMALIZATION,
        methods_applied=methods + [CanonicalizationMethod.WEAK_TEXT_NORMALIZATION],
        exact=False,
        ambiguous=False,
        reason="Only weak formatting normalization was possible; no mathematical equivalence was proven.",
    )


def canonicalize_answers(
    answers: Iterable[str | None],
    *,
    answer_type: str | None = None,
    allow_heuristic: bool = True,
) -> tuple[CanonicalizedAnswer, ...]:
    """Canonicalize a sequence of answers deterministically."""

    return tuple(
        canonicalize_answer(
            answer,
            answer_type=answer_type,
            allow_heuristic=allow_heuristic,
        )
        for answer in answers
    )


def canonicalize_answer_key(
    answer: str | None,
    *,
    answer_type: str | None = None,
    require_strong: bool = True,
    allow_heuristic: bool = True,
) -> str | None:
    """Compatibility helper for clustering modules that need one key."""

    return canonicalize_answer(
        answer,
        answer_type=answer_type,
        allow_heuristic=allow_heuristic,
    ).cluster_key(require_strong=require_strong)


def canonicalize_competition_answer(
    answer: str | None,
    *,
    answer_type: str | None = "non_negative_integer",
    allow_heuristic: bool = True,
    require_strong: bool = True,
) -> str | None:
    """Return a submission-safe canonical integer answer or ``None``."""

    result = canonicalize_answer(
        answer,
        answer_type=answer_type,
        allow_heuristic=allow_heuristic,
    )
    if result.status is not CanonicalizationStatus.SUCCESS or not result.canonical_text:
        return None
    if require_strong and result.strength is not CanonicalizationStrength.STRONG:
        return None

    text = result.canonical_text.strip()
    if not _INTEGER_RE.fullmatch(text):
        return None

    value = int(text)
    if value < ANSWER_MIN or value > ANSWER_MAX:
        return None
    return str(value)


def canonicalize_competition_answers(
    answers: Iterable[str | None],
    *,
    answer_type: str | None = "non_negative_integer",
    allow_heuristic: bool = True,
    require_strong: bool = True,
) -> tuple[str, ...]:
    """Canonicalize a sequence of competition answers and drop invalid outputs."""

    normalized: list[str] = []
    for answer in answers:
        canonical = canonicalize_competition_answer(
            answer,
            answer_type=answer_type,
            allow_heuristic=allow_heuristic,
            require_strong=require_strong,
        )
        if canonical is not None:
            normalized.append(canonical)
    return tuple(normalized)




def _extract_explicit_integer_binding(text: str, *, answer_type: str | None) -> str | None:
    hint = str(answer_type or "").lower()
    if hint and hint not in {"integer", "non_negative_integer", "positive_integer", "natural_number"}:
        return None
    candidate = text.strip().rstrip(".")
    match = _EXPLICIT_INTEGER_EQ_RE.fullmatch(candidate)
    if match is not None:
        return str(int(match.group(1)))
    boxed_match = re.fullmatch(r"\?boxed\{\s*([+-]?\d+)\s*\}", candidate)
    if boxed_match is not None:
        return str(int(boxed_match.group(1)))
    if _INTEGER_RE.fullmatch(candidate):
        return str(int(candidate))
    if re.fullmatch(r"[+-]?\d+\.0+", candidate):
        return str(int(Decimal(candidate)))
    return None


def _canonicalize_collection(
    *,
    original_text: str,
    normalized_text: str,
    spec: _CollectionSpec,
    answer_type: str | None,
    inherited_methods: Sequence[CanonicalizationMethod],
) -> CanonicalizedAnswer:
    if spec.ambiguous:
        method = _collection_method(spec.kind)
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=None,
            status=CanonicalizationStatus.AMBIGUOUS,
            strength=CanonicalizationStrength.NONE,
            value_kind=spec.kind,
            method=CanonicalizationMethod.AMBIGUITY_DETECTED,
            methods_applied=list(inherited_methods) + [method, CanonicalizationMethod.AMBIGUITY_DETECTED],
            exact=False,
            ambiguous=True,
            reason=spec.reason,
        )

    component_results = tuple(
        canonicalize_answer(item, answer_type=None, allow_heuristic=False)
        for item in spec.items
    )
    failed = next(
        (item for item in component_results if item.status is not CanonicalizationStatus.SUCCESS),
        None,
    )
    if failed is not None:
        reason = (
            f"Collection item '{failed.normalized_text or failed.original_text}' "
            f"could not be canonicalized safely: {failed.reason or failed.status.value}."
        )
        status = (
            CanonicalizationStatus.AMBIGUOUS
            if failed.status is CanonicalizationStatus.AMBIGUOUS
            else CanonicalizationStatus.UNSUPPORTED
        )
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=None,
            status=status,
            strength=CanonicalizationStrength.NONE,
            value_kind=spec.kind,
            method=CanonicalizationMethod.UNSUPPORTED_FORM,
            methods_applied=list(inherited_methods) + [_collection_method(spec.kind), CanonicalizationMethod.UNSUPPORTED_FORM],
            exact=False,
            ambiguous=status is CanonicalizationStatus.AMBIGUOUS,
            reason=reason,
        )

    components = tuple(item.canonical_text or item.normalized_text for item in component_results)
    if spec.kind is CanonicalValueKind.SET:
        ordered = tuple(sorted(set(components)))
        canonical_text = "{" + ", ".join(ordered) + "}"
    elif spec.kind is CanonicalValueKind.LIST:
        ordered = components
        canonical_text = "[" + ", ".join(ordered) + "]"
    else:
        ordered = components
        canonical_text = "(" + ", ".join(ordered) + ")"

    strength = (
        CanonicalizationStrength.STRONG
        if all(item.strength is CanonicalizationStrength.STRONG for item in component_results)
        else CanonicalizationStrength.WEAK
    )
    reason = None
    if strength is CanonicalizationStrength.WEAK:
        reason = "Collection canonicalization succeeded, but at least one element used weak normalization."
    methods = list(inherited_methods)
    methods.append(_collection_method(spec.kind))
    for item in component_results:
        methods.extend(item.methods_applied)
    return _result(
        original_text=original_text,
        normalized_text=normalized_text,
        canonical_text=canonical_text,
        status=CanonicalizationStatus.SUCCESS,
        strength=strength,
        value_kind=spec.kind,
        method=_collection_method(spec.kind),
        methods_applied=methods,
        exact=all(item.exact for item in component_results),
        ambiguous=False,
        reason=reason,
        components=ordered,
    )


def _canonicalize_numeric_literal(
    *,
    original_text: str,
    normalized_text: str,
    inherited_methods: Sequence[CanonicalizationMethod],
) -> CanonicalizedAnswer | None:
    compact = normalized_text.replace(" ", "")
    if _INTEGER_RE.fullmatch(compact):
        canonical = str(int(compact))
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=canonical,
            status=CanonicalizationStatus.SUCCESS,
            strength=CanonicalizationStrength.STRONG,
            value_kind=CanonicalValueKind.INTEGER,
            method=CanonicalizationMethod.INTEGER_NORMALIZATION,
            methods_applied=list(inherited_methods) + [CanonicalizationMethod.INTEGER_NORMALIZATION],
            exact=True,
            ambiguous=False,
        )

    if _FRACTION_RE.fullmatch(compact):
        numerator_text, denominator_text = compact.split("/", 1)
        denominator = int(denominator_text)
        if denominator == 0:
            return _result(
                original_text=original_text,
                normalized_text=normalized_text,
                canonical_text=None,
                status=CanonicalizationStatus.UNSUPPORTED,
                strength=CanonicalizationStrength.NONE,
                value_kind=CanonicalValueKind.RATIONAL,
                method=CanonicalizationMethod.UNSUPPORTED_FORM,
                methods_applied=list(inherited_methods) + [CanonicalizationMethod.UNSUPPORTED_FORM],
                exact=False,
                ambiguous=False,
                reason="Division by zero is not a canonical answer.",
            )
        fraction = Fraction(int(numerator_text), denominator)
        canonical, kind = _format_fraction(fraction)
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=canonical,
            status=CanonicalizationStatus.SUCCESS,
            strength=CanonicalizationStrength.STRONG,
            value_kind=kind,
            method=CanonicalizationMethod.RATIONAL_NORMALIZATION,
            methods_applied=list(inherited_methods) + [CanonicalizationMethod.RATIONAL_NORMALIZATION],
            exact=True,
            ambiguous=False,
        )

    if _DECIMAL_RE.fullmatch(compact):
        try:
            decimal_value = Decimal(compact)
        except InvalidOperation:
            return None
        fraction = Fraction(decimal_value)
        canonical, kind = _format_fraction(fraction)
        method = (
            CanonicalizationMethod.INTEGER_NORMALIZATION
            if kind is CanonicalValueKind.INTEGER
            else CanonicalizationMethod.RATIONAL_NORMALIZATION
        )
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=canonical,
            status=CanonicalizationStatus.SUCCESS,
            strength=CanonicalizationStrength.STRONG,
            value_kind=kind,
            method=method,
            methods_applied=list(inherited_methods) + [method],
            exact=True,
            ambiguous=False,
        )

    return None


def _canonicalize_symbolic_expression(
    *,
    original_text: str,
    normalized_text: str,
    inherited_methods: Sequence[CanonicalizationMethod],
) -> CanonicalizedAnswer | None:
    try:
        expr = parse_expr(normalized_text, transformations=_SYMPY_TRANSFORMS, evaluate=True)
    except Exception:
        return None

    if isinstance(expr, sp.logic.boolalg.Boolean) or isinstance(expr, Relational):
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=None,
            status=CanonicalizationStatus.UNSUPPORTED,
            strength=CanonicalizationStrength.NONE,
            value_kind=CanonicalValueKind.UNKNOWN,
            method=CanonicalizationMethod.UNSUPPORTED_FORM,
            methods_applied=list(inherited_methods) + [CanonicalizationMethod.UNSUPPORTED_FORM],
            exact=False,
            ambiguous=False,
            reason="Relational or boolean expressions are not canonicalized as final answers.",
        )

    simplified = sp.simplify(expr)
    if _contains_non_finite_value(simplified):
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=None,
            status=CanonicalizationStatus.UNSUPPORTED,
            strength=CanonicalizationStrength.NONE,
            value_kind=CanonicalValueKind.UNKNOWN,
            method=CanonicalizationMethod.UNSUPPORTED_FORM,
            methods_applied=list(inherited_methods) + [CanonicalizationMethod.UNSUPPORTED_FORM],
            exact=False,
            ambiguous=False,
            reason="Non-finite symbolic values are not canonicalized.",
        )

    if not simplified.free_symbols:
        exact_value = sp.nsimplify(simplified, rational=True)
        canonical_text, kind = _format_sympy_scalar(exact_value)
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=canonical_text,
            status=CanonicalizationStatus.SUCCESS,
            strength=CanonicalizationStrength.STRONG,
            value_kind=kind,
            method=CanonicalizationMethod.NUMERIC_EXPRESSION_SIMPLIFICATION,
            methods_applied=list(inherited_methods) + [CanonicalizationMethod.NUMERIC_EXPRESSION_SIMPLIFICATION],
            exact=True,
            ambiguous=False,
        )

    if _is_safe_algebraic_expression(simplified):
        canonical_expr = _normalize_safe_expression(simplified)
        canonical_text = sp.sstr(canonical_expr, order="lex")
        return _result(
            original_text=original_text,
            normalized_text=normalized_text,
            canonical_text=canonical_text,
            status=CanonicalizationStatus.SUCCESS,
            strength=CanonicalizationStrength.STRONG,
            value_kind=CanonicalValueKind.EXPRESSION,
            method=CanonicalizationMethod.SAFE_EXPRESSION_SIMPLIFICATION,
            methods_applied=list(inherited_methods) + [CanonicalizationMethod.SAFE_EXPRESSION_SIMPLIFICATION],
            exact=False,
            ambiguous=False,
        )

    canonical_text = sp.sstr(expr, order="lex")
    return _result(
        original_text=original_text,
        normalized_text=normalized_text,
        canonical_text=canonical_text,
        status=CanonicalizationStatus.SUCCESS,
        strength=CanonicalizationStrength.WEAK,
        value_kind=CanonicalValueKind.EXPRESSION,
        method=CanonicalizationMethod.STRUCTURAL_EXPRESSION_NORMALIZATION,
        methods_applied=list(inherited_methods) + [CanonicalizationMethod.STRUCTURAL_EXPRESSION_NORMALIZATION],
        exact=False,
        ambiguous=False,
        reason="Expression was parsed structurally, but only conservative formatting normalization was safe.",
    )


def _normalize_input_text(text: str, methods: list[CanonicalizationMethod]) -> str:
    normalized = str(text).strip()
    if normalized != text:
        methods.append(CanonicalizationMethod.WHITESPACE_NORMALIZATION)

    updated = "".join(_UNICODE_REPLACEMENTS.get(ch, ch) for ch in normalized)
    if updated != normalized:
        methods.append(CanonicalizationMethod.FORMAT_NORMALIZATION)
        normalized = updated

    updated = _strip_math_delimiters(normalized)
    if updated != normalized:
        methods.append(CanonicalizationMethod.LATEX_NORMALIZATION)
        normalized = updated

    updated = _normalize_latex_text(normalized)
    if updated != normalized:
        methods.append(CanonicalizationMethod.LATEX_NORMALIZATION)
        normalized = updated

    updated = _collapse_whitespace(normalized)
    if updated != normalized:
        methods.append(CanonicalizationMethod.WHITESPACE_NORMALIZATION)
        normalized = updated

    return normalized


def _strip_math_delimiters(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed and stripped:
        changed = False
        for left, right in (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if stripped.startswith(left) and stripped.endswith(right) and len(stripped) > len(left) + len(right):
                stripped = stripped[len(left) : -len(right)].strip()
                changed = True
    return stripped


def _normalize_latex_text(text: str) -> str:
    normalized = text
    for token, replacement in _LATEX_LITERAL_REPLACEMENTS.items():
        normalized = normalized.replace(token, replacement)
    normalized = _strip_wrapping_commands(normalized)
    normalized = _replace_latex_fractions(normalized)
    normalized = _replace_latex_sqrt(normalized)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    return normalized.strip()


def _strip_wrapping_commands(text: str) -> str:
    current = text.strip()
    while True:
        changed = False
        for command in _WRAPPING_COMMANDS:
            prefix = f"\\{command}" + "{"
            if current.startswith(prefix) and current.endswith("}"):
                payload, end_index = _extract_braced_payload(current, len(prefix) - 1)
                if payload is not None and end_index == len(current):
                    current = payload.strip()
                    changed = True
        if not changed:
            return current


def _replace_latex_fractions(text: str) -> str:
    for command in (r"\frac", r"\dfrac", r"\tfrac"):
        text = _replace_fraction_command(text, command)
    return text


def _replace_fraction_command(text: str, command: str) -> str:
    result: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text.startswith(command, i):
            numerator, after_num = _extract_braced_payload(text, i + len(command))
            denominator, after_den = _extract_braced_payload(text, after_num) if numerator is not None else (None, after_num)
            if numerator is not None and denominator is not None:
                result.append(f"(({numerator})/({denominator}))")
                i = after_den
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _replace_latex_sqrt(text: str) -> str:
    result: list[str] = []
    i = 0
    length = len(text)
    command = r"\sqrt"
    while i < length:
        if text.startswith(command, i):
            payload, end_index = _extract_braced_payload(text, i + len(command))
            if payload is not None:
                result.append(f"sqrt({payload})")
                i = end_index
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _extract_braced_payload(text: str, start: int) -> tuple[str | None, int]:
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    payload: list[str] = []
    i = start
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
            if depth > 1:
                payload.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(payload), i + 1
            payload.append(char)
        else:
            if depth >= 1:
                payload.append(char)
        i += 1
    return None, start


def _collapse_whitespace(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip())
    collapsed = re.sub(r"\s*,\s*", ", ", collapsed)
    collapsed = re.sub(r"\s*/\s*", "/", collapsed)
    return collapsed.strip()


def _strip_common_answer_wrapper(text: str) -> str:
    match = _COMMON_WRAPPER_RE.match(text)
    return match.group(1).strip() if match else text


def _detect_explicit_issue(
    text: str,
) -> tuple[CanonicalizationStatus, CanonicalizationMethod, str] | None:
    for pattern, reason in _AMBIGUOUS_SNIPPETS:
        if re.search(pattern, text):
            return (
                CanonicalizationStatus.AMBIGUOUS,
                CanonicalizationMethod.AMBIGUITY_DETECTED,
                reason,
            )
    for pattern, reason in _UNSUPPORTED_SNIPPETS:
        if re.search(pattern, text):
            return (
                CanonicalizationStatus.UNSUPPORTED,
                CanonicalizationMethod.UNSUPPORTED_FORM,
                reason,
            )
    if _TOP_LEVEL_RELATION_RE.search(text):
        return (
            CanonicalizationStatus.UNSUPPORTED,
            CanonicalizationMethod.UNSUPPORTED_FORM,
            "Relational statements are not canonicalized as final answers.",
        )
    if _contains_top_level_word(text, _TOP_LEVEL_OR_RE):
        return (
            CanonicalizationStatus.AMBIGUOUS,
            CanonicalizationMethod.AMBIGUITY_DETECTED,
            "Multiple candidate answers were detected.",
        )
    return None


def _parse_collection_literal(text: str, *, answer_type: str | None) -> _CollectionSpec | None:
    stripped = text.strip()
    if len(stripped) < 2:
        return None
    hint = (answer_type or "").strip().lower()

    if stripped.startswith("{") and stripped.endswith("}"):
        inner = stripped[1:-1].strip()
        if _contains_top_level_char(inner, "|") or _contains_top_level_char(inner, ":"):
            return _CollectionSpec(
                kind=CanonicalValueKind.SET,
                items=(),
                ambiguous=True,
                reason="Set-builder notation is not canonicalized safely.",
            )
        items = () if not inner else tuple(_split_top_level(inner))
        return _CollectionSpec(kind=CanonicalValueKind.SET, items=items)

    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        items = () if not inner else tuple(_split_top_level(inner))
        return _CollectionSpec(kind=CanonicalValueKind.LIST, items=items)

    if stripped.startswith("(") and stripped.endswith(")") and _contains_top_level_char(stripped[1:-1], ","):
        inner = stripped[1:-1].strip()
        items = tuple(_split_top_level(inner))
        if len(items) == 2 and not _hint_matches(hint, _TUPLE_HINTS + _LIST_HINTS):
            return _CollectionSpec(
                kind=CanonicalValueKind.TUPLE,
                items=items,
                ambiguous=True,
                reason="Parenthesized two-item answers are ambiguous without a tuple/coordinate answer_type hint.",
            )
        return _CollectionSpec(kind=CanonicalValueKind.TUPLE, items=items)

    return None


def _split_top_level(text: str) -> list[str]:
    depth = 0
    current: list[str] = []
    parts: list[str] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _contains_top_level_char(text: str, needle: str) -> bool:
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == needle and depth == 0:
            return True
    return False


def _contains_top_level_word(text: str, pattern: re.Pattern[str]) -> bool:
    depth = 0
    chars: list[str] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        chars.append(char if depth == 0 else " ")
    return pattern.search("".join(chars)) is not None


def _hint_matches(hint: str, needles: Sequence[str]) -> bool:
    return any(token in hint for token in needles)


def _contains_non_finite_value(expr: sp.Basic) -> bool:
    return bool(expr.has(sp.zoo) or expr.has(sp.oo) or expr.has(-sp.oo) or expr.has(sp.nan))


def _is_safe_algebraic_expression(expr: sp.Expr) -> bool:
    if expr.has(sp.Float):
        return False
    if expr.atoms(sp.Function):
        return False
    for power in expr.atoms(sp.Pow):
        exponent = power.exp
        if exponent.is_integer is False:
            return False
        if exponent.is_integer is None and exponent.free_symbols:
            return False
    return True


def _normalize_safe_expression(expr: sp.Expr) -> sp.Expr:
    normalized = sp.cancel(sp.together(expr))
    normalized = sp.factor_terms(normalized)
    return normalized


def _format_fraction(value: Fraction) -> tuple[str, CanonicalValueKind]:
    if value.denominator == 1:
        return str(value.numerator), CanonicalValueKind.INTEGER
    return f"{value.numerator}/{value.denominator}", CanonicalValueKind.RATIONAL


def _format_sympy_scalar(value: sp.Expr) -> tuple[str, CanonicalValueKind]:
    if isinstance(value, sp.Integer):
        return str(int(value)), CanonicalValueKind.INTEGER
    if isinstance(value, sp.Rational):
        if int(value.q) == 1:
            return str(int(value.p)), CanonicalValueKind.INTEGER
        return f"{int(value.p)}/{int(value.q)}", CanonicalValueKind.RATIONAL
    return sp.sstr(value, order="lex"), CanonicalValueKind.EXPRESSION


def _collection_method(kind: CanonicalValueKind) -> CanonicalizationMethod:
    if kind is CanonicalValueKind.SET:
        return CanonicalizationMethod.SET_NORMALIZATION
    if kind is CanonicalValueKind.LIST:
        return CanonicalizationMethod.LIST_NORMALIZATION
    return CanonicalizationMethod.TUPLE_NORMALIZATION


def _result(
    *,
    original_text: str,
    normalized_text: str,
    canonical_text: str | None,
    status: CanonicalizationStatus,
    strength: CanonicalizationStrength,
    value_kind: CanonicalValueKind,
    method: CanonicalizationMethod,
    methods_applied: Sequence[CanonicalizationMethod] = (),
    exact: bool,
    ambiguous: bool,
    reason: str | None = None,
    components: Sequence[str] = (),
) -> CanonicalizedAnswer:
    seen: set[CanonicalizationMethod] = set()
    ordered_methods: list[CanonicalizationMethod] = []
    for item in methods_applied:
        if item not in seen:
            ordered_methods.append(item)
            seen.add(item)
    if method not in seen:
        ordered_methods.append(method)
    return CanonicalizedAnswer(
        original_text=original_text,
        normalized_text=normalized_text,
        canonical_text=canonical_text,
        status=status,
        strength=strength,
        value_kind=value_kind,
        method=method,
        methods_applied=tuple(ordered_methods),
        exact=exact,
        ambiguous=ambiguous,
        reason=reason,
        components=tuple(components),
    )


__all__ = [
    "CanonicalizationMethod",
    "CanonicalizationStatus",
    "CanonicalizationStrength",
    "CanonicalValueKind",
    "CanonicalizedAnswer",
    "canonicalize_answer",
    "canonicalize_answer_key",
    "canonicalize_answers",
    "canonicalize_competition_answer",
    "canonicalize_competition_answers",
]

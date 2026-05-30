from __future__ import annotations

import re
from dataclasses import dataclass

from src.common.constants import PARSING
from src.common.schemas import LatexSpan, NormalizationChange, NormalizationReport


_UNICODE_REPLACEMENTS = {
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "−": "-",
    "–": "-",
    "—": "-",
    "×": "*",
    "·": "*",
    "⋅": "*",
    "÷": "/",
    "∈": " in ",
    "∉": " notin ",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∞": "inf",
    "√": r"\sqrt",
    "≡": r"\equiv",
    "∪": r"\cup",
    "∩": r"\cap",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "⊃": r"\supset",
    "∅": r"\emptyset",
    "∀": r"\forall",
    "∃": r"\exists",
    "→": r"\to",
    "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow",
    "ℤ": "integers",
    "ℕ": "natural_numbers",
    "ℚ": "rational_numbers",
    "ℝ": "real_numbers",
    "ℂ": "complex_numbers",
    "⊥": r"\perp",
    "∥": r"\parallel",
    "⟂": r"\perp",
    "°": " deg",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
}

_KNOWN_COMMAND_REWRITES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\\dfrac"), r"\\frac", "fraction"),
    (re.compile(r"\\tfrac"), r"\\frac", "fraction"),
    (re.compile(r"\\cfrac"), r"\\frac", "fraction"),
    (re.compile(r"\\left"), "", "delimiter"),
    (re.compile(r"\\right"), "", "delimiter"),
    (re.compile(r"\\!"), "", "spacing"),
    (re.compile(r"\\,"), " ", "spacing"),
    (re.compile(r"\\;"), " ", "spacing"),
    (re.compile(r"\\:"), " ", "spacing"),
    (re.compile(r"\\quad"), " ", "spacing"),
    (re.compile(r"\\qquad"), " ", "spacing"),
    (re.compile(r"\\cdot"), "*", "spacing"),
    (re.compile(r"\\times"), "*", "spacing"),
    (re.compile(r"\\div"), "/", "spacing"),
    (re.compile(r"\\mid"), "|", "set_interval"),
    (re.compile(r"\\colon"), ":", "spacing"),
    (re.compile(r"\\ldots"), "...", "spacing"),
    (re.compile(r"\\dots"), "...", "spacing"),
)

_MATH_SPAN_PATTERN = re.compile(
    r"(?P<display>\$\$.*?\$\$)|(?P<inline>\$[^$\n]+\$)|(?P<bracket>\\\[.*?\\\])|(?P<paren>\\\(.*?\\\))",
    re.DOTALL,
)


@dataclass(frozen=True)
class NormalizationResult:
    original_text: str
    normalized_text: str
    spans: tuple[LatexSpan, ...]
    report: NormalizationReport


def _apply_change(stage: str, text: str, fn, changes: list[NormalizationChange]) -> str:
    new_text = fn(text)
    if new_text != text:
        changes.append(NormalizationChange(stage=stage, before=text, after=new_text))
    return new_text


def _replace_unicode(text: str) -> str:
    out = text
    for src, dst in _UNICODE_REPLACEMENTS.items():
        out = out.replace(src, dst)
    return out


def _normalize_known_commands(text: str) -> str:
    out = text
    for pattern, repl, _bucket in _KNOWN_COMMAND_REWRITES:
        out = pattern.sub(repl, out)
    out = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathbf\s*\{([^{}]*)\}", r"\1", out)
    return out


def _normalize_frac(text: str) -> str:
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    out = text
    for _ in range(PARSING.NORMALIZATION_MAX_PASSES):
        new_out = pattern.sub(r"((\1)/(\2))", out)
        if new_out == out:
            break
        out = new_out
    return out


def _normalize_binom(text: str) -> str:
    pattern = re.compile(r"\\binom\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    out = text
    for _ in range(PARSING.NORMALIZATION_MAX_PASSES):
        new_out = pattern.sub(r"binom(\1, \2)", out)
        if new_out == out:
            break
        out = new_out
    return out


def _normalize_sqrt(text: str) -> str:
    return re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)


def _normalize_mod(text: str) -> str:
    text = re.sub(r"\\pmod\{([^{}]+)\}", r"(mod \1)", text)
    text = re.sub(r"\\pmod\s+([A-Za-z0-9_()+\-*/^]+)", r"(mod \1)", text)
    text = re.sub(r"\\mod\{([^{}]+)\}", r"(mod \1)", text)
    text = re.sub(r"\\mod\s+([A-Za-z0-9_()+\-*/^]+)", r"(mod \1)", text)
    text = re.sub(r"\\equiv", "≡", text)
    text = re.sub(r"\(\s*mod\s+([^()]+?)\s*\)", lambda m: f"(mod {m.group(1).strip()})", text)
    return text


def _normalize_power(text: str) -> str:
    text = re.sub(r"([A-Za-z0-9_()]+)\^\{([^{}]+)\}", r"\1^(\2)", text)
    text = re.sub(r"([A-Za-z0-9_()]+)\^([A-Za-z0-9+\-]+)", r"\1^(\2)", text)
    return text


def _normalize_indexed_symbols(text: str) -> str:
    text = re.sub(r"([A-Za-z][A-Za-z0-9]*)_\{([^{}]+)\}", r"\1_(\2)", text)
    text = re.sub(r"([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9+\-]+)", r"\1_(\2)", text)
    return text


def _normalize_floor_ceil(text: str) -> str:
    text = re.sub(r"\\lfloor\s*([^\\]+?)\s*\\rfloor", r"floor(\1)", text)
    text = re.sub(r"\\lceil\s*([^\\]+?)\s*\\rceil", r"ceil(\1)", text)
    return text


def _normalize_geometry(text: str) -> str:
    text = re.sub(r"\\overline\s*\{\s*([A-Za-z]{2,})\s*\}", r"segment(\1)", text)
    text = re.sub(r"\\angle\s*([A-Za-z]{3})", r"angle(\1)", text)
    text = re.sub(r"\\triangle\s*([A-Za-z]{3})", r"triangle(\1)", text)
    text = re.sub(r"\\parallel", "||", text)
    text = re.sub(r"\\perp", "perp", text)
    return text


def _normalize_sets_and_intervals(text: str) -> str:
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = text.replace(r"\lbrace", "{").replace(r"\rbrace", "}")
    text = text.replace(r"\lbrack", "[").replace(r"\rbrack", "]")
    text = text.replace(r"\langle", "<").replace(r"\rangle", ">")
    text = text.replace(r"\emptyset", "emptyset")
    text = re.sub(r"\\in", " in ", text)
    text = re.sub(r"\\notin", " notin ", text)
    text = re.sub(r"\\subseteq", " subseteq ", text)
    text = re.sub(r"\\subset", " subset ", text)
    text = re.sub(r"\\supseteq", " supseteq ", text)
    text = re.sub(r"\\supset", " supset ", text)
    text = re.sub(r"\\cup", " union ", text)
    text = re.sub(r"\\cap", " inter ", text)
    text = re.sub(r"\{\s*([^{}]+?)\s*\|\s*([^{}]+?)\}", r"{ \1 | \2 }", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text


def _normalize_domains(text: str) -> str:
    replacements = (
        (r"\bnon\s*-?negative integers?\b", "nonnegative_integers"),
        (r"\bpositive integers?\b", "positive_integers"),
        (r"\bnatural numbers?\b", "natural_numbers"),
        (r"\breal numbers?\b", "real_numbers"),
        (r"\brational numbers?\b", "rational_numbers"),
        (r"\bintegers?\b", "integers"),
    )
    out = text
    for pat, repl in replacements:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([=<>+\-*/,:;])\s*", r" \1 ", text)
    text = re.sub(r"<\s*=", "<=", text)
    text = re.sub(r">\s*=", ">=", text)
    text = re.sub(r"!\s*=", "!=", text)
    text = re.sub(r"\(\s*mod\s+([^()]+?)\s*\)", lambda m: f"(mod {m.group(1).strip()})", text)
    text = re.sub(r"\s+([,;:)\]])", r"\1", text)
    text = re.sub(r"([([{}])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_delimiters(raw: str) -> tuple[str, str]:
    if raw.startswith("$$") and raw.endswith("$$"):
        return "$$", "$$"
    if raw.startswith("$") and raw.endswith("$"):
        return "$", "$"
    if raw.startswith(r"\[") and raw.endswith(r"\]"):
        return r"\[", r"\]"
    if raw.startswith(r"\(") and raw.endswith(r"\)"):
        return r"\(", r"\)"
    return "", ""


def _normalize_segment(text: str) -> str:
    out = text
    out = _replace_unicode(out)
    out = _normalize_known_commands(out)
    out = _normalize_frac(out)
    out = _normalize_binom(out)
    out = _normalize_sqrt(out)
    out = _normalize_power(out)
    out = _normalize_indexed_symbols(out)
    out = _normalize_mod(out)
    out = _normalize_floor_ceil(out)
    out = _normalize_geometry(out)
    out = _normalize_sets_and_intervals(out)
    out = _normalize_domains(out)
    out = _normalize_spacing(out)
    return out


class LatexNormalizer:
    def normalize(self, text: str) -> str:
        return normalize_math_text(text)

    def normalize_with_report(self, text: str) -> NormalizationResult:
        return normalize_math_text_with_report(text)

    def extract_normalized_spans(self, text: str) -> list[LatexSpan]:
        return extract_latex_spans(text)

    def to_sympy_compatible(self, expr: str) -> str:
        normalized = normalize_math_text(expr)
        normalized = normalized.replace("^(", "**(")
        normalized = re.sub(r"(?<!\*)\^(?=\()", "**", normalized)
        normalized = re.sub(r"(?<!\*)\^(?=[A-Za-z0-9])", "**", normalized)
        normalized = normalized.replace("binom(", "binomial(")
        normalized = normalized.replace("segment(", "segment_")
        normalized = normalized.replace("angle(", "angle_")
        normalized = normalized.replace("triangle(", "triangle_")
        normalized = normalized.replace(" (mod ", " mod ")
        return normalized


def extract_latex_spans(text: str) -> list[LatexSpan]:
    spans: list[LatexSpan] = []
    source = text or ""

    for match in _MATH_SPAN_PATTERN.finditer(source):
        raw = match.group(0)
        prefix, suffix = _detect_delimiters(raw)
        inner = raw[len(prefix) : len(raw) - len(suffix)] if suffix else raw[len(prefix) :]
        normalized = _normalize_segment(inner)
        spans.append(
            LatexSpan(
                raw_text=raw,
                normalized_text=normalized,
                start=match.start(),
                end=match.end(),
            )
        )

    if not spans:
        for match in re.finditer(r"\$[^$]+\$|\\[A-Za-z]+(?:\{[^{}]*\})*", source):
            raw = match.group(0)
            normalized = normalize_math_text(raw)
            spans.append(
                LatexSpan(
                    raw_text=raw,
                    normalized_text=normalized,
                    start=match.start(),
                    end=match.end(),
                )
            )

    return spans


def normalize_math_with_report(text: str) -> tuple[str, NormalizationReport]:
    original = text or ""
    changes: list[NormalizationChange] = []
    normalized = original

    normalized = _apply_change("unicode", normalized, _replace_unicode, changes)
    normalized = _apply_change("known_commands", normalized, _normalize_known_commands, changes)
    normalized = _apply_change("fraction", normalized, _normalize_frac, changes)
    normalized = _apply_change("binomial", normalized, _normalize_binom, changes)
    normalized = _apply_change("sqrt", normalized, _normalize_sqrt, changes)
    normalized = _apply_change("modular", normalized, _normalize_mod, changes)
    normalized = _apply_change("power", normalized, _normalize_power, changes)
    normalized = _apply_change("indexed_symbols", normalized, _normalize_indexed_symbols, changes)
    normalized = _apply_change("floor_ceil", normalized, _normalize_floor_ceil, changes)
    normalized = _apply_change("geometry", normalized, _normalize_geometry, changes)
    normalized = _apply_change("sets_intervals", normalized, _normalize_sets_and_intervals, changes)
    normalized = _apply_change("domains", normalized, _normalize_domains, changes)
    normalized = _apply_change("spacing", normalized, _normalize_spacing, changes)

    report = NormalizationReport(
        original_text=original,
        normalized_text=normalized,
        changes=changes,
        warnings=[],
    )
    return normalized, report


def normalize_math_text(text: str) -> str:
    return normalize_math_with_report(text)[0]


def normalize_math_text_with_report(text: str) -> NormalizationResult:
    normalized, report = normalize_math_with_report(text)
    spans = tuple(extract_latex_spans(text or ""))
    return NormalizationResult(
        original_text=text or "",
        normalized_text=normalized,
        spans=spans,
        report=report,
    )


__all__ = [
    "LatexNormalizer",
    "NormalizationResult",
    "extract_latex_spans",
    "normalize_math_text",
    "normalize_math_text_with_report",
    "normalize_math_with_report",
]

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
import hashlib
import inspect
import re
from typing import Any, Callable, Iterable, Mapping

from src.common.constants import PARSING
from src.common import schemas as S
from src.parsing.constraint_extractor import extract_constraints
from src.parsing.latex_normalizer import (
    LatexNormalizer,
    extract_latex_spans,
    normalize_math_text,
    normalize_math_with_report,
)
from src.parsing.symbol_table import SymbolTable


MODEL_TIMEOUT_FALLBACK = "model_timeout"
MODEL_EXCEPTION_FALLBACK = "model_exception"
MODEL_INVALID_OUTPUT_FALLBACK = "model_invalid_output"
DETERMINISTIC_ONLY_FALLBACK = "deterministic_only"


class ParseFailure(RuntimeError):
    """Raised when parsing cannot produce a valid canonical ParsedProblem."""


@dataclass(frozen=True)
class ModelParseHints:
    normalized_text: str | None = None
    variable_candidates: tuple[str, ...] = ()
    extra_constraint_texts: tuple[str, ...] = ()
    answer_type: Any | None = None
    domain_seed: Any | None = None
    archetype_cues: tuple[Any, ...] = ()
    objective: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawProblemUnits:
    variable_candidates: tuple[str, ...]
    domain_mentions: tuple[str, ...]
    quantifiers: tuple[Any, ...]
    objective_text: str | None
    target_mode: Any | None
    sentence_count: int


@dataclass(frozen=True)
class CanonicalParseIR:
    raw_text: str
    normalized_text: str
    latex_spans: tuple[Any, ...]
    raw_units: RawProblemUnits
    symbol_table: SymbolTable
    constraints: tuple[Any, ...]
    objective: Any | None
    answer_type: Any
    problem_domain_seed: Any
    archetype_cues: tuple[Any, ...]
    parse_quality: Any
    canonical_problem_graph: Any


def _enum_member_by_name_or_value(enum_cls: Any, candidates: list[str], default: Any = None) -> Any:
    if enum_cls is None:
        return default
    lowered = {c.lower() for c in candidates}
    for member in enum_cls:
        if member.name.lower() in lowered or str(member.value).lower() in lowered:
            return member
    return default


def _coerce_enum(enum_cls: Any, value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if enum_cls is None:
        return default
    if isinstance(value, enum_cls):
        return value
    normalized = str(value).strip().lower()
    for member in enum_cls:
        if member.name.lower() == normalized or str(member.value).lower() == normalized:
            return member
    return default


class ProblemParser:
    """Canonical parsing entrypoint for imo3_hybrid."""

    def __init__(self) -> None:
        self._normalizer = LatexNormalizer()

        self._let_pattern = re.compile(
            r"\blet\s+(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)\b",
            re.IGNORECASE,
        )
        self._find_all_pattern = re.compile(
            r"\bfind all\s+(?:positive integers|nonnegative integers|integers|reals|rationals|natural numbers)?\s*(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)",
            re.IGNORECASE,
        )
        self._forall_pattern = re.compile(
            r"\bfor all\s+(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)",
            re.IGNORECASE,
        )
        self._exists_pattern = re.compile(
            r"\bthere exists\s+(?:an?\s+)?(?:integer|real|rational|prime|natural number|natural_number)?\s*(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)",
            re.IGNORECASE,
        )

        self._objective_patterns: list[tuple[Any, re.Pattern[str]]] = [
            (
                _enum_member_by_name_or_value(
                    getattr(S, "QuantifierType", None),
                    ["find_all", "all_solutions", "solve", "find"],
                ),
                re.compile(
                    r"\b(find|determine|compute|evaluate)\b\s+(?P<target>.+?)(?:\.|;|,|$)",
                    re.IGNORECASE,
                ),
            ),
            (
                _enum_member_by_name_or_value(
                    getattr(S, "QuantifierType", None),
                    ["count", "how_many"],
                ),
                re.compile(
                    r"\bhow many\s+(?P<target>.+?)(?:\?|\.|;|,|$)",
                    re.IGNORECASE,
                ),
            ),
            (
                _enum_member_by_name_or_value(
                    getattr(S, "QuantifierType", None),
                    ["construct", "existence_construct"],
                ),
                re.compile(
                    r"\bconstruct\s+(?P<target>.+?)(?:\.|;|,|$)",
                    re.IGNORECASE,
                ),
            ),
            (
                _enum_member_by_name_or_value(
                    getattr(S, "QuantifierType", None),
                    ["maximize", "maximise", "maximum"],
                ),
                re.compile(
                    r"\bmaximize\s+(?P<target>.+?)(?:\.|;|,|$)",
                    re.IGNORECASE,
                ),
            ),
            (
                _enum_member_by_name_or_value(
                    getattr(S, "QuantifierType", None),
                    ["minimize", "minimise", "minimum"],
                ),
                re.compile(
                    r"\bminimize\s+(?P<target>.+?)(?:\.|;|,|$)",
                    re.IGNORECASE,
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_sync(
        self,
        text: str,
        *,
        problem_id: str | None = None,
        model_parser: Callable[[str], Mapping[str, Any] | ModelParseHints] | None = None,
        model_timeout_sec: float | None = None,
        hidden_constraint_inference: Callable[
            [str, list[Any], SymbolTable, str],
            Iterable[Any],
        ]
        | None = None,
        problem_graph_builder: Callable[
            [list[Any], SymbolTable, Any | None],
            Any,
        ]
        | None = None,
    ) -> Any:
        return self.parse_with_fallback(
            text,
            problem_id=problem_id,
            model_parser=model_parser,
            model_timeout_sec=model_timeout_sec,
            hidden_constraint_inference=hidden_constraint_inference,
            problem_graph_builder=problem_graph_builder,
        )

    async def parse_async(
        self,
        text: str,
        *,
        problem_id: str | None = None,
        model_parser: Callable[[str], Mapping[str, Any] | ModelParseHints] | None = None,
        model_timeout_sec: float | None = None,
        hidden_constraint_inference: Callable[
            [str, list[Any], SymbolTable, str],
            Iterable[Any],
        ]
        | None = None,
        problem_graph_builder: Callable[
            [list[Any], SymbolTable, Any | None],
            Any,
        ]
        | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self.parse_sync,
            text,
            problem_id=problem_id,
            model_parser=model_parser,
            model_timeout_sec=model_timeout_sec,
            hidden_constraint_inference=hidden_constraint_inference,
            problem_graph_builder=problem_graph_builder,
        )

    def parse_text_to_problem(self, text: str, *, problem_id: str | None = None) -> Any:
        return self.parse_sync(text, problem_id=problem_id)

    def parse_with_fallback(
        self,
        text: str,
        *,
        problem_id: str | None = None,
        model_parser: Callable[[str], Mapping[str, Any] | ModelParseHints] | None = None,
        model_timeout_sec: float | None = None,
        hidden_constraint_inference: Callable[
            [str, list[Any], SymbolTable, str],
            Iterable[Any],
        ]
        | None = None,
        problem_graph_builder: Callable[
            [list[Any], SymbolTable, Any | None],
            Any,
        ]
        | None = None,
    ) -> Any:
        raw_text = self._validate_input(text)
        pid = problem_id or self._derive_problem_id(raw_text)

        fallback_reason = DETERMINISTIC_ONLY_FALLBACK
        model_hints: ModelParseHints | None = None

        if model_parser is not None:
            try:
                model_hints = self._run_model_parser(raw_text, model_parser, model_timeout_sec)
                fallback_reason = "model_parse_succeeded"
            except FuturesTimeoutError:
                fallback_reason = MODEL_TIMEOUT_FALLBACK
            except (TypeError, ValueError):
                fallback_reason = MODEL_INVALID_OUTPUT_FALLBACK
            except Exception:
                fallback_reason = MODEL_EXCEPTION_FALLBACK

        ir = self._build_canonical_ir(
            raw_text,
            problem_id=pid,
            model_hints=model_hints,
            fallback_reason=fallback_reason,
            hidden_constraint_inference=hidden_constraint_inference,
            problem_graph_builder=problem_graph_builder,
        )
        return self._compile_parsed_problem(problem_id=pid, ir=ir)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _build_canonical_ir(
        self,
        raw_text: str,
        *,
        problem_id: str,
        model_hints: ModelParseHints | None,
        fallback_reason: str,
        hidden_constraint_inference: Callable[[str, list[Any], SymbolTable, str], Iterable[Any]] | None,
        problem_graph_builder: Callable[[list[Any], SymbolTable, Any | None], Any] | None,
    ) -> CanonicalParseIR:
        normalized_text, normalization_report = normalize_math_with_report(raw_text)
        if model_hints is not None and model_hints.normalized_text:
            normalized_text = model_hints.normalized_text.strip()
        latex_spans = tuple(extract_latex_spans(raw_text))

        raw_units = self.extract_raw_problem_units(normalized_text)
        symbol_table = SymbolTable(problem_namespace=problem_id)
        scope_id = self._get_root_scope_id(symbol_table)

        self._register_symbols(
            normalized_text,
            raw_units,
            symbol_table=symbol_table,
            scope_id=scope_id,
            model_hints=model_hints,
        )

        base_extraction = self._extract_constraints_wrapper(
            normalized_text,
            symbol_table=symbol_table,
            scope_id=scope_id,
        )
        constraints = list(getattr(base_extraction, "constraints", []) or [])

        if model_hints is not None and model_hints.extra_constraint_texts:
            for extra_text in model_hints.extra_constraint_texts:
                extra_result = self._extract_constraints_wrapper(
                    extra_text,
                    symbol_table=symbol_table,
                    scope_id=scope_id,
                )
                constraints.extend(list(getattr(extra_result, "constraints", []) or []))

        semantic_gaps: list[str] = []
        if hidden_constraint_inference is not None:
            try:
                inferred = list(
                    hidden_constraint_inference(
                        normalized_text,
                        constraints,
                        symbol_table,
                        scope_id,
                    )
                )
                constraints.extend(inferred)
            except Exception as exc:
                semantic_gaps.append(f"hidden_constraint_inference_failed:{type(exc).__name__}")
        else:
            semantic_gaps.append("hidden_constraint_inference_absent")

        objective = self._resolve_objective(
            normalized_text,
            extraction=base_extraction,
            raw_units=raw_units,
            model_hints=model_hints,
        )
        answer_type = self._infer_answer_type(
            normalized_text,
            objective=objective,
            model_hints=model_hints,
            constraints=constraints,
        )
        domain_seed = self._infer_problem_domain(
            normalized_text,
            constraints,
            model_hints,
        )
        archetype_cues = self._infer_archetype_cues(
            normalized_text,
            constraints,
            model_hints,
        )

        if problem_graph_builder is not None:
            try:
                canonical_problem_graph = problem_graph_builder(
                    constraints,
                    symbol_table,
                    objective,
                )
            except Exception as exc:
                semantic_gaps.append(f"problem_graph_builder_failed:{type(exc).__name__}")
                canonical_problem_graph = self._build_lightweight_problem_graph(
                    constraints,
                    symbol_table,
                    objective,
                )
        else:
            semantic_gaps.append("canonical_problem_graph_builder_absent")
            canonical_problem_graph = self._build_lightweight_problem_graph(
                constraints,
                symbol_table,
                objective,
            )

        parse_quality = self._build_parse_quality_report(
            normalized_text=normalized_text,
            normalization_report=normalization_report,
            raw_units=raw_units,
            symbol_table=symbol_table,
            constraints=constraints,
            objective=objective,
            answer_type=answer_type,
            fallback_used=(model_hints is None),
            fallback_reason=fallback_reason,
            semantic_gaps=semantic_gaps,
        )

        return CanonicalParseIR(
            raw_text=raw_text,
            normalized_text=normalized_text,
            latex_spans=latex_spans,
            raw_units=raw_units,
            symbol_table=symbol_table,
            constraints=tuple(self._dedupe_constraints(constraints)),
            objective=objective,
            answer_type=answer_type,
            problem_domain_seed=domain_seed,
            archetype_cues=archetype_cues,
            parse_quality=parse_quality,
            canonical_problem_graph=canonical_problem_graph,
        )

    def extract_raw_problem_units(self, normalized_text: str) -> RawProblemUnits:
        quantifier_enum = getattr(S, "QuantifierType", None)

        variable_candidates: list[str] = []
        domain_mentions: list[str] = []
        quantifiers: list[Any] = []
        objective_text: str | None = None
        target_mode: Any | None = None

        for pattern in (
            self._let_pattern,
            self._find_all_pattern,
            self._forall_pattern,
            self._exists_pattern,
        ):
            for match in pattern.finditer(normalized_text):
                vars_group = match.groupdict().get("vars", "")
                variable_candidates.extend(self._split_vars(vars_group))

        quantifier_specs = [
            (_enum_member_by_name_or_value(quantifier_enum, ["forall", "for_all"]), self._forall_pattern),
            (_enum_member_by_name_or_value(quantifier_enum, ["exists", "there_exists"]), self._exists_pattern),
            (_enum_member_by_name_or_value(quantifier_enum, ["find_all", "all_solutions", "solve"]), self._find_all_pattern),
        ]

        for qtype, pattern in quantifier_specs:
            if qtype is None:
                continue
            for match in pattern.finditer(normalized_text):
                quantifiers.append(
                    S.QuantifiedVariable(
                        quantifier_type=qtype,
                        variables=self._split_vars(match.groupdict().get("vars", "")),
                        raw_text=match.group(0),
                        span_start=match.start(),
                        span_end=match.end(),
                    )
                )

        lowered = normalized_text.lower()
        for raw_alias, canonical in PARSING.DOMAIN_CANONICAL_ALIASES.items():
            if raw_alias.replace("_", " ") in lowered or raw_alias in lowered:
                domain_mentions.append(canonical)

        for mode, pattern in self._objective_patterns:
            if mode is None:
                continue
            match = pattern.search(normalized_text)
            if match is not None:
                objective_text = match.group("target").strip()
                target_mode = mode
                break

        sentence_count = max(
            1,
            len(re.findall(PARSING.SENTENCE_SPLIT_PATTERN, normalized_text)) + 1,
        )

        return RawProblemUnits(
            variable_candidates=tuple(dict.fromkeys(v for v in variable_candidates if v)),
            domain_mentions=tuple(dict.fromkeys(domain_mentions)),
            quantifiers=tuple(
                sorted(
                    quantifiers,
                    key=lambda q: ((getattr(q, "span_start", -1) or -1), (getattr(q, "span_end", -1) or -1)),
                )
            ),
            objective_text=objective_text,
            target_mode=target_mode,
            sentence_count=sentence_count,
        )

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile_parsed_problem(self, *, problem_id: str, ir: CanonicalParseIR) -> Any:
        variables = self._compile_variable_decls(ir.symbol_table)
        if not ir.constraints:
            raise ParseFailure(
                "Parser produced zero constraints; refusing to emit an untyped ParsedProblem"
            )

        return S.ParsedProblem(
            problem_id=problem_id,
            raw_text=ir.raw_text,
            normalized_text=ir.normalized_text,
            latex_spans=list(ir.latex_spans),
            variables=variables,
            constraints=list(ir.constraints),
            objective=ir.objective,
            answer_type=ir.answer_type,
            problem_domain_seed=ir.problem_domain_seed,
            archetype_cues=list(ir.archetype_cues),
            parse_quality=ir.parse_quality,
            canonical_problem_graph=ir.canonical_problem_graph,
        )

    # ------------------------------------------------------------------
    # Model / fallback unification
    # ------------------------------------------------------------------

    def _run_model_parser(
        self,
        raw_text: str,
        model_parser: Callable[[str], Mapping[str, Any] | ModelParseHints],
        timeout_sec: float | None,
    ) -> ModelParseHints:
        if timeout_sec is None or timeout_sec <= 0:
            return self._coerce_model_hints(model_parser(raw_text))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(model_parser, raw_text)
            try:
                result = future.result(timeout=timeout_sec)
            finally:
                future.cancel()
        return self._coerce_model_hints(result)

    def _coerce_model_hints(
        self,
        payload: Mapping[str, Any] | ModelParseHints,
    ) -> ModelParseHints:
        if isinstance(payload, ModelParseHints):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError("model_parser must return a Mapping or ModelParseHints")

        objective_payload = payload.get("objective")
        objective = None
        if isinstance(objective_payload, S.ObjectiveSpec):
            objective = objective_payload
        elif isinstance(objective_payload, Mapping):
            try:
                objective = S.ObjectiveSpec.model_validate(objective_payload)
            except Exception:
                objective = None

        answer_type = _coerce_enum(
            getattr(S, "AnswerType", None),
            payload.get("answer_type"),
            default=_enum_member_by_name_or_value(getattr(S, "AnswerType", None), ["unknown"]),
        )
        domain_seed = _coerce_enum(
            getattr(S, "ProblemDomain", None),
            payload.get("domain_seed"),
            default=_enum_member_by_name_or_value(getattr(S, "ProblemDomain", None), ["unknown"]),
        )

        archetype_values: list[Any] = []
        for item in payload.get("archetype_cues", ()):
            coerced = _coerce_enum(getattr(S, "Archetype", None), item)
            if coerced is not None:
                archetype_values.append(coerced)

        return ModelParseHints(
            normalized_text=payload.get("normalized_text"),
            variable_candidates=tuple(payload.get("variable_candidates", ())),
            extra_constraint_texts=tuple(payload.get("extra_constraint_texts", ())),
            answer_type=answer_type,
            domain_seed=domain_seed,
            archetype_cues=tuple(dict.fromkeys(archetype_values)),
            objective=objective,
            metadata=dict(payload.get("metadata", {})),
        )

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _extract_constraints_wrapper(
        self,
        text: str,
        *,
        symbol_table: SymbolTable,
        scope_id: str,
    ) -> Any:
        sig = inspect.signature(extract_constraints)
        kwargs: dict[str, Any] = {}

        if "symbol_table" in sig.parameters:
            kwargs["symbol_table"] = symbol_table
        if "scope_id" in sig.parameters:
            kwargs["scope_id"] = scope_id

        return extract_constraints(text, **kwargs)

    def _register_symbols(
        self,
        normalized_text: str,
        raw_units: RawProblemUnits,
        *,
        symbol_table: SymbolTable,
        scope_id: str,
        model_hints: ModelParseHints | None,
    ) -> None:
        candidate_vars = list(raw_units.variable_candidates)
        if model_hints is not None:
            candidate_vars.extend(model_hints.variable_candidates)

        register = getattr(symbol_table, "register_symbol", None)
        if callable(register):
            for var_name in dict.fromkeys(v for v in candidate_vars if v):
                try:
                    register(
                        var_name,
                        scope_id=scope_id,
                        role=_enum_member_by_name_or_value(getattr(S, "SymbolRole", None), ["variable"]),
                        domain=self._guess_domain_for_variable(var_name, raw_units.domain_mentions),
                    )
                except TypeError:
                    register(var_name)

            token_pattern = re.compile(PARSING.SIMPLE_SYMBOL_PATTERN)
            for token in token_pattern.findall(normalized_text):
                try:
                    register(
                        token,
                        scope_id=scope_id,
                        role=_enum_member_by_name_or_value(getattr(S, "SymbolRole", None), ["variable"]),
                    )
                except TypeError:
                    register(token)

    def _resolve_objective(
        self,
        normalized_text: str,
        *,
        extraction: Any,
        raw_units: RawProblemUnits,
        model_hints: ModelParseHints | None,
    ) -> Any | None:
        if model_hints is not None and model_hints.objective is not None:
            return model_hints.objective

        for node in list(getattr(extraction, "objective_constraints", []) or []):
            raw_objective = getattr(node, "metadata", {}).get("objective")
            if isinstance(raw_objective, S.ObjectiveSpec):
                return raw_objective
            if isinstance(raw_objective, Mapping):
                try:
                    return S.ObjectiveSpec.model_validate(raw_objective)
                except Exception:
                    pass

        if raw_units.objective_text is not None and raw_units.target_mode is not None:
            return S.ObjectiveSpec(
                objective_mode=raw_units.target_mode,
                target_text=raw_units.objective_text,
                normalized_target_text=normalize_math_text(raw_units.objective_text),
                subject_variables=[],
            )

        _ = normalized_text
        return None

    def _infer_answer_type(
        self,
        normalized_text: str,
        *,
        objective: Any | None,
        model_hints: ModelParseHints | None,
        constraints: list[Any],
    ) -> Any:
        answer_enum = getattr(S, "AnswerType", None)
        unknown_answer = _enum_member_by_name_or_value(answer_enum, ["unknown"])

        if model_hints is not None and model_hints.answer_type is not None:
            return model_hints.answer_type

        lowered = normalized_text.lower()
        domains = {
            assumption
            for c in constraints
            for assumption in list(getattr(c, "domain_assumptions", []) or [])
        }

        objective_mode = getattr(objective, "objective_mode", None)
        if objective_mode is not None:
            count_mode = _enum_member_by_name_or_value(getattr(S, "QuantifierType", None), ["count", "how_many"])
            construct_mode = _enum_member_by_name_or_value(getattr(S, "QuantifierType", None), ["construct"])
            find_all_mode = _enum_member_by_name_or_value(getattr(S, "QuantifierType", None), ["find_all", "all_solutions"])

            if objective_mode == count_mode:
                return _enum_member_by_name_or_value(answer_enum, ["nonnegative_integer", "non_negative_integer", "integer"], unknown_answer)
            if objective_mode == construct_mode:
                return _enum_member_by_name_or_value(answer_enum, ["construction"], unknown_answer)
            if objective_mode == find_all_mode:
                return _enum_member_by_name_or_value(answer_enum, ["set"], unknown_answer)

        if any(
            domain in domains
            for domain in (
                "integers",
                "positive_integers",
                "nonnegative_integers",
                "natural_numbers",
                "primes",
            )
        ):
            if "how many" in lowered or "number of" in lowered:
                return _enum_member_by_name_or_value(answer_enum, ["nonnegative_integer", "integer"], unknown_answer)
            return _enum_member_by_name_or_value(answer_enum, ["integer"], unknown_answer)
        if "rational" in lowered:
            return _enum_member_by_name_or_value(answer_enum, ["rational"], unknown_answer)
        if "real" in lowered:
            return _enum_member_by_name_or_value(answer_enum, ["real"], unknown_answer)
        if "set of" in lowered or "find all" in lowered:
            return _enum_member_by_name_or_value(answer_enum, ["set"], unknown_answer)
        return unknown_answer

    def _infer_problem_domain(
        self,
        normalized_text: str,
        constraints: list[Any],
        model_hints: ModelParseHints | None,
    ) -> Any:
        domain_enum = getattr(S, "ProblemDomain", None)
        unknown_domain = _enum_member_by_name_or_value(domain_enum, ["unknown"])

        if model_hints is not None and model_hints.domain_seed is not None:
            return model_hints.domain_seed

        lowered = normalized_text.lower()
        if any(
            k in lowered
            for k in (
                "triangle",
                "angle",
                "circle",
                "perpendicular",
                "parallel",
                "segment(",
            )
        ):
            return _enum_member_by_name_or_value(domain_enum, ["geometry"], unknown_domain)
        if any(
            str(getattr(c, "relation_type", "")).lower() in {"divisibility", "congruence", "integrality"}
            or str(getattr(getattr(c, "relation_type", None), "value", "")).lower() in {"divisibility", "congruence", "integrality"}
            for c in constraints
        ):
            return _enum_member_by_name_or_value(domain_enum, ["number_theory", "number theory"], unknown_domain)
        if any(
            k in lowered
            for k in ("how many", "bijection", "permutation", "graph", "arrangement")
        ):
            return _enum_member_by_name_or_value(domain_enum, ["combinatorics"], unknown_domain)
        if any(
            k in lowered
            for k in ("function", "polynomial", "equation", "inequality", "sqrt", "binom(")
        ):
            return _enum_member_by_name_or_value(domain_enum, ["algebra"], unknown_domain)
        return unknown_domain

    def _infer_archetype_cues(
        self,
        normalized_text: str,
        constraints: list[Any],
        model_hints: ModelParseHints | None,
    ) -> tuple[Any, ...]:
        archetype_enum = getattr(S, "Archetype", None)
        unknown_arch = _enum_member_by_name_or_value(archetype_enum, ["unknown"])

        if model_hints is not None and model_hints.archetype_cues:
            return tuple(dict.fromkeys(model_hints.archetype_cues))

        lowered = normalized_text.lower()
        cues: list[Any] = []

        cue_specs = [
            (("invariant", "remains unchanged", "preserved"), ["invariant"]),
            (("symmetric", "symmetry", "without loss of generality"), ["symmetry"]),
            (("contradiction", "impossible", "assume not"), ["contradiction"]),
            (("construct", "show that there exists"), ["construction"]),
            (("induction", "for all n"), ["induction"]),
            (("bijection", "one-to-one", "onto"), ["bijection"]),
            (("case 1", "case 2", "consider cases"), ["case_split", "cases"]),
            (("pigeonhole", "boxes", "drawers"), ["pigeonhole"]),
            (("generating function", "coefficient"), ["generating_function"]),
            (("smallest", "largest", "minimum", "maximum", "maximize", "minimize"), ["extremal"]),
        ]

        for keywords, names in cue_specs:
            if any(k in lowered for k in keywords):
                member = _enum_member_by_name_or_value(archetype_enum, names)
                if member is not None:
                    cues.append(member)

        if any(
            str(getattr(getattr(c, "relation_type", None), "value", getattr(c, "relation_type", ""))).lower() == "congruence"
            for c in constraints
        ) or "mod" in lowered:
            member = _enum_member_by_name_or_value(archetype_enum, ["modular", "mod"])
            if member is not None:
                cues.append(member)

        if not cues and unknown_arch is not None:
            cues.append(unknown_arch)

        return tuple(dict.fromkeys(cues))

    def _build_lightweight_problem_graph(
        self,
        constraints: list[Any],
        symbol_table: SymbolTable,
        objective: Any | None,
    ) -> Any:
        node_ids: list[str] = []
        edges: list[Any] = []

        symbol_entries = self._get_symbol_entries(symbol_table)
        symbol_ids = sorted(symbol_entries.keys())
        node_ids.extend(symbol_ids)

        for constraint in constraints:
            constraint_id = getattr(constraint, "constraint_id", None)
            if constraint_id is None:
                continue
            node_ids.append(constraint_id)

            lhs = getattr(constraint, "lhs", None)
            rhs = getattr(constraint, "rhs", None)

            for symbol_id in list(getattr(lhs, "symbol_ids", []) or []):
                edges.append(
                    S.ConstraintGraphEdge(
                        source_node_id=symbol_id,
                        target_node_id=constraint_id,
                        relation_label="mentions_lhs",
                    )
                )
            for symbol_id in list(getattr(rhs, "symbol_ids", []) or []):
                edges.append(
                    S.ConstraintGraphEdge(
                        source_node_id=symbol_id,
                        target_node_id=constraint_id,
                        relation_label="mentions_rhs",
                    )
                )

        if objective is not None:
            node_ids.append("objective::target")

        return S.CanonicalProblemGraph(
            node_ids=list(dict.fromkeys(node_ids)),
            edges=edges,
            metadata={
                "builder": "lightweight_parser_graph",
                "constraint_count": len(constraints),
                "symbol_count": len(symbol_ids),
                "objective_present": objective is not None,
            },
        )

    def _build_parse_quality_report(
        self,
        *,
        normalized_text: str,
        normalization_report: Any,
        raw_units: RawProblemUnits,
        symbol_table: SymbolTable,
        constraints: list[Any],
        objective: Any | None,
        answer_type: Any,
        fallback_used: bool,
        fallback_reason: str,
        semantic_gaps: list[str],
    ) -> Any:
        answer_enum = getattr(S, "AnswerType", None)
        unknown_answer = _enum_member_by_name_or_value(answer_enum, ["unknown"])

        missing_fields: list[str] = []
        confidence_by_field: dict[str, float] = {}
        penalties: dict[str, float] = {}

        variable_count = len(self._get_symbol_entries(symbol_table))
        constraint_count = len(constraints)
        normalized_change_count = len(getattr(normalization_report, "changes", []))

        confidence_by_field["normalized_text"] = 0.95 if normalized_text else 0.0
        confidence_by_field["variables"] = min(1.0, 0.25 + 0.1 * variable_count)
        confidence_by_field["constraints"] = min(1.0, 0.2 + 0.12 * constraint_count)
        confidence_by_field["objective"] = 0.9 if objective is not None else 0.35
        confidence_by_field["answer_type"] = 0.9 if answer_type != unknown_answer else 0.3
        confidence_by_field["quantifiers"] = 0.85 if raw_units.quantifiers else 0.4

        if variable_count == 0:
            missing_fields.append("variables")
            penalties["routing_uncertainty"] = 0.2
        if constraint_count == 0:
            missing_fields.append("constraints")
            penalties["state_init_uncertainty"] = 0.35
        if objective is None:
            missing_fields.append("objective")
            penalties["aggregation_uncertainty"] = 0.1
        if answer_type == unknown_answer:
            missing_fields.append("answer_type")
            penalties["final_selector_uncertainty"] = 0.15

        if fallback_used:
            semantic_gaps = [*semantic_gaps, f"fallback_reason:{fallback_reason}"]
            penalties["retrieval_confidence_penalty"] = 0.1
            penalties["routing_confidence_penalty"] = 0.1

        if normalized_change_count > 8:
            semantic_gaps.append("heavy_notation_normalization")
        if raw_units.sentence_count > 8:
            semantic_gaps.append("long_problem_statement")

        return S.ParseQualityReport(
            missing_fields=missing_fields,
            confidence_by_field=confidence_by_field,
            fallback_used=fallback_used,
            semantic_gaps=semantic_gaps,
            suggested_downstream_penalties=penalties,
        )

    def _compile_variable_decls(self, symbol_table: SymbolTable) -> list[Any]:
        symbol_entries = self._get_symbol_entries(symbol_table)
        symbol_role_enum = getattr(S, "SymbolRole", None)
        variable_role = _enum_member_by_name_or_value(symbol_role_enum, ["variable"])

        compiled: list[Any] = []
        seen: set[str] = set()

        for symbol_id in sorted(symbol_entries):
            symbol = symbol_entries[symbol_id]
            if variable_role is not None and getattr(symbol, "role", None) != variable_role:
                continue
            if symbol_id in seen:
                continue
            seen.add(symbol_id)
            compiled.append(
                S.VariableDecl(
                    symbol=symbol,
                    declared_domain=getattr(symbol, "domain", None),
                    source_text=getattr(symbol, "display_name", None),
                )
            )
        return compiled

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _validate_input(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("parse input must be a string")
        stripped = text.strip()
        if not stripped:
            raise ValueError("parse input must not be empty")
        if len(stripped) > PARSING.MAX_INPUT_CHARS:
            raise ValueError(
                f"parse input exceeds MAX_INPUT_CHARS={PARSING.MAX_INPUT_CHARS}"
            )
        return stripped

    def _derive_problem_id(self, raw_text: str) -> str:
        digest = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:16]
        return f"problem_{digest}"

    def _split_vars(self, text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]

    def _guess_domain_for_variable(
        self,
        var_name: str,
        domain_mentions: Iterable[str],
    ) -> str | None:
        _ = var_name
        ordered = list(domain_mentions)
        return ordered[0] if ordered else None

    def _dedupe_constraints(self, constraints: list[Any]) -> list[Any]:
        seen: dict[tuple[Any, ...], Any] = {}
        for c in constraints:
            lhs = getattr(c, "lhs", None)
            rhs = getattr(c, "rhs", None)
            key = (
                str(getattr(getattr(c, "relation_type", None), "value", getattr(c, "relation_type", None))),
                getattr(c, "operator", None),
                getattr(lhs, "normalized_text", None),
                getattr(rhs, "normalized_text", None),
                tuple(getattr(c, "domain_assumptions", []) or []),
                tuple(
                    (
                        str(getattr(getattr(q, "quantifier_type", None), "value", getattr(q, "quantifier_type", None))),
                        tuple(getattr(q, "variables", []) or []),
                        getattr(q, "raw_text", None),
                    )
                    for q in (getattr(c, "quantified_variables", []) or [])
                ),
            )
            if key not in seen:
                seen[key] = c
        return list(seen.values())

    def _get_root_scope_id(self, symbol_table: SymbolTable) -> str:
        scope_id = getattr(symbol_table, "root_scope_id", None)
        if isinstance(scope_id, str) and scope_id:
            return scope_id
        return "root"

    def _get_symbol_entries(self, symbol_table: SymbolTable) -> dict[str, Any]:
        entries = getattr(symbol_table, "_symbols_by_id", None)
        if isinstance(entries, dict):
            return entries
        exported = getattr(symbol_table, "export", None)
        if callable(exported):
            try:
                payload = exported()
                if isinstance(payload, Mapping) and isinstance(payload.get("symbols_by_id"), Mapping):
                    return dict(payload["symbols_by_id"])
            except Exception:
                pass
        return {}


def parse_sync(
    text: str,
    *,
    problem_id: str | None = None,
    model_parser: Callable[[str], Mapping[str, Any] | ModelParseHints] | None = None,
    model_timeout_sec: float | None = None,
    hidden_constraint_inference: Callable[[str, list[Any], SymbolTable, str], Iterable[Any]] | None = None,
    problem_graph_builder: Callable[[list[Any], SymbolTable, Any | None], Any] | None = None,
) -> Any:
    return ProblemParser().parse_sync(
        text,
        problem_id=problem_id,
        model_parser=model_parser,
        model_timeout_sec=model_timeout_sec,
        hidden_constraint_inference=hidden_constraint_inference,
        problem_graph_builder=problem_graph_builder,
    )


async def parse_async(
    text: str,
    *,
    problem_id: str | None = None,
    model_parser: Callable[[str], Mapping[str, Any] | ModelParseHints] | None = None,
    model_timeout_sec: float | None = None,
    hidden_constraint_inference: Callable[[str, list[Any], SymbolTable, str], Iterable[Any]] | None = None,
    problem_graph_builder: Callable[[list[Any], SymbolTable, Any | None], Any] | None = None,
) -> Any:
    return await ProblemParser().parse_async(
        text,
        problem_id=problem_id,
        model_parser=model_parser,
        model_timeout_sec=model_timeout_sec,
        hidden_constraint_inference=hidden_constraint_inference,
        problem_graph_builder=problem_graph_builder,
    )


def parse_text_to_problem(text: str, *, problem_id: str | None = None) -> Any:
    return ProblemParser().parse_text_to_problem(text, problem_id=problem_id)


def parse_with_fallback(
    text: str,
    *,
    problem_id: str | None = None,
    model_parser: Callable[[str], Mapping[str, Any] | ModelParseHints] | None = None,
    model_timeout_sec: float | None = None,
    hidden_constraint_inference: Callable[[str, list[Any], SymbolTable, str], Iterable[Any]] | None = None,
    problem_graph_builder: Callable[[list[Any], SymbolTable, Any | None], Any] | None = None,
) -> Any:
    return ProblemParser().parse_with_fallback(
        text,
        problem_id=problem_id,
        model_parser=model_parser,
        model_timeout_sec=model_timeout_sec,
        hidden_constraint_inference=hidden_constraint_inference,
        problem_graph_builder=problem_graph_builder,
    )


__all__ = [
    "MODEL_TIMEOUT_FALLBACK",
    "MODEL_EXCEPTION_FALLBACK",
    "MODEL_INVALID_OUTPUT_FALLBACK",
    "DETERMINISTIC_ONLY_FALLBACK",
    "ModelParseHints",
    "RawProblemUnits",
    "ProblemParser",
    "ParseFailure",
    "parse_sync",
    "parse_async",
    "parse_text_to_problem",
    "parse_with_fallback",
]

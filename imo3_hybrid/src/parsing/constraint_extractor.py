from __future__ import annotations

import hashlib
import re

from src.common.constants import PARSING
from src.common.schemas import (
    ConstraintExtractionReport,
    ConstraintExtractionResult,
    ConstraintNode,
    ConstraintOrigin,
    ConstraintProvenance,
    ConstraintStrength,
    MathOperand,
    ObjectiveSpec,
    QuantifiedVariable,
    QuantifierType,
    RelationType,
)
from src.parsing.latex_normalizer import normalize_math_text
from src.parsing.symbol_table import SymbolResolution, SymbolTable


def _stable_id(prefix: str, payload: str) -> str:
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def _split_vars(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


class ConstraintExtractor:
    def __init__(self) -> None:
        self._domain_aliases = PARSING.DOMAIN_CANONICAL_ALIASES
        self._quantifier_patterns = [
            (QuantifierType.FORALL, re.compile(r"\bfor all\s+(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)", re.IGNORECASE)),
            (QuantifierType.EXISTS, re.compile(r"\bthere exists\s+(?:an?\s+)?(?:integer|real|rational|prime|natural number|natural_number)?\s*(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)", re.IGNORECASE)),
            (QuantifierType.FIND_ALL, re.compile(r"\bfind all\s+(?:integers|reals|rationals|positive integers|natural numbers)?\s*(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)", re.IGNORECASE)),
            (QuantifierType.COUNT, re.compile(r"\bhow many\b", re.IGNORECASE)),
            (QuantifierType.CONSTRUCT, re.compile(r"\bconstruct\b", re.IGNORECASE)),
            (QuantifierType.MAXIMIZE, re.compile(r"\bmaximize\b", re.IGNORECASE)),
            (QuantifierType.MINIMIZE, re.compile(r"\bminimize\b", re.IGNORECASE)),
        ]
        self._domain_patterns = [
            re.compile(
                r"\b(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)\s+(?:are|is)\s+(?P<domain>positive_integers|non_negative_integers|nonnegative_integers|integers|natural_numbers|real_numbers|rational_numbers|primes)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)\s+in\s+(?P<domain>integers|real_numbers|rational_numbers|natural_numbers|primes)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bfor all\s+(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)\s+in\s+(?P<domain>integers|real_numbers|rational_numbers|natural_numbers|primes)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bthere exists\s+(?:an?\s+)?(?P<domain>integer|real|rational|prime|natural_number)\s+(?P<vars>[A-Za-z](?:\s*,\s*[A-Za-z])*)\b",
                re.IGNORECASE,
            ),
        ]
        self._positivity_phrases = [
            ("positive_integers", ">", "0"),
            ("positive_integer", ">", "0"),
            ("nonnegative_integers", ">=", "0"),
            ("non_negative_integers", ">=", "0"),
            ("nonnegative_integer", ">=", "0"),
            ("non_negative_integer", ">=", "0"),
        ]
        self._congruence_pattern = re.compile(r"(?P<lhs>.+?)\s*(?:≡|\\equiv)\s*(?P<rhs>.+?)\s*\(mod\s+(?P<modulus>.+?)\)", re.IGNORECASE)
        self._divides_pattern = re.compile(r"(?P<divisor>[A-Za-z0-9_()+\-*/^ ]+?)\s+divides\s+(?P<dividend>[A-Za-z0-9_()+\-*/^ ]+)", re.IGNORECASE)
        self._bar_divides_pattern = re.compile(r"(?P<divisor>[A-Za-z0-9_()+\-*/^ ]+?)\s*\|\s*(?P<dividend>[A-Za-z0-9_()+\-*/^ ]+)")
        self._is_divisible_by_pattern = re.compile(r"(?P<dividend>[A-Za-z0-9_()+\-*/^ ]+?)\s+is\s+divisible\s+by\s+(?P<divisor>[A-Za-z0-9_()+\-*/^ ]+)", re.IGNORECASE)
        self._relation_pattern = re.compile(r"(?P<lhs>[^<>=!\n;]+?)\s*(?P<op><=|>=|!=|=|<|>)\s*(?P<rhs>[^<>=!\n;]+)")
        self._find_target_pattern = re.compile(r"\b(find|determine|compute|evaluate)\b\s+(?P<target>.+?)(?:\.|;|,|$)", re.IGNORECASE)
        self._how_many_pattern = re.compile(r"\bhow many\s+(?P<target>.+?)(?:\?|\.|;|,|$)", re.IGNORECASE)
        self._maxmin_pattern = re.compile(r"\b(?P<mode>maximize|minimize)\s+(?P<target>.+?)(?:\.|;|,|$)", re.IGNORECASE)

    def extract(
        self,
        text: str,
        *,
        symbol_table: SymbolTable,
        scope_id: str | None = None,
    ) -> ConstraintExtractionResult:
        scope = scope_id or symbol_table.root_scope_id
        normalized = normalize_math_text(text or "").strip()

        constraints: list[ConstraintNode] = []
        objective_constraints: list[ConstraintNode] = []
        unresolved_operands: set[str] = set()

        quantifiers = self._extract_quantifiers(normalized)
        constraints.extend(self._extract_domain_constraints(normalized, symbol_table, scope, quantifiers, unresolved_operands))
        constraints.extend(self._extract_congruences(normalized, symbol_table, scope, quantifiers, unresolved_operands))
        constraints.extend(self._extract_divisibility(normalized, symbol_table, scope, quantifiers, unresolved_operands))
        constraints.extend(self._extract_relations(normalized, symbol_table, scope, quantifiers, unresolved_operands))
        objective_constraints.extend(self._extract_objectives(normalized, symbol_table, scope, quantifiers, unresolved_operands))

        constraints = self._dedupe(constraints)
        objective_constraints = self._dedupe(objective_constraints)

        report = ConstraintExtractionReport(
            normalized_text=normalized,
            stage_counts={
                "quantifiers": len(quantifiers),
                "constraints": len(constraints),
                "objective_constraints": len(objective_constraints),
            },
            warnings=[],
            unresolved_operands=sorted(unresolved_operands),
            quantifiers=quantifiers,
        )
        return ConstraintExtractionResult(
            constraints=constraints,
            objective_constraints=objective_constraints,
            report=report,
        )

    def _extract_quantifiers(self, text: str) -> list[QuantifiedVariable]:
        out: list[QuantifiedVariable] = []
        for qtype, pattern in self._quantifier_patterns:
            for match in pattern.finditer(text):
                vars_group = match.groupdict().get("vars", "") if match.groupdict() else ""
                out.append(
                    QuantifiedVariable(
                        quantifier_type=qtype,
                        variables=_split_vars(vars_group) if vars_group else [],
                        raw_text=match.group(0),
                        span_start=match.start(),
                        span_end=match.end(),
                    )
                )
        out.sort(key=lambda q: ((q.span_start or -1), (q.span_end or -1)))
        return out

    def _extract_domain_constraints(
        self,
        text: str,
        symbol_table: SymbolTable,
        scope_id: str,
        quantifiers: list[QuantifiedVariable],
        unresolved_operands: set[str],
    ) -> list[ConstraintNode]:
        out: list[ConstraintNode] = []
        for pattern in self._domain_patterns:
            for match in pattern.finditer(text):
                vars_list = _split_vars(match.group("vars"))
                raw_domain = match.group("domain").strip().lower().replace(" ", "_")
                domain = self._domain_aliases.get(raw_domain, raw_domain)
                qvars = self._active_quantifiers(match.start(), quantifiers)

                for var_name in vars_list:
                    lhs = self._make_operand(var_name, symbol_table, scope_id, unresolved_operands)
                    rhs = MathOperand(
                        raw_text=domain,
                        normalized_text=domain,
                        scope_id=scope_id,
                        symbol_ids=[],
                        canonical_mentions=[],
                        operand_kind="domain",
                    )
                    out.append(
                        self._make_constraint(
                            relation_type=RelationType.DOMAIN_MEMBERSHIP,
                            origin=ConstraintOrigin.EXPLICIT_DOMAIN,
                            strength=ConstraintStrength.HARD,
                            lhs=lhs,
                            rhs=rhs,
                            operator="in",
                            metadata={"domain": domain},
                            domain_assumptions=[domain],
                            quantified_variables=qvars,
                            explicit=True,
                            inferred_ready=True,
                            provenance=self._make_provenance(text, match.start(), match.end(), "domain_membership"),
                        )
                    )
                    if domain in {"integers", "positive_integers", "nonnegative_integers", "natural_numbers", "primes"}:
                        out.append(
                            self._make_constraint(
                                relation_type=RelationType.INTEGRALITY,
                                origin=ConstraintOrigin.EXPLICIT_DOMAIN,
                                strength=ConstraintStrength.HARD,
                                lhs=lhs,
                                rhs=MathOperand(
                                    raw_text="integers",
                                    normalized_text="integers",
                                    scope_id=scope_id,
                                    symbol_ids=[],
                                    canonical_mentions=[],
                                    operand_kind="domain",
                                ),
                                operator="in",
                                metadata={"integrality": True, "derived_from_domain": domain},
                                domain_assumptions=["integers"],
                                quantified_variables=qvars,
                                explicit=True,
                                inferred_ready=True,
                                provenance=self._make_provenance(text, match.start(), match.end(), "integrality"),
                            )
                        )

                for phrase, op, zero in self._positivity_phrases:
                    if domain == phrase:
                        for var_name in vars_list:
                            out.append(
                                self._make_constraint(
                                    relation_type=RelationType.POSITIVITY if op == ">" else RelationType.BOUNDEDNESS,
                                    origin=ConstraintOrigin.INFERRED_READY,
                                    strength=ConstraintStrength.HARD,
                                    lhs=self._make_operand(var_name, symbol_table, scope_id, unresolved_operands),
                                    rhs=self._make_operand(zero, symbol_table, scope_id, unresolved_operands, operand_kind="expression"),
                                    operator=op,
                                    metadata={"derived_from_domain": domain},
                                    domain_assumptions=[domain],
                                    quantified_variables=qvars,
                                    explicit=False,
                                    inferred_ready=True,
                                    provenance=self._make_provenance(text, match.start(), match.end(), "positivity_from_domain"),
                                )
                            )
        return out

    def _extract_congruences(
        self,
        text: str,
        symbol_table: SymbolTable,
        scope_id: str,
        quantifiers: list[QuantifiedVariable],
        unresolved_operands: set[str],
    ) -> list[ConstraintNode]:
        out: list[ConstraintNode] = []
        for match in self._congruence_pattern.finditer(text):
            out.append(
                self._make_constraint(
                    relation_type=RelationType.CONGRUENCE,
                    origin=ConstraintOrigin.EXPLICIT_NORMALIZED,
                    strength=ConstraintStrength.HARD,
                    lhs=self._make_operand(match.group("lhs"), symbol_table, scope_id, unresolved_operands),
                    rhs=self._make_operand(match.group("rhs"), symbol_table, scope_id, unresolved_operands),
                    operator="≡",
                    metadata={"modulus": self._make_operand(match.group("modulus"), symbol_table, scope_id, unresolved_operands).normalized_text},
                    domain_assumptions=[],
                    quantified_variables=self._active_quantifiers(match.start(), quantifiers),
                    explicit=True,
                    inferred_ready=True,
                    provenance=self._make_provenance(text, match.start(), match.end(), "congruence"),
                )
            )
        return out

    def _extract_divisibility(
        self,
        text: str,
        symbol_table: SymbolTable,
        scope_id: str,
        quantifiers: list[QuantifiedVariable],
        unresolved_operands: set[str],
    ) -> list[ConstraintNode]:
        out: list[ConstraintNode] = []
        patterns = [
            (_divides := self._divides_pattern, "divides", "divides"),
            (_bar := self._bar_divides_pattern, "bar_divides", "|"),
            (_is_div := self._is_divisible_by_pattern, "is_divisible_by", "divides"),
        ]
        for pattern, stage_name, operator in patterns:
            for match in pattern.finditer(text):
                if stage_name == "is_divisible_by":
                    lhs_text = match.group("divisor")
                    rhs_text = match.group("dividend")
                else:
                    lhs_text = match.group("divisor")
                    rhs_text = match.group("dividend")
                out.append(
                    self._make_constraint(
                        relation_type=RelationType.DIVISIBILITY,
                        origin=ConstraintOrigin.EXPLICIT_NORMALIZED,
                        strength=ConstraintStrength.HARD,
                        lhs=self._make_operand(lhs_text, symbol_table, scope_id, unresolved_operands),
                        rhs=self._make_operand(rhs_text, symbol_table, scope_id, unresolved_operands),
                        operator=operator,
                        metadata={},
                        domain_assumptions=[],
                        quantified_variables=self._active_quantifiers(match.start(), quantifiers),
                        explicit=True,
                        inferred_ready=True,
                        provenance=self._make_provenance(text, match.start(), match.end(), stage_name),
                    )
                )
        return out

    def _extract_relations(
        self,
        text: str,
        symbol_table: SymbolTable,
        scope_id: str,
        quantifiers: list[QuantifiedVariable],
        unresolved_operands: set[str],
    ) -> list[ConstraintNode]:
        out: list[ConstraintNode] = []
        seen: set[tuple[int, int]] = set()
        sentence_chunks = self._split_sentences(text)

        for sentence, sent_start, _sent_end in sentence_chunks:
            for match in self._relation_pattern.finditer(sentence):
                global_start = sent_start + match.start()
                global_end = sent_start + match.end()
                if (global_start, global_end) in seen:
                    continue
                seen.add((global_start, global_end))

                lhs_text = match.group("lhs").strip()
                rhs_text = match.group("rhs").strip()
                op = match.group("op")

                out.append(
                    self._make_constraint(
                        relation_type=self._relation_type_from_operator(op, lhs_text, rhs_text),
                        origin=ConstraintOrigin.EXPLICIT_NORMALIZED,
                        strength=ConstraintStrength.HARD,
                        lhs=self._make_operand(lhs_text, symbol_table, scope_id, unresolved_operands),
                        rhs=self._make_operand(rhs_text, symbol_table, scope_id, unresolved_operands),
                        operator=op,
                        metadata={},
                        domain_assumptions=[],
                        quantified_variables=self._active_quantifiers(global_start, quantifiers),
                        explicit=True,
                        inferred_ready=True,
                        provenance=self._make_provenance(text, global_start, global_end, "relation"),
                    )
                )
        return out

    def _extract_objectives(
        self,
        text: str,
        symbol_table: SymbolTable,
        scope_id: str,
        quantifiers: list[QuantifiedVariable],
        unresolved_operands: set[str],
    ) -> list[ConstraintNode]:
        out: list[ConstraintNode] = []

        for pattern, mode in [
            (self._find_target_pattern, QuantifierType.FIND_ALL),
            (self._how_many_pattern, QuantifierType.COUNT),
        ]:
            for match in pattern.finditer(text):
                target_text = match.group("target").strip()
                operand = self._make_operand(target_text, symbol_table, scope_id, unresolved_operands, operand_kind="objective")
                objective = ObjectiveSpec(
                    objective_mode=mode,
                    target_text=target_text,
                    normalized_target_text=operand.normalized_text,
                    subject_variables=[],
                )
                out.append(
                    self._make_constraint(
                        relation_type=RelationType.OBJECTIVE,
                        origin=ConstraintOrigin.EXPLICIT_OBJECTIVE,
                        strength=ConstraintStrength.HARD,
                        lhs=operand,
                        rhs=None,
                        operator="target",
                        metadata={"objective": objective.model_dump(mode="json")},
                        domain_assumptions=[],
                        quantified_variables=self._active_quantifiers(match.start(), quantifiers),
                        explicit=True,
                        inferred_ready=False,
                        provenance=self._make_provenance(text, match.start(), match.end(), "objective"),
                    )
                )

        for match in self._maxmin_pattern.finditer(text):
            mode = QuantifierType.MAXIMIZE if match.group("mode").lower() == "maximize" else QuantifierType.MINIMIZE
            target_text = match.group("target").strip()
            operand = self._make_operand(target_text, symbol_table, scope_id, unresolved_operands, operand_kind="objective")
            objective = ObjectiveSpec(
                objective_mode=mode,
                target_text=target_text,
                normalized_target_text=operand.normalized_text,
                subject_variables=[],
            )
            out.append(
                self._make_constraint(
                    relation_type=RelationType.OBJECTIVE,
                    origin=ConstraintOrigin.EXPLICIT_OBJECTIVE,
                    strength=ConstraintStrength.HARD,
                    lhs=operand,
                    rhs=None,
                    operator="target",
                    metadata={"objective": objective.model_dump(mode="json")},
                    domain_assumptions=[],
                    quantified_variables=self._active_quantifiers(match.start(), quantifiers),
                    explicit=True,
                    inferred_ready=False,
                    provenance=self._make_provenance(text, match.start(), match.end(), "objective"),
                )
            )
        return out

    def _make_operand(
        self,
        text: str,
        symbol_table: SymbolTable,
        scope_id: str,
        unresolved_operands: set[str],
        operand_kind: str | None = None,
    ) -> MathOperand:
        raw = text.strip()
        normalized = normalize_math_text(raw).strip()
        canonicalized = symbol_table.canonicalize_symbol_mentions(normalized, scope_id=scope_id)

        symbol_ids: list[str] = []
        canonical_mentions: list[str] = []

        for token in re.findall(PARSING.SIMPLE_SYMBOL_PATTERN, normalized):
            resolution: SymbolResolution = symbol_table.resolve(token, scope_id=scope_id)
            if resolution.symbol_id is not None:
                symbol_ids.append(resolution.symbol_id)
                if resolution.canonical_name:
                    canonical_mentions.append(resolution.canonical_name)
            else:
                if re.fullmatch(PARSING.SIMPLE_SYMBOL_PATTERN, token):
                    entry = symbol_table.register_symbol(token, scope_id=scope_id)
                    symbol_ids.append(entry.symbol_id)
                    canonical_mentions.append(entry.canonical_name)
                else:
                    unresolved_operands.add(token)

        kind = operand_kind or ("symbol" if re.fullmatch(PARSING.SIMPLE_SYMBOL_PATTERN, normalized) else "expression")
        return MathOperand(
            raw_text=raw,
            normalized_text=canonicalized,
            scope_id=scope_id,
            symbol_ids=list(dict.fromkeys(symbol_ids)),
            canonical_mentions=list(dict.fromkeys(canonical_mentions)),
            operand_kind=kind,
        )

    def _make_constraint(
        self,
        *,
        relation_type: RelationType,
        origin: ConstraintOrigin,
        strength: ConstraintStrength,
        lhs: MathOperand | None,
        rhs: MathOperand | None,
        operator: str | None,
        metadata: dict,
        domain_assumptions: list[str],
        quantified_variables: list[QuantifiedVariable],
        explicit: bool,
        inferred_ready: bool,
        provenance: ConstraintProvenance,
    ) -> ConstraintNode:
        basis = "|".join(
            [
                relation_type.value,
                lhs.normalized_text if lhs else "",
                operator or "",
                rhs.normalized_text if rhs else "",
                provenance.normalized_source_text,
                str(provenance.span_start),
                str(provenance.span_end),
            ]
        )
        return ConstraintNode(
            constraint_id=_stable_id("cst", basis),
            relation_type=relation_type,
            origin=origin,
            strength=strength,
            lhs=lhs,
            rhs=rhs,
            operator=operator,
            metadata=metadata,
            domain_assumptions=sorted(set(domain_assumptions)),
            quantified_variables=quantified_variables,
            explicit=explicit,
            inferred_ready=inferred_ready,
            provenance=provenance,
        )

    def _make_provenance(self, text: str, start: int, end: int, stage: str) -> ConstraintProvenance:
        source = text[start:end]
        return ConstraintProvenance(
            source_text=source,
            normalized_source_text=source,
            span_start=start,
            span_end=end,
            stage=stage,
            source_channel="text",
        )

    def _active_quantifiers(self, position: int, quantifiers: list[QuantifiedVariable]) -> list[QuantifiedVariable]:
        return [q for q in quantifiers if (q.span_start or -1) <= position]

    def _relation_type_from_operator(self, op: str, lhs_text: str, rhs_text: str) -> RelationType:
        if op == "=":
            return RelationType.EQUALITY
        if op == "!=":
            return RelationType.INEQUALITY
        if op in {"<", ">", "<=", ">="}:
            if re.fullmatch(r"-?\d+", lhs_text.strip()) or re.fullmatch(r"-?\d+", rhs_text.strip()):
                if op in {"<", ">"} and (lhs_text.strip() == "0" or rhs_text.strip() == "0"):
                    return RelationType.POSITIVITY
                return RelationType.BOUNDEDNESS
            return RelationType.ORDERING
        return RelationType.ORDERING

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        pieces: list[tuple[str, int, int]] = []
        start = 0
        for match in re.finditer(PARSING.SENTENCE_SPLIT_PATTERN, text):
            end = match.start()
            chunk = text[start:end].strip()
            if chunk:
                chunk_start = start + (0 if text[start:end] == text[start:end].lstrip() else len(text[start:end]) - len(text[start:end].lstrip()))
                pieces.append((chunk, chunk_start, end))
            start = match.end()
        tail = text[start:].strip()
        if tail:
            chunk_start = start + (0 if text[start:] == text[start:].lstrip() else len(text[start:]) - len(text[start:].lstrip()))
            pieces.append((tail, chunk_start, len(text)))
        return pieces

    def _dedupe(self, constraints: list[ConstraintNode]) -> list[ConstraintNode]:
        seen: dict[tuple, ConstraintNode] = {}
        for c in constraints:
            key = (
                c.relation_type.value,
                c.operator,
                c.lhs.normalized_text if c.lhs else None,
                c.rhs.normalized_text if c.rhs else None,
                tuple(c.domain_assumptions),
                tuple((q.quantifier_type.value, tuple(q.variables), q.raw_text) for q in c.quantified_variables),
            )
            if key not in seen:
                seen[key] = c
        return list(seen.values())


def extract_constraints(
    text: str,
    *,
    symbol_table: SymbolTable,
    scope_id: str | None = None,
) -> ConstraintExtractionResult:
    return ConstraintExtractor().extract(text, symbol_table=symbol_table, scope_id=scope_id)

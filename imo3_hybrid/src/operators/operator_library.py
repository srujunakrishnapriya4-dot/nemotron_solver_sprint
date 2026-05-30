from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.common.schemas import RouteDecision
from src.state_graph.node import ReasoningStateNode

from .operator_types import (
    ApplicabilityStatus,
    FailureMode,
    OperatorApplicability,
    OperatorCondition,
    OperatorDescriptor,
    OperatorFamily,
    OperatorNeighbor,
    OperatorPostcondition,
    RetrievalCompatibilityKind,
    OperatorUsageStats,
    PostconditionKind,
    PreconditionKind,
    ToolCapability,
)


def _safe_lower_set(values: Iterable[str]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


class OperatorLibrary:
    """
    Canonical runtime operator registry with:
    - typed operator descriptors
    - applicability checking against structured reasoning state
    - deterministic ordering
    - success statistics and neighbor/fallback support
    """

    def __init__(self) -> None:
        self._descriptors: dict[str, OperatorDescriptor] = {}
        self._usage_stats: dict[str, OperatorUsageStats] = {}
        self._load_core_registry()

    def _load_core_registry(self) -> None:
        for descriptor in _core_operator_descriptors():
            self.register(descriptor)

    def register(self, descriptor: OperatorDescriptor) -> None:
        self._descriptors[descriptor.operator_name] = descriptor
        self._usage_stats.setdefault(descriptor.operator_name, OperatorUsageStats())

    def register_many(self, descriptors: Iterable[OperatorDescriptor]) -> None:
        for descriptor in descriptors:
            self.register(descriptor)

    def get(self, operator_name: str) -> OperatorDescriptor | None:
        return self._descriptors.get(operator_name)

    def usage_stats(self, operator_name: str) -> OperatorUsageStats:
        return self._usage_stats.get(operator_name, OperatorUsageStats())

    def all_operator_names(self) -> list[str]:
        return sorted(self._descriptors.keys())

    def all_descriptors(self) -> list[OperatorDescriptor]:
        return [self._descriptors[name] for name in self.all_operator_names()]

    def family_members(self, family: OperatorFamily) -> list[str]:
        return sorted(
            [name for name, desc in self._descriptors.items() if desc.family is family]
        )

    def check_applicability(
        self,
        operator_name: str,
        node: ReasoningStateNode,
        route: RouteDecision | None = None,
        *,
        available_tools: Iterable[str] | None = None,
        retrieval_context: dict[str, Any] | None = None,
    ) -> OperatorApplicability:
        descriptor = self.get(operator_name)
        if descriptor is None:
            return OperatorApplicability(
                operator_name=operator_name,
                status=ApplicabilityStatus.INAPPLICABLE,
                score=0.0,
                failed_conditions=["missing_descriptor"],
                failure_modes=[FailureMode.UNKNOWN],
                diagnostics={},
            )

        state_features = self._extract_state_features(node, route)
        tool_set = _safe_lower_set(available_tools or [])

        passed: list[str] = []
        failed: list[str] = []
        warnings: list[str] = []
        failure_modes: list[FailureMode] = []

        total_weight = 0.0
        passed_weight = 0.0

        for condition in descriptor.preconditions:
            total_weight += condition.weight
            ok, detail, failure_mode = self._evaluate_condition(
                condition=condition,
                features=state_features,
                tool_set=tool_set,
            )
            if ok:
                passed.append(detail)
                passed_weight += condition.weight
            else:
                failed.append(detail)
                if failure_mode is not None:
                    failure_modes.append(failure_mode)

        if descriptor.compatible_domains and route is not None:
            top_domain = self._top_domain(route)
            if top_domain and top_domain not in descriptor.compatible_domains:
                failed.append(f"domain_not_compatible:{top_domain}")
                failure_modes.append(FailureMode.DOMAIN_MISMATCH)
            elif top_domain:
                passed.append(f"domain_compatible:{top_domain}")

        if descriptor.compatible_archetypes and route is not None:
            top_archetypes = self._top_archetypes(route, top_k=4)
            overlap = sorted(set(top_archetypes) & set(descriptor.compatible_archetypes))
            if not overlap:
                warnings.append("no_top_archetype_overlap")
                failure_modes.append(FailureMode.ARCHETYPE_MISMATCH)
            else:
                passed.append(f"archetype_overlap:{','.join(overlap)}")

        if descriptor.tool_capabilities:
            required_tools = {cap.value for cap in descriptor.tool_capabilities if cap is not ToolCapability.NONE}
            if required_tools and not (required_tools & tool_set):
                warnings.append("preferred_tool_not_available")
                failure_modes.append(FailureMode.TOOL_UNAVAILABLE)

        score = 1.0 if total_weight <= 0 else max(0.0, min(1.0, passed_weight / total_weight))
        retrieval_score, retrieval_signals = self._retrieval_compatibility_score(
            descriptor=descriptor,
            features=state_features,
            retrieval_context=retrieval_context or {},
        )
        if retrieval_score > 0.0:
            score = max(score, min(1.0, 0.70 * score + 0.30 * retrieval_score))

        if failed:
            status = ApplicabilityStatus.WEAK if score >= 0.45 else ApplicabilityStatus.INAPPLICABLE
        else:
            status = ApplicabilityStatus.APPLICABLE if score >= 0.60 else ApplicabilityStatus.WEAK

        if operator_name == "brute_force_small":
            search_bound = float(state_features.get("finite_search_bound", 0.0) or 0.0)
            if search_bound > 10000:
                status = ApplicabilityStatus.INAPPLICABLE
                score = 0.0
                failed.append("finite_search_bound_too_large")
                failure_modes.append(FailureMode.SEARCH_SPACE_TOO_LARGE)

        return OperatorApplicability(
            operator_name=operator_name,
            status=status,
            score=round(score, 6),
            passed_conditions=passed,
            failed_conditions=failed,
            warnings=warnings,
            failure_modes=list(dict.fromkeys(failure_modes)),
            retrieval_compatibility_score=round(retrieval_score, 6),
            compatible_retrieval_signals=retrieval_signals,
            diagnostics={**state_features, "retrieval_context": dict(retrieval_context or {})},
        )

    def compatible_operators(
        self,
        node: ReasoningStateNode,
        route: RouteDecision | None = None,
        *,
        available_tools: Iterable[str] | None = None,
        include_weak: bool = True,
    ) -> list[OperatorApplicability]:
        out: list[OperatorApplicability] = []
        for name in self.all_operator_names():
            app = self.check_applicability(name, node, route, available_tools=available_tools)
            if app.status is ApplicabilityStatus.APPLICABLE or (include_weak and app.status is ApplicabilityStatus.WEAK):
                out.append(app)
        out.sort(key=lambda a: (-a.score, a.operator_name))
        return out

    def fallback_neighbors(self, operator_name: str) -> list[OperatorNeighbor]:
        descriptor = self.get(operator_name)
        if descriptor is None:
            return []
        return sorted(descriptor.repair_neighbors, key=lambda x: (-x.weight, x.operator_name))

    def update_usage_stats(
        self,
        operator_name: str,
        *,
        success: bool,
        verifier_gain: float = 0.0,
        symbolic_ok: bool | None = None,
        repair_triggered: bool = False,
    ) -> None:
        current = self._usage_stats.get(operator_name, OperatorUsageStats())
        self._usage_stats[operator_name] = current.updated(
            success=success,
            verifier_gain=verifier_gain,
            symbolic_ok=symbolic_ok,
            repair_triggered=repair_triggered,
        )

    def export_registry(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.all_operator_names():
            desc = self._descriptors[name]
            stats = self._usage_stats.get(name, OperatorUsageStats())
            rows.append(
                {
                    "descriptor": desc.model_dump(mode="json"),
                    "usage_stats": stats.model_dump(mode="json"),
                }
            )
        return rows

    def save_json(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.export_registry(), f, indent=2, ensure_ascii=False, sort_keys=True)

    @classmethod
    def load_json(cls, path: str) -> "OperatorLibrary":
        lib = cls()
        target = Path(path)
        if not target.exists():
            return lib
        data = json.loads(target.read_text(encoding="utf-8"))
        lib._descriptors.clear()
        lib._usage_stats.clear()
        for row in data:
            desc = OperatorDescriptor(**row["descriptor"])
            stats = OperatorUsageStats(**row["usage_stats"])
            lib._descriptors[desc.operator_name] = desc
            lib._usage_stats[desc.operator_name] = stats
        return lib

    def load_mined_rows(self, rows: list[dict[str, Any]]) -> int:
        loaded = 0
        for row in rows:
            try:
                descriptor = OperatorDescriptor(**row)
            except Exception:
                continue
            if descriptor.operator_name not in self._descriptors:
                self.register(descriptor)
                loaded += 1
        return loaded

    def _extract_state_features(
        self,
        node: ReasoningStateNode,
        route: RouteDecision | None,
    ) -> dict[str, Any]:
        constraints = list(getattr(node, "constraints", []) or [])
        invariants = list(getattr(node, "invariants", []) or [])
        goals = list(getattr(node, "goals", []) or [])
        extracted_objects = dict(getattr(node, "extracted_objects", {}) or {})
        metadata = dict(getattr(node, "metadata", {}) or {})
        proof_obligations = list(getattr(node, "proof_obligations", ()) or ())

        constraint_texts = [
            getattr(c, "normalized_lhs", None) or getattr(c, "lhs", None) or getattr(c, "summary", lambda: "")()
            for c in constraints
        ]
        constraint_blob = " ".join(str(x) for x in constraint_texts if x)

        invariant_blob = " ".join(
            str(getattr(inv, "normalized_expression", "") or getattr(inv, "expression", ""))
            for inv in invariants
        )

        goal_blob = " ".join(
            str(getattr(g, "normalized_text", "") or getattr(g, "text", ""))
            for g in goals
        )

        domain = ""
        if route is not None:
            domain = self._top_domain(route) or ""
        if not domain:
            domain = str(extracted_objects.get("domain", "")).lower()

        archetypes = self._top_archetypes(route, top_k=6) if route is not None else []

        finite_bound = metadata.get("finite_search_bound")
        if finite_bound is None:
            finite_bound = metadata.get("search_space_bound", 0)

        open_obligations = [item for item in proof_obligations if getattr(item, "is_open", lambda: False)()]
        contradicted_obligations = [
            item
            for item in proof_obligations
            if str(getattr(getattr(item, "status", None), "value", getattr(item, "status", ""))) == "contradicted"
        ]
        evidence_kinds = [
            str(getattr(getattr(item, "evidence_kind_required", None), "value", getattr(item, "evidence_kind_required", "")))
            for item in open_obligations
            if getattr(getattr(item, "evidence_kind_required", None), "value", getattr(item, "evidence_kind_required", ""))
        ]
        operator_prefix = [
            getattr(step, "operator_name", "")
            for step in list(getattr(node, "operator_history", ()) or ())[-3:]
            if getattr(step, "operator_name", "")
        ]
        failure_modes: list[str] = []
        for key in ("failure_type", "failure_mode", "latest_failure_type", "active_failure_mode"):
            value = metadata.get(key)
            if value:
                failure_modes.append(str(value))

        return {
            "domain": domain,
            "archetypes": archetypes,
            "constraint_count": len(constraints),
            "invariant_count": len(invariants),
            "goal_count": len(goals),
            "has_integer_domain": "integer" in constraint_blob.lower()
            or "natural" in constraint_blob.lower()
            or domain == "number_theory",
            "has_parity_signal": "parity" in invariant_blob.lower()
            or "even" in constraint_blob.lower()
            or "odd" in constraint_blob.lower(),
            "has_modular_signal": "mod" in constraint_blob.lower()
            or "congru" in constraint_blob.lower(),
            "has_symmetry_signal": "symmetry" in invariant_blob.lower()
            or "wlog" in goal_blob.lower(),
            "has_polynomial_signal": "polynomial" in goal_blob.lower()
            or "^" in constraint_blob
            or "root" in constraint_blob.lower(),
            "has_optimization_signal": "maximize" in goal_blob.lower()
            or "minimize" in goal_blob.lower()
            or "maximum" in goal_blob.lower()
            or "minimum" in goal_blob.lower(),
            "has_counting_signal": "count" in goal_blob.lower()
            or "ways" in goal_blob.lower()
            or "bijection" in invariant_blob.lower(),
            "has_geometry_entities": bool(extracted_objects.get("geometry_entities")),
            "has_recurrence_signal": "recurrence" in goal_blob.lower()
            or "sequence" in goal_blob.lower(),
            "open_goal_count": len([g for g in goals if getattr(g, "status", "open") != "satisfied"]),
            "open_obligation_count": len(open_obligations),
            "contradicted_obligation_count": len(contradicted_obligations),
            "required_evidence_kinds": tuple(sorted(dict.fromkeys(item.lower() for item in evidence_kinds if item))),
            "failure_modes": tuple(sorted(dict.fromkeys(item.lower() for item in failure_modes if item))),
            "operator_prefix": tuple(operator_prefix),
            "finite_search_bound": float(finite_bound or 0),
            "symbolic_valid": bool(getattr(node, "symbolic_valid", False)),
            "merge_ready": bool(getattr(node, "merge_ready", False)),
        }

    def _retrieval_compatibility_score(
        self,
        *,
        descriptor: OperatorDescriptor,
        features: dict[str, Any],
        retrieval_context: dict[str, Any],
    ) -> tuple[float, list[str]]:
        if not descriptor.retrieval_compatibility and not retrieval_context:
            return 0.0, []

        context_arch = {str(item).lower() for item in retrieval_context.get("archetypes", ()) if str(item).strip()}
        context_evidence = {str(item).lower() for item in retrieval_context.get("evidence_kinds", ()) if str(item).strip()}
        context_failures = {str(item).lower() for item in retrieval_context.get("failure_modes", ()) if str(item).strip()}
        context_prefix = {str(item).lower() for item in retrieval_context.get("operator_prefix", ()) if str(item).strip()}
        context_tags = {str(item).lower() for item in retrieval_context.get("tags", ()) if str(item).strip()}
        context_repair = {str(item).lower() for item in retrieval_context.get("repair_operators", ()) if str(item).strip()}

        matched_weight = 0.0
        total_weight = 0.0
        reasons: list[str] = []
        for item in descriptor.retrieval_compatibility:
            total_weight += item.weight
            value = item.value.lower()
            matched = False
            if item.kind is RetrievalCompatibilityKind.ARCHETYPE:
                matched = value in context_arch or value in {str(x).lower() for x in features.get("archetypes", ())}
            elif item.kind is RetrievalCompatibilityKind.EVIDENCE_KIND:
                matched = value in context_evidence or value in set(features.get("required_evidence_kinds", ()))
            elif item.kind is RetrievalCompatibilityKind.FAILURE_MODE:
                matched = value in context_failures or value in set(features.get("failure_modes", ()))
            elif item.kind is RetrievalCompatibilityKind.OPERATOR_PREFIX:
                matched = value in context_prefix or value in {str(x).lower() for x in features.get("operator_prefix", ())}
            elif item.kind is RetrievalCompatibilityKind.REPAIR_CONTEXT:
                matched = value in context_repair
            elif item.kind is RetrievalCompatibilityKind.TAG:
                matched = value in context_tags
            if matched:
                matched_weight += item.weight
                reasons.append(f"{item.kind.value}:{item.value}")

        if not descriptor.retrieval_compatibility:
            return 0.0, []
        return max(0.0, min(1.0, matched_weight / max(total_weight, 1e-6))), reasons[:6]

    def _evaluate_condition(
        self,
        *,
        condition: OperatorCondition,
        features: dict[str, Any],
        tool_set: set[str],
    ) -> tuple[bool, str, FailureMode | None]:
        field_value = features.get(condition.field_name)

        if condition.kind is PreconditionKind.TOOL:
            ok = str(condition.expected_value or "").lower() in tool_set
            if condition.negate:
                ok = not ok
            return ok, f"tool:{condition.expected_value}", None if ok else FailureMode.TOOL_UNAVAILABLE

        if condition.operator == "truthy":
            ok = bool(field_value)
        elif condition.operator == "equals":
            ok = str(field_value).lower() == str(condition.expected_value).lower()
        elif condition.operator == "contains":
            if isinstance(field_value, (list, tuple, set)):
                ok = str(condition.expected_value).lower() in {str(v).lower() for v in field_value}
            else:
                ok = str(condition.expected_value).lower() in str(field_value).lower()
        elif condition.operator == "gte":
            ok = float(field_value or 0.0) >= float(condition.expected_value or 0.0)
        elif condition.operator == "lte":
            ok = float(field_value or 0.0) <= float(condition.expected_value or 0.0)
        else:
            ok = bool(field_value)

        if condition.negate:
            ok = not ok

        failure_mode: FailureMode | None = None
        if not ok:
            if condition.kind is PreconditionKind.DOMAIN:
                failure_mode = FailureMode.DOMAIN_MISMATCH
            elif condition.kind is PreconditionKind.ARCHETYPE:
                failure_mode = FailureMode.ARCHETYPE_MISMATCH
            elif condition.kind is PreconditionKind.SEARCH_SPACE:
                failure_mode = FailureMode.SEARCH_SPACE_TOO_LARGE
            elif condition.kind is PreconditionKind.INVARIANT:
                failure_mode = FailureMode.INVARIANT_CONFLICT
            else:
                failure_mode = FailureMode.PRECONDITION_FAIL

        detail = f"{condition.field_name}:{condition.operator}:{condition.expected_value}"
        return ok, detail, failure_mode

    @staticmethod
    def _top_domain(route: RouteDecision) -> str | None:
        distribution = getattr(route, "problem_type", {}) or {}
        if not distribution:
            return None
        return sorted(distribution.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]

    @staticmethod
    def _top_archetypes(route: RouteDecision, *, top_k: int = 4) -> list[str]:
        distribution = getattr(route, "archetypes", {}) or {}
        return [name for name, _ in sorted(distribution.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:top_k]]


def _cond(
    *,
    cid: str,
    kind: PreconditionKind,
    field_name: str,
    operator: str = "truthy",
    expected: str | None = None,
    weight: float = 1.0,
    description: str = "",
) -> OperatorCondition:
    return OperatorCondition(
        condition_id=cid,
        kind=kind,
        field_name=field_name,
        operator=operator,
        expected_value=expected,
        weight=weight,
        description=description,
    )


def _post(pid: str, kind: PostconditionKind, description: str) -> OperatorPostcondition:
    return OperatorPostcondition(
        postcondition_id=pid,
        kind=kind,
        description=description,
    )


def _neighbors(*names: tuple[str, str, float]) -> list[OperatorNeighbor]:
    out: list[OperatorNeighbor] = []
    for name, reason, weight in names:
        out.append(OperatorNeighbor(operator_name=name, reason=reason, weight=weight))
    return out


def _core_operator_descriptors() -> list[OperatorDescriptor]:
    return [
        OperatorDescriptor(
            operator_name="substitution",
            family=OperatorFamily.ALGEBRAIC,
            description="Substitute one expression into another to reduce degrees of freedom.",
            compatible_domains=["algebra", "number_theory", "mixed"],
            compatible_archetypes=["symmetry", "bounding", "construction"],
            tool_capabilities=[ToolCapability.SYMPY],
            preconditions=[
                _cond(cid="sub-1", kind=PreconditionKind.CONSTRAINT, field_name="constraint_count", operator="gte", expected="1"),
            ],
            postconditions=[
                _post("sub-p1", PostconditionKind.REPRESENTATION_CHANGED, "simplifies the active constraint system"),
            ],
            failure_modes=[FailureMode.PRECONDITION_FAIL],
            repair_neighbors=_neighbors(
                ("symbolic_manipulation", "algebraic_simplification_neighbor", 1.0),
                ("symmetry_reduction", "structure_preserving_fallback", 0.6),
            ),
            tags=["rewrite", "algebra"],
        ),
        OperatorDescriptor(
            operator_name="symbolic_manipulation",
            family=OperatorFamily.ALGEBRAIC,
            description="Factor, expand, normalize, or rearrange symbolic relations.",
            compatible_domains=["algebra", "number_theory", "mixed"],
            compatible_archetypes=["symmetry", "contradiction", "bounding"],
            tool_capabilities=[ToolCapability.SYMPY],
            preconditions=[
                _cond(cid="symb-1", kind=PreconditionKind.CONSTRAINT, field_name="constraint_count", operator="gte", expected="1"),
            ],
            postconditions=[
                _post("symb-p1", PostconditionKind.REPRESENTATION_CHANGED, "algebraically normalized representation"),
            ],
            repair_neighbors=_neighbors(
                ("substitution", "equivalent_rewrite_neighbor", 0.9),
                ("vieta", "polynomial_structure_neighbor", 0.5),
            ),
        ),
        OperatorDescriptor(
            operator_name="modular_arithmetic",
            family=OperatorFamily.NUMBER_THEORETIC,
            description="Introduce congruence reasoning and modulus-based pruning.",
            compatible_domains=["number_theory", "mixed", "algebra"],
            compatible_archetypes=["modular", "contradiction", "invariant"],
            tool_capabilities=[ToolCapability.NUMBER_THEORY, ToolCapability.BRUTE_FORCE],
            preconditions=[
                _cond(cid="mod-1", kind=PreconditionKind.STATE, field_name="has_integer_domain", operator="truthy"),
            ],
            postconditions=[
                _post("mod-p1", PostconditionKind.INVARIANT_ADDED, "modular invariant or congruence class added"),
                _post("mod-p2", PostconditionKind.SEARCH_SPACE_REDUCED, "residue classes prune candidates"),
            ],
            failure_modes=[FailureMode.DOMAIN_MISMATCH],
            repair_neighbors=_neighbors(
                ("parity_mod_reduction", "smaller_modulus_fallback", 1.0),
                ("contradiction", "modular_contradiction_neighbor", 0.7),
            ),
            tags=["mod", "congruence"],
        ),
        OperatorDescriptor(
            operator_name="parity_mod_reduction",
            family=OperatorFamily.NUMBER_THEORETIC,
            description="Use parity or tiny moduli as a cheap number-theoretic reduction.",
            compatible_domains=["number_theory", "mixed"],
            compatible_archetypes=["modular", "invariant", "contradiction"],
            tool_capabilities=[ToolCapability.BRUTE_FORCE, ToolCapability.NUMBER_THEORY],
            preconditions=[
                _cond(cid="par-1", kind=PreconditionKind.STATE, field_name="has_integer_domain", operator="truthy"),
            ],
            postconditions=[
                _post("par-p1", PostconditionKind.SEARCH_SPACE_REDUCED, "parity constraints remove impossible cases"),
            ],
            repair_neighbors=_neighbors(
                ("modular_arithmetic", "lift_to_stronger_modulus", 0.9),
                ("contradiction", "use_obtained_parity_conflict", 0.7),
            ),
            tags=["parity", "cheap_check"],
        ),
        OperatorDescriptor(
            operator_name="invariant_introduction",
            family=OperatorFamily.STRUCTURAL,
            description="Propose a conserved or monotone quantity and attach it to the state.",
            compatible_domains=["combinatorics", "number_theory", "mixed", "algebra"],
            compatible_archetypes=["invariant", "extremal", "symmetry", "modular"],
            tool_capabilities=[ToolCapability.NONE],
            preconditions=[
                _cond(cid="inv-1", kind=PreconditionKind.CONSTRAINT, field_name="constraint_count", operator="gte", expected="1"),
            ],
            postconditions=[
                _post("inv-p1", PostconditionKind.INVARIANT_ADDED, "active invariant strengthens state pruning"),
            ],
            repair_neighbors=_neighbors(
                ("extremal_argument", "monovariant_neighbor", 0.8),
                ("symmetry_reduction", "structural_compression_neighbor", 0.6),
            ),
            tags=["invariant", "monovariant"],
        ),
        OperatorDescriptor(
            operator_name="symmetry_reduction",
            family=OperatorFamily.STRUCTURAL,
            description="Exploit symmetry or WLOG ordering to compress equivalent cases.",
            compatible_domains=["algebra", "geometry", "combinatorics", "mixed"],
            compatible_archetypes=["symmetry", "bounding", "construction"],
            tool_capabilities=[ToolCapability.NONE],
            preconditions=[
                _cond(cid="sym-1", kind=PreconditionKind.STATE, field_name="has_symmetry_signal", operator="truthy"),
            ],
            postconditions=[
                _post("sym-p1", PostconditionKind.SEARCH_SPACE_REDUCED, "equivalent cases collapsed"),
                _post("sym-p2", PostconditionKind.REPRESENTATION_CHANGED, "canonicalized variable ordering"),
            ],
            repair_neighbors=_neighbors(
                ("substitution", "rewrite_after_symmetry", 0.7),
                ("coordinate_change", "geometry_symmetry_neighbor", 0.6),
            ),
            tags=["wlog", "canonicalization"],
        ),
        OperatorDescriptor(
            operator_name="bounding",
            family=OperatorFamily.ALGEBRAIC,
            description="Derive upper/lower bounds and squeeze toward the target.",
            compatible_domains=["algebra", "combinatorics", "mixed"],
            compatible_archetypes=["bounding", "extremal"],
            tool_capabilities=[ToolCapability.SYMPY],
            preconditions=[
                _cond(cid="bnd-1", kind=PreconditionKind.STATE, field_name="has_optimization_signal", operator="truthy", weight=1.2),
            ],
            postconditions=[
                _post("bnd-p1", PostconditionKind.SEARCH_SPACE_REDUCED, "bounds shrink feasible set"),
            ],
            repair_neighbors=_neighbors(
                ("extremal_argument", "extremal_companion", 0.9),
                ("am_gm", "inequality_specialization", 0.7),
                ("cauchy_schwarz", "inequality_specialization", 0.6),
            ),
        ),
        OperatorDescriptor(
            operator_name="extremal_argument",
            family=OperatorFamily.COMBINATORIAL,
            description="Select a minimal or maximal object and reason from that extremal choice.",
            compatible_domains=["combinatorics", "number_theory", "mixed"],
            compatible_archetypes=["extremal", "invariant", "pigeonhole"],
            preconditions=[
                _cond(cid="ext-1", kind=PreconditionKind.STATE, field_name="has_optimization_signal", operator="truthy", weight=1.1),
            ],
            postconditions=[
                _post("ext-p1", PostconditionKind.GOAL_REFINED, "proof shifts to extremal witness"),
            ],
            repair_neighbors=_neighbors(
                ("bounding", "bound_extremal_value", 0.8),
                ("contradiction", "derive_extremal_conflict", 0.7),
            ),
        ),
        OperatorDescriptor(
            operator_name="contradiction",
            family=OperatorFamily.STRUCTURAL,
            description="Assume the negation and derive inconsistency with constraints or invariants.",
            compatible_domains=["algebra", "number_theory", "geometry", "combinatorics", "mixed"],
            compatible_archetypes=["contradiction", "pigeonhole", "invariant", "modular"],
            preconditions=[
                _cond(cid="con-1", kind=PreconditionKind.GOAL, field_name="open_goal_count", operator="gte", expected="1"),
            ],
            postconditions=[
                _post("con-p1", PostconditionKind.CONTRADICTION_EXPOSED, "impossible branch or assumption refuted"),
            ],
            repair_neighbors=_neighbors(
                ("case_work", "enumerate_assumption_failures", 0.6),
                ("modular_arithmetic", "derive_arithmetic_conflict", 0.6),
            ),
        ),
        OperatorDescriptor(
            operator_name="induction",
            family=OperatorFamily.STRUCTURAL,
            description="Introduce a recursive induction structure over n or construction depth.",
            compatible_domains=["number_theory", "combinatorics", "mixed"],
            compatible_archetypes=["induction", "construction"],
            tool_capabilities=[ToolCapability.BRUTE_FORCE],
            preconditions=[
                _cond(cid="ind-1", kind=PreconditionKind.STATE, field_name="has_recurrence_signal", operator="truthy"),
            ],
            postconditions=[
                _post("ind-p1", PostconditionKind.GOAL_REFINED, "base case and induction step created"),
            ],
            repair_neighbors=_neighbors(
                ("case_work", "expand_small_base_cases", 0.6),
                ("constructive_build", "convert_induction_to_explicit_step", 0.5),
            ),
        ),
        OperatorDescriptor(
            operator_name="case_work",
            family=OperatorFamily.SEARCH,
            description="Split into a small number of explicit logically exhaustive cases.",
            compatible_domains=["algebra", "number_theory", "combinatorics", "geometry", "mixed"],
            compatible_archetypes=["contradiction", "construction", "modular", "symmetry"],
            preconditions=[
                _cond(cid="case-1", kind=PreconditionKind.GOAL, field_name="open_goal_count", operator="gte", expected="1"),
            ],
            postconditions=[
                _post("case-p1", PostconditionKind.CASE_SPLIT_CREATED, "branch expansion over explicit cases"),
            ],
            repair_neighbors=_neighbors(
                ("contradiction", "turn_bad_case_into_conflict", 0.7),
                ("constructive_build", "case_specific_construction", 0.5),
            ),
        ),
        OperatorDescriptor(
            operator_name="double_counting",
            family=OperatorFamily.COMBINATORIAL,
            description="Count or evaluate the same structure from two viewpoints.",
            compatible_domains=["combinatorics", "mixed"],
            compatible_archetypes=["bijection", "pigeonhole", "construction"],
            preconditions=[
                _cond(cid="dc-1", kind=PreconditionKind.STATE, field_name="has_counting_signal", operator="truthy"),
            ],
            postconditions=[
                _post("dc-p1", PostconditionKind.CONSTRAINT_ADDED, "new equality from two counts"),
            ],
            repair_neighbors=_neighbors(
                ("bijection", "structural_counting_neighbor", 0.8),
                ("pigeonhole", "force_repetition_after_count", 0.5),
            ),
        ),
        OperatorDescriptor(
            operator_name="bijection",
            family=OperatorFamily.COMBINATORIAL,
            description="Build a structure-preserving mapping between two counted sets.",
            compatible_domains=["combinatorics", "mixed"],
            compatible_archetypes=["bijection", "construction", "double_counting"],
            preconditions=[
                _cond(cid="bij-1", kind=PreconditionKind.STATE, field_name="has_counting_signal", operator="truthy"),
            ],
            postconditions=[
                _post("bij-p1", PostconditionKind.CONSTRAINT_ADDED, "equal cardinality via explicit correspondence"),
            ],
            repair_neighbors=_neighbors(
                ("double_counting", "counting_fallback", 0.8),
                ("constructive_build", "construct_the_map", 0.6),
            ),
        ),
        OperatorDescriptor(
            operator_name="pigeonhole",
            family=OperatorFamily.COMBINATORIAL,
            description="Force repetition or collision by comparing objects and containers.",
            compatible_domains=["combinatorics", "number_theory", "mixed"],
            compatible_archetypes=["pigeonhole", "contradiction", "extremal"],
            preconditions=[
                _cond(cid="pig-1", kind=PreconditionKind.STATE, field_name="has_counting_signal", operator="truthy"),
            ],
            postconditions=[
                _post("pig-p1", PostconditionKind.CONSTRAINT_ADDED, "existence of collision or repetition"),
            ],
            repair_neighbors=_neighbors(
                ("contradiction", "collision_implies_conflict", 0.8),
                ("double_counting", "count_to_force_container_overflow", 0.6),
            ),
        ),
        OperatorDescriptor(
            operator_name="constructive_build",
            family=OperatorFamily.SEARCH,
            description="Explicitly construct an object, assignment, or witness satisfying the target.",
            compatible_domains=["combinatorics", "geometry", "mixed", "algebra"],
            compatible_archetypes=["construction", "bijection", "induction"],
            preconditions=[
                _cond(cid="cb-1", kind=PreconditionKind.GOAL, field_name="open_goal_count", operator="gte", expected="1"),
            ],
            postconditions=[
                _post("cb-p1", PostconditionKind.GOAL_REFINED, "witness or constructive pattern proposed"),
            ],
            repair_neighbors=_neighbors(
                ("case_work", "constructive_cases", 0.6),
                ("induction", "recursive_construction_neighbor", 0.5),
            ),
        ),
        OperatorDescriptor(
            operator_name="coordinate_change",
            family=OperatorFamily.GEOMETRIC,
            description="Change coordinates or basis to expose geometry/algebra structure.",
            compatible_domains=["geometry", "algebra", "mixed"],
            compatible_archetypes=["coordinate_geometry", "symmetry", "construction"],
            tool_capabilities=[ToolCapability.GEOMETRY, ToolCapability.SYMPY],
            preconditions=[
                _cond(cid="coord-1", kind=PreconditionKind.STATE, field_name="has_geometry_entities", operator="truthy"),
            ],
            postconditions=[
                _post("coord-p1", PostconditionKind.REPRESENTATION_CHANGED, "problem transferred to coordinates"),
            ],
            repair_neighbors=_neighbors(
                ("inversion", "geometry_transformation_neighbor", 0.6),
                ("symmetry_reduction", "coordinate_canonicalization_neighbor", 0.5),
            ),
        ),
        OperatorDescriptor(
            operator_name="inversion",
            family=OperatorFamily.GEOMETRIC,
            description="Apply inversive or projective transformation to simplify incidence structure.",
            compatible_domains=["geometry", "mixed"],
            compatible_archetypes=["coordinate_geometry", "symmetry", "construction"],
            tool_capabilities=[ToolCapability.GEOMETRY],
            preconditions=[
                _cond(cid="invgeo-1", kind=PreconditionKind.STATE, field_name="has_geometry_entities", operator="truthy"),
            ],
            postconditions=[
                _post("invgeo-p1", PostconditionKind.REPRESENTATION_CHANGED, "circles/lines transformed to simpler incidence form"),
            ],
            repair_neighbors=_neighbors(
                ("coordinate_change", "geometry_representation_fallback", 0.8),
                ("symmetry_reduction", "preserve_geometric_structure", 0.4),
            ),
        ),
        OperatorDescriptor(
            operator_name="vieta",
            family=OperatorFamily.ALGEBRAIC,
            description="Use polynomial-root relations to trade root structure for coefficient structure.",
            compatible_domains=["algebra", "mixed"],
            compatible_archetypes=["symmetry", "invariant", "construction"],
            tool_capabilities=[ToolCapability.SYMPY],
            preconditions=[
                _cond(cid="vieta-1", kind=PreconditionKind.STATE, field_name="has_polynomial_signal", operator="truthy"),
            ],
            postconditions=[
                _post("vieta-p1", PostconditionKind.CONSTRAINT_ADDED, "symmetric root relations added"),
            ],
            repair_neighbors=_neighbors(
                ("symbolic_manipulation", "polynomial_cleanup_neighbor", 0.7),
                ("substitution", "switch_to_root_sum_variables", 0.7),
            ),
        ),
        OperatorDescriptor(
            operator_name="am_gm",
            family=OperatorFamily.ALGEBRAIC,
            description="Apply AM-GM to nonnegative expressions to derive bounds or equality conditions.",
            compatible_domains=["algebra", "mixed"],
            compatible_archetypes=["bounding", "extremal"],
            tool_capabilities=[ToolCapability.SYMPY],
            preconditions=[
                _cond(cid="amgm-1", kind=PreconditionKind.STATE, field_name="has_optimization_signal", operator="truthy"),
            ],
            postconditions=[
                _post("amgm-p1", PostconditionKind.SEARCH_SPACE_REDUCED, "inequality narrows feasible region"),
            ],
            repair_neighbors=_neighbors(
                ("bounding", "general_bound_neighbor", 0.8),
                ("cauchy_schwarz", "inequality_family_neighbor", 0.5),
            ),
        ),
        OperatorDescriptor(
            operator_name="cauchy_schwarz",
            family=OperatorFamily.ALGEBRAIC,
            description="Use Cauchy-Schwarz or related quadratic bounds.",
            compatible_domains=["algebra", "mixed"],
            compatible_archetypes=["bounding", "extremal"],
            tool_capabilities=[ToolCapability.SYMPY],
            preconditions=[
                _cond(cid="cs-1", kind=PreconditionKind.STATE, field_name="has_optimization_signal", operator="truthy"),
            ],
            postconditions=[
                _post("cs-p1", PostconditionKind.SEARCH_SPACE_REDUCED, "quadratic inequality bound introduced"),
            ],
            repair_neighbors=_neighbors(
                ("am_gm", "inequality_family_neighbor", 0.6),
                ("bounding", "general_bound_neighbor", 0.7),
            ),
        ),
        OperatorDescriptor(
            operator_name="brute_force_small",
            family=OperatorFamily.VERIFICATION,
            description="Enumerate small finite candidates exactly when the search space is safe.",
            compatible_domains=["number_theory", "combinatorics", "mixed", "algebra"],
            compatible_archetypes=["modular", "construction", "invariant", "case_work"],
            tool_capabilities=[ToolCapability.BRUTE_FORCE, ToolCapability.SEARCH_ENUMERATOR],
            preconditions=[
                _cond(cid="bf-1", kind=PreconditionKind.SEARCH_SPACE, field_name="finite_search_bound", operator="lte", expected="10000", weight=1.3),
            ],
            postconditions=[
                _post("bf-p1", PostconditionKind.VERIFICATION_SIGNAL_ADDED, "explicit candidate verification result attached"),
            ],
            failure_modes=[FailureMode.SEARCH_SPACE_TOO_LARGE],
            repair_neighbors=_neighbors(
                ("modular_arithmetic", "prune_before_enumeration", 0.8),
                ("case_work", "shrink_search_to_cases", 0.6),
            ),
            tags=["verification", "enumeration"],
        ),
    ]


__all__ = ["OperatorLibrary"]

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from src.common.constants import PARSING
from src.common.schemas import DomainSpec, IndexedSymbolSpec, SymbolAlias, SymbolRef, SymbolRole
from src.parsing.latex_normalizer import normalize_math_text


@dataclass(frozen=True)
class ScopeRecord:
    scope_id: str
    kind: str
    parent_scope_id: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolResolution:
    query_text: str
    normalized_query: str
    scope_id: str
    symbol_id: str | None
    canonical_name: str | None
    matched_via: str


def _stable_digest(payload: str, prefix: str) -> str:
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


class SymbolTable:
    def __init__(self, problem_namespace: str = "problem") -> None:
        self.problem_namespace = normalize_math_text(problem_namespace or "problem") or "problem"
        self.root_scope_id = _stable_digest(f"scope::{self.problem_namespace}::root", "scope")
        self._scopes: dict[str, ScopeRecord] = {
            self.root_scope_id: ScopeRecord(
                scope_id=self.root_scope_id,
                kind="problem",
                parent_scope_id=None,
                label=PARSING.DEFAULT_SCOPE_LABEL,
                metadata={"problem_namespace": self.problem_namespace},
            )
        }
        self._symbols_by_id: dict[str, SymbolRef] = {}
        self._bindings_by_scope: dict[str, dict[str, str]] = {self.root_scope_id: {}}
        self._aliases_by_scope: dict[str, dict[str, SymbolAlias]] = {self.root_scope_id: {}}
        self._mentions_by_symbol_id: dict[str, list[str]] = {}
        self._canonical_key_to_symbol_id: dict[str, str] = {}
        self._family_key_to_symbol_id: dict[str, str] = {}

        self._indexed_pattern = re.compile(PARSING.INDEXED_SYMBOL_PATTERN)
        self._simple_symbol_pattern = re.compile(rf"^{PARSING.SIMPLE_SYMBOL_PATTERN}$")

    def ensure_scope(
        self,
        scope_id: str | None = None,
        *,
        kind: str = "local",
        parent_scope_id: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if scope_id is None:
            scope_id = _stable_digest(
                f"scope::{self.problem_namespace}::{parent_scope_id or self.root_scope_id}::{kind}::{label or ''}",
                "scope",
            )
        if scope_id in self._scopes:
            return scope_id

        parent = parent_scope_id or self.root_scope_id
        if parent not in self._scopes:
            raise KeyError(f"Unknown parent scope_id={parent!r}")

        self._scopes[scope_id] = ScopeRecord(
            scope_id=scope_id,
            kind=kind,
            parent_scope_id=parent,
            label=label,
            metadata=dict(metadata or {}),
        )
        self._bindings_by_scope.setdefault(scope_id, {})
        self._aliases_by_scope.setdefault(scope_id, {})
        return scope_id

    def get_scope(self, scope_id: str) -> ScopeRecord:
        return self._scopes[scope_id]

    def iter_scope_chain(self, scope_id: str) -> list[str]:
        if scope_id not in self._scopes:
            raise KeyError(f"Unknown scope_id={scope_id!r}")
        out: list[str] = []
        cur: str | None = scope_id
        while cur is not None:
            out.append(cur)
            cur = self._scopes[cur].parent_scope_id
        return out

    def normalize_symbol_text(self, text: str) -> str:
        normalized = normalize_math_text(text or "")
        normalized = normalized.strip()
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    def parse_indexed_symbol(self, text: str) -> IndexedSymbolSpec | None:
        normalized = self.normalize_symbol_text(text)
        match = self._indexed_pattern.match(normalized)
        if not match:
            return None
        index_text = match.group("index")
        tokens = [
            token
            for token in re.split(r"([^A-Za-z0-9]+)", index_text)
            if token and not token.isspace()
        ]
        return IndexedSymbolSpec(
            base_name=match.group("base"),
            index_text=index_text,
            index_tokens=tokens,
        )

    def register_symbol(
        self,
        mention_text: str,
        *,
        scope_id: str | None = None,
        declaration_scope_id: str | None = None,
        role: SymbolRole = SymbolRole.VARIABLE,
        canonical_name: str | None = None,
        display_name: str | None = None,
        domain: str | None = None,
        qualifiers: list[str] | None = None,
        assumptions: list[str] | None = None,
        annotations: dict[str, str] | None = None,
        source: str | None = None,
    ) -> SymbolRef:
        scope = self.ensure_scope(scope_id or self.root_scope_id)
        decl_scope = self.ensure_scope(declaration_scope_id or scope)

        normalized = self.normalize_symbol_text(mention_text)
        if not normalized:
            raise ValueError("mention_text normalizes to empty text")

        indexed = self.parse_indexed_symbol(normalized)
        base_name = indexed.base_name if indexed else normalized
        family_symbol_id = None
        if indexed is not None:
            family_symbol_id = self._ensure_family_symbol(
                base_name=base_name,
                declaration_scope_id=decl_scope,
                role=role,
            )

        resolved_canonical = canonical_name or (
            f"{indexed.base_name}__idx__{indexed.index_text}" if indexed else base_name
        )

        key = self._make_canonical_key(
            declaration_scope_id=decl_scope,
            role=role,
            canonical_name=resolved_canonical,
            indexed=indexed,
        )

        if key in self._canonical_key_to_symbol_id:
            symbol_id = self._canonical_key_to_symbol_id[key]
            existing = self._symbols_by_id[symbol_id]
            updated = existing.model_copy(
                update={
                    "display_name": display_name or existing.display_name,
                    "domain": DomainSpec(
                        domain=domain or existing.domain.domain,
                        qualifiers=sorted(set(existing.domain.qualifiers).union(qualifiers or [])),
                        assumptions=sorted(set(existing.domain.assumptions).union(assumptions or [])),
                    ),
                    "annotations": {**existing.annotations, **(annotations or {})},
                    "mention_count": existing.mention_count + 1,
                }
            )
            self._symbols_by_id[symbol_id] = updated
            self._bindings_by_scope.setdefault(scope, {})[normalized] = symbol_id
            self._mentions_by_symbol_id.setdefault(symbol_id, []).append(normalized)
            return updated

        symbol_id = _stable_digest(f"symbol::{self.problem_namespace}::{key}", "sym")
        symbol = SymbolRef(
            symbol_id=symbol_id,
            canonical_name=resolved_canonical,
            display_name=display_name or mention_text,
            base_name=base_name,
            role=role,
            declaration_scope_id=decl_scope,
            family_symbol_id=family_symbol_id,
            indexed=indexed,
            domain=DomainSpec(
                domain=domain,
                qualifiers=sorted(set(qualifiers or [])),
                assumptions=sorted(set(assumptions or [])),
            ),
            annotations=dict(sorted((annotations or {}).items())),
            symmetry_group_ids=[],
            aliases=[],
            mention_count=1,
        )
        self._symbols_by_id[symbol_id] = symbol
        self._canonical_key_to_symbol_id[key] = symbol_id
        self._bindings_by_scope.setdefault(scope, {})[normalized] = symbol_id
        self._bindings_by_scope.setdefault(decl_scope, {})[normalized] = symbol_id
        self._mentions_by_symbol_id.setdefault(symbol_id, []).append(normalized)
        return symbol

    def register_alias(
        self,
        alias_text: str,
        target_symbol_id: str,
        *,
        scope_id: str | None = None,
        source: str | None = None,
        is_branch_local: bool = False,
    ) -> SymbolAlias:
        if target_symbol_id not in self._symbols_by_id:
            raise KeyError(f"Unknown target_symbol_id={target_symbol_id!r}")

        scope = self.ensure_scope(scope_id or self.root_scope_id)
        normalized_alias = self.normalize_symbol_text(alias_text)
        alias = SymbolAlias(
            alias_text=alias_text,
            normalized_alias=normalized_alias,
            scope_id=scope,
            source=source,
            is_branch_local=is_branch_local,
        )
        self._aliases_by_scope.setdefault(scope, {})[normalized_alias] = alias
        self._bindings_by_scope.setdefault(scope, {})[normalized_alias] = target_symbol_id

        existing = self._symbols_by_id[target_symbol_id]
        updated_aliases = list(existing.aliases)
        updated_aliases.append(alias)
        self._symbols_by_id[target_symbol_id] = existing.model_copy(update={"aliases": updated_aliases})
        return alias

    def resolve(self, query_text: str, *, scope_id: str | None = None) -> SymbolResolution:
        scope = scope_id or self.root_scope_id
        normalized = self.normalize_symbol_text(query_text)
        if not normalized:
            return SymbolResolution(query_text, normalized, scope, None, None, "none")

        for chain_scope in self.iter_scope_chain(scope):
            bound = self._bindings_by_scope.get(chain_scope, {}).get(normalized)
            if bound is not None:
                symbol = self._symbols_by_id[bound]
                via = "alias" if normalized in self._aliases_by_scope.get(chain_scope, {}) else "canonical"
                return SymbolResolution(query_text, normalized, scope, symbol.symbol_id, symbol.canonical_name, via)

        indexed = self.parse_indexed_symbol(normalized)
        if indexed is not None:
            family_key = self._make_family_key(scope, SymbolRole.VARIABLE, indexed.base_name)
            family_symbol_id = self._family_key_to_symbol_id.get(family_key)
            if family_symbol_id is not None:
                family = self._symbols_by_id[family_symbol_id]
                return SymbolResolution(query_text, normalized, scope, family.symbol_id, family.canonical_name, "family")

        return SymbolResolution(query_text, normalized, scope, None, None, "none")

    def canonicalize_symbol_mentions(self, text: str, *, scope_id: str | None = None) -> str:
        scope = scope_id or self.root_scope_id

        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            resolved = self.resolve(token, scope_id=scope)
            if resolved.symbol_id is None:
                return token
            return self._symbols_by_id[resolved.symbol_id].canonical_name

        pattern = re.compile(rf"{PARSING.SIMPLE_SYMBOL_PATTERN}")
        return pattern.sub(repl, text)

    def get_symbol(self, symbol_id: str) -> SymbolRef:
        return self._symbols_by_id[symbol_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_namespace": self.problem_namespace,
            "root_scope_id": self.root_scope_id,
            "scopes": [scope.__dict__ for scope in self._scopes.values()],
            "symbols": [symbol.model_dump(mode="json") for symbol in self._symbols_by_id.values()],
            "bindings_by_scope": self._bindings_by_scope,
            "aliases_by_scope": {
                scope_id: {alias: item.model_dump(mode="json") for alias, item in entries.items()}
                for scope_id, entries in self._aliases_by_scope.items()
            },
            "mentions_by_symbol_id": self._mentions_by_symbol_id,
            "canonical_key_to_symbol_id": self._canonical_key_to_symbol_id,
            "family_key_to_symbol_id": self._family_key_to_symbol_id,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SymbolTable":
        obj = cls(problem_namespace=payload["problem_namespace"])
        obj.root_scope_id = payload["root_scope_id"]
        obj._scopes = {row["scope_id"]: ScopeRecord(**row) for row in payload["scopes"]}
        obj._symbols_by_id = {row["symbol_id"]: SymbolRef.model_validate(row) for row in payload["symbols"]}
        obj._bindings_by_scope = {k: dict(v) for k, v in payload["bindings_by_scope"].items()}
        obj._aliases_by_scope = {
            scope_id: {alias: SymbolAlias.model_validate(item) for alias, item in entries.items()}
            for scope_id, entries in payload["aliases_by_scope"].items()
        }
        obj._mentions_by_symbol_id = {k: list(v) for k, v in payload["mentions_by_symbol_id"].items()}
        obj._canonical_key_to_symbol_id = dict(payload["canonical_key_to_symbol_id"])
        obj._family_key_to_symbol_id = dict(payload["family_key_to_symbol_id"])
        return obj

    @classmethod
    def from_json(cls, payload: str) -> "SymbolTable":
        return cls.from_dict(json.loads(payload))

    def _ensure_family_symbol(self, *, base_name: str, declaration_scope_id: str, role: SymbolRole) -> str:
        family_key = self._make_family_key(declaration_scope_id, role, base_name)
        if family_key in self._family_key_to_symbol_id:
            return self._family_key_to_symbol_id[family_key]

        symbol_id = _stable_digest(f"symbol::{self.problem_namespace}::{family_key}", "sym")
        family_symbol = SymbolRef(
            symbol_id=symbol_id,
            canonical_name=base_name,
            display_name=base_name,
            base_name=base_name,
            role=role,
            declaration_scope_id=declaration_scope_id,
            family_symbol_id=None,
            indexed=None,
            domain=DomainSpec(),
            annotations={"family_root": "true"},
            symmetry_group_ids=[],
            aliases=[],
            mention_count=0,
        )
        self._symbols_by_id[symbol_id] = family_symbol
        self._family_key_to_symbol_id[family_key] = symbol_id
        return symbol_id

    def _make_family_key(self, declaration_scope_id: str, role: SymbolRole, base_name: str) -> str:
        return f"family::{declaration_scope_id}::{role.value}::{base_name}"

    def _make_canonical_key(
        self,
        *,
        declaration_scope_id: str,
        role: SymbolRole,
        canonical_name: str,
        indexed: IndexedSymbolSpec | None,
    ) -> str:
        index_part = indexed.index_text if indexed is not None else "-"
        return f"decl::{declaration_scope_id}::role::{role.value}::canon::{canonical_name}::index::{index_part}"

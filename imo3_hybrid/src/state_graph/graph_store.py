from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Literal

from .node import MergeCompatibility, ReasoningStateNode
from .edge import ReasoningStateEdge


GraphInsertStatus = Literal["inserted", "merged", "duplicate", "suppressed"]
GraphEdgeInsertStatus = Literal["inserted", "duplicate", "rewritten", "suppressed"]


@dataclass(frozen=True)
class GraphInsertOutcome:
    status: GraphInsertStatus
    canonical_node_id: str
    input_node_id: str
    inserted_node_id: str | None = None
    duplicate_of_node_id: str | None = None
    alias_created: bool = False
    compatibility: MergeCompatibility | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GraphEdgeInsertOutcome:
    status: GraphEdgeInsertStatus
    edge_id: str | None
    source_node_id: str
    target_node_id: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FrontierCandidate:
    node_id: str
    depth: int
    confidence: float
    verifier_score: float
    symbolic_valid: bool
    merge_ready: bool
    open_goal_count: int
    open_obligation_count: int
    child_count: int
    score: float


@dataclass(frozen=True)
class GraphSnapshot:
    root_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    alias_map: Mapping[str, str]
    canonical_hash_index: Mapping[str, tuple[str, ...]]


class StateGraphStore:
    def __init__(self) -> None:
        self._nodes: dict[str, ReasoningStateNode] = {}
        self._edges: dict[str, ReasoningStateEdge] = {}

        self._node_aliases: dict[str, str] = {}
        self._canonical_hash_index: dict[str, list[str]] = {}

        self._children_by_parent: dict[str, list[str]] = {}
        self._parents_by_child: dict[str, list[str]] = {}

        self._edge_ids_by_source: dict[str, list[str]] = {}
        self._edge_ids_by_target: dict[str, list[str]] = {}
        self._edge_dedup_index: dict[tuple[str, str, str, str], str] = {}

        self._root_node_ids: list[str] = []
        self._node_insertion_order: list[str] = []
        self._edge_insertion_order: list[str] = []

    def canonical_node_id(self, node_id: str) -> str:
        seen: set[str] = set()
        current = node_id
        while current in self._node_aliases and current not in seen:
            seen.add(current)
            current = self._node_aliases[current]
        return current

    def has_node(self, node_id: str) -> bool:
        return self.canonical_node_id(node_id) in self._nodes

    def get_node(self, node_id: str) -> ReasoningStateNode | None:
        return self._nodes.get(self.canonical_node_id(node_id))

    def get_edge(self, edge_id: str) -> ReasoningStateEdge | None:
        return self._edges.get(edge_id)

    def iter_nodes(self):
        for node_id in self._node_insertion_order:
            if node_id in self._nodes:
                yield self._nodes[node_id]

    def iter_edges(self):
        for edge_id in self._edge_insertion_order:
            if edge_id in self._edges:
                yield self._edges[edge_id]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def root_node_ids(self) -> tuple[str, ...]:
        return tuple(self._root_node_ids)

    @property
    def alias_map(self) -> Mapping[str, str]:
        return dict(self._node_aliases)

    @property
    def canonical_hash_index(self) -> Mapping[str, tuple[str, ...]]:
        return {k: tuple(v) for k, v in self._canonical_hash_index.items()}

    def insert_node(
        self,
        node: ReasoningStateNode,
        *,
        merge_equivalent: bool = True,
        suppress_dominated_duplicates: bool = True,
    ) -> GraphInsertOutcome:
        input_node_id = node.node_id

        if input_node_id in self._nodes:
            return GraphInsertOutcome(
                status="duplicate",
                canonical_node_id=input_node_id,
                input_node_id=input_node_id,
                duplicate_of_node_id=input_node_id,
                reasons=("node_id_exists",),
            )

        if input_node_id in self._node_aliases:
            canonical_id = self.canonical_node_id(input_node_id)
            return GraphInsertOutcome(
                status="duplicate",
                canonical_node_id=canonical_id,
                input_node_id=input_node_id,
                duplicate_of_node_id=canonical_id,
                reasons=("node_id_aliased",),
            )

        if merge_equivalent:
            for candidate_id in self._canonical_hash_index.get(node.state_fingerprint, []):
                existing = self._nodes[candidate_id]
                compatibility = existing.merge_compatibility(node)

                if compatibility.equivalent:
                    self._nodes[candidate_id] = self._merge_nodes(existing, node, compatibility=compatibility)
                    self._node_aliases[input_node_id] = candidate_id
                    return GraphInsertOutcome(
                        status="merged",
                        canonical_node_id=candidate_id,
                        input_node_id=input_node_id,
                        duplicate_of_node_id=candidate_id,
                        alias_created=True,
                        compatibility=compatibility,
                        reasons=("equivalent_state",) + compatibility.reasons,
                    )

                if (
                    suppress_dominated_duplicates
                    and compatibility.compatible
                    and compatibility.dominance in {"self", "equal"}
                ):
                    self._nodes[candidate_id] = self._merge_nodes(existing, node, compatibility=compatibility)
                    self._node_aliases[input_node_id] = candidate_id
                    return GraphInsertOutcome(
                        status="suppressed",
                        canonical_node_id=candidate_id,
                        input_node_id=input_node_id,
                        duplicate_of_node_id=candidate_id,
                        alias_created=True,
                        compatibility=compatibility,
                        reasons=("dominated_duplicate",) + compatibility.reasons,
                    )

        self._store_new_node(node)
        return GraphInsertOutcome(
            status="inserted",
            canonical_node_id=node.node_id,
            input_node_id=input_node_id,
            inserted_node_id=node.node_id,
            reasons=("new_node",),
        )

    def _store_new_node(self, node: ReasoningStateNode) -> None:
        self._nodes[node.node_id] = node
        self._node_insertion_order.append(node.node_id)
        self._canonical_hash_index.setdefault(node.state_fingerprint, []).append(node.node_id)

        if node.provenance.depth == 0 and not node.provenance.parent_node_ids:
            self._append_unique(self._root_node_ids, node.node_id)

        for parent_id in node.provenance.parent_node_ids:
            canonical_parent = self.canonical_node_id(parent_id)
            self._append_unique(self._parents_by_child.setdefault(node.node_id, []), canonical_parent)
            self._append_unique(self._children_by_parent.setdefault(canonical_parent, []), node.node_id)

    def replace_node(self, node: ReasoningStateNode) -> ReasoningStateNode:
        canonical_id = self.canonical_node_id(node.node_id)
        existing = self._nodes.get(canonical_id)
        if existing is None:
            self._store_new_node(node)
            return node

        if existing.state_fingerprint != node.state_fingerprint:
            prior = self._canonical_hash_index.get(existing.state_fingerprint, [])
            self._canonical_hash_index[existing.state_fingerprint] = [x for x in prior if x != canonical_id]
            if not self._canonical_hash_index[existing.state_fingerprint]:
                del self._canonical_hash_index[existing.state_fingerprint]
            self._canonical_hash_index.setdefault(node.state_fingerprint, []).append(canonical_id)

        self._nodes[canonical_id] = node
        return node

    def insert_edge(
        self,
        edge: ReasoningStateEdge,
        *,
        suppress_self_loops: bool = True,
    ) -> GraphEdgeInsertOutcome:
        source_id = self.canonical_node_id(edge.source_node_id)
        target_id = self.canonical_node_id(edge.target_node_id)

        if suppress_self_loops and source_id == target_id:
            return GraphEdgeInsertOutcome(
                status="suppressed",
                edge_id=None,
                source_node_id=source_id,
                target_node_id=target_id,
                reasons=("self_loop",),
            )

        rewritten = edge.with_updates(source_node_id=source_id, target_node_id=target_id)
        dedup_key = (
            source_id,
            target_id,
            rewritten.edge_type.value,
            rewritten.provenance.operator_name or "",
        )

        if dedup_key in self._edge_dedup_index:
            existing_edge_id = self._edge_dedup_index[dedup_key]
            return GraphEdgeInsertOutcome(
                status="duplicate",
                edge_id=existing_edge_id,
                source_node_id=source_id,
                target_node_id=target_id,
                reasons=("edge_exists",),
            )

        self._edges[rewritten.edge_id] = rewritten
        self._edge_insertion_order.append(rewritten.edge_id)
        self._edge_dedup_index[dedup_key] = rewritten.edge_id

        self._append_unique(self._edge_ids_by_source.setdefault(source_id, []), rewritten.edge_id)
        self._append_unique(self._edge_ids_by_target.setdefault(target_id, []), rewritten.edge_id)
        self._append_unique(self._children_by_parent.setdefault(source_id, []), target_id)
        self._append_unique(self._parents_by_child.setdefault(target_id, []), source_id)

        if target_id in self._root_node_ids:
            self._root_node_ids = [rid for rid in self._root_node_ids if rid != target_id]

        return GraphEdgeInsertOutcome(
            status="rewritten" if (edge.source_node_id != source_id or edge.target_node_id != target_id) else "inserted",
            edge_id=rewritten.edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            reasons=("edge_added",),
        )

    def get_children(self, node_id: str) -> tuple[ReasoningStateNode, ...]:
        canonical_id = self.canonical_node_id(node_id)
        return tuple(self._nodes[cid] for cid in self._children_by_parent.get(canonical_id, []) if cid in self._nodes)

    def get_parents(self, node_id: str) -> tuple[ReasoningStateNode, ...]:
        canonical_id = self.canonical_node_id(node_id)
        return tuple(self._nodes[pid] for pid in self._parents_by_child.get(canonical_id, []) if pid in self._nodes)

    def get_outgoing_edges(self, node_id: str) -> tuple[ReasoningStateEdge, ...]:
        canonical_id = self.canonical_node_id(node_id)
        return tuple(self._edges[eid] for eid in self._edge_ids_by_source.get(canonical_id, []) if eid in self._edges)

    def get_incoming_edges(self, node_id: str) -> tuple[ReasoningStateEdge, ...]:
        canonical_id = self.canonical_node_id(node_id)
        return tuple(self._edges[eid] for eid in self._edge_ids_by_target.get(canonical_id, []) if eid in self._edges)

    def is_leaf(self, node_id: str) -> bool:
        return len(self._children_by_parent.get(self.canonical_node_id(node_id), [])) == 0

    def leaf_nodes(self, *, include_terminal: bool = False) -> tuple[ReasoningStateNode, ...]:
        out: list[ReasoningStateNode] = []
        for node in self.iter_nodes():
            if self.is_leaf(node.node_id):
                if include_terminal or not node.is_terminal:
                    out.append(node)
        return tuple(out)

    def frontier_candidates(
        self,
        *,
        include_terminal: bool = False,
        limit: int | None = None,
        min_confidence: float = 0.0,
    ) -> tuple[FrontierCandidate, ...]:
        candidates: list[FrontierCandidate] = []
        for node in self.iter_nodes():
            if not include_terminal and node.is_terminal:
                continue
            if node.confidence < min_confidence:
                continue
            if not self.is_leaf(node.node_id):
                continue
            open_goals = len(node.remaining_open_goals())
            open_obligations = len(node.open_proof_obligations())
            child_count = len(self._children_by_parent.get(node.node_id, []))
            score = (
                0.45 * node.verifier_score
                + 0.30 * node.confidence
                + 0.15 * (1.0 if node.symbolic_valid else 0.0)
                + 0.05 * (1.0 if node.merge_ready else 0.0)
                + 0.03 * max(0.0, 1.0 - min(open_goals, 10) / 10.0)
                + 0.015 * max(0.0, 1.0 - min(open_obligations, 10) / 10.0)
                + 0.005 * max(0.0, 1.0 - min(child_count, 10) / 10.0)
            )
            candidates.append(
                FrontierCandidate(
                    node_id=node.node_id,
                    depth=node.provenance.depth,
                    confidence=node.confidence,
                    verifier_score=node.verifier_score,
                    symbolic_valid=node.symbolic_valid,
                    merge_ready=node.merge_ready,
                    open_goal_count=open_goals,
                    open_obligation_count=open_obligations,
                    child_count=child_count,
                    score=score,
                )
            )

        ordered = sorted(
            candidates,
            key=lambda c: (
                -c.score,
                c.open_goal_count,
                c.open_obligation_count,
                c.depth,
                self._node_insertion_order.index(c.node_id),
            ),
        )
        if limit is not None:
            ordered = ordered[:limit]
        return tuple(ordered)

    def best_frontier_nodes(
        self,
        *,
        include_terminal: bool = False,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> tuple[ReasoningStateNode, ...]:
        return tuple(
            self._nodes[item.node_id]
            for item in self.frontier_candidates(
                include_terminal=include_terminal,
                limit=limit,
                min_confidence=min_confidence,
            )
            if item.node_id in self._nodes
        )

    def lineage_node_ids(self, node_id: str) -> tuple[str, ...]:
        node = self.get_node(node_id)
        if node is None:
            return tuple()
        if node.provenance.lineage:
            return tuple(self.canonical_node_id(x) for x in node.provenance.lineage)
        return self.path_to_root_ids(node_id)

    def path_to_root_ids(self, node_id: str) -> tuple[str, ...]:
        canonical_id = self.canonical_node_id(node_id)
        if canonical_id not in self._nodes:
            return tuple()

        path: list[str] = []
        current = canonical_id
        seen: set[str] = set()
        while current in self._nodes and current not in seen:
            seen.add(current)
            path.append(current)
            parents = self._parents_by_child.get(current, [])
            if not parents:
                break
            current = parents[0]
        path.reverse()
        return tuple(path)

    def path_to_root(self, node_id: str) -> tuple[ReasoningStateNode, ...]:
        return tuple(self._nodes[nid] for nid in self.path_to_root_ids(node_id) if nid in self._nodes)

    def descendants(self, node_id: str) -> tuple[ReasoningStateNode, ...]:
        canonical_id = self.canonical_node_id(node_id)
        out: list[ReasoningStateNode] = []
        queue: list[str] = list(self._children_by_parent.get(canonical_id, []))
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen or current not in self._nodes:
                continue
            seen.add(current)
            out.append(self._nodes[current])
            queue.extend(self._children_by_parent.get(current, []))
        return tuple(out)

    def ancestors(self, node_id: str) -> tuple[ReasoningStateNode, ...]:
        canonical_id = self.canonical_node_id(node_id)
        out: list[ReasoningStateNode] = []
        queue: list[str] = list(self._parents_by_child.get(canonical_id, []))
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen or current not in self._nodes:
                continue
            seen.add(current)
            out.append(self._nodes[current])
            queue.extend(self._parents_by_child.get(current, []))
        return tuple(out)

    def equivalent_nodes(self, node: ReasoningStateNode) -> tuple[ReasoningStateNode, ...]:
        return tuple(self._nodes[nid] for nid in self._canonical_hash_index.get(node.state_fingerprint, []) if nid in self._nodes)

    def dominance_candidates(self, node: ReasoningStateNode) -> tuple[ReasoningStateNode, ...]:
        out: list[ReasoningStateNode] = []
        for nid in self._canonical_hash_index.get(node.state_fingerprint, []):
            if nid == node.node_id or nid not in self._nodes:
                continue
            other = self._nodes[nid]
            comp = other.merge_compatibility(node)
            if comp.compatible and comp.dominance in {"self", "equal", "other"}:
                out.append(other)
        return tuple(out)

    def register_transposition(self, duplicate_node_id: str, canonical_node_id: str) -> None:
        self._node_aliases[duplicate_node_id] = self.canonical_node_id(canonical_node_id)

    def snapshot(self) -> GraphSnapshot:
        return GraphSnapshot(
            root_node_ids=tuple(self._root_node_ids),
            node_ids=tuple(self._node_insertion_order),
            edge_ids=tuple(self._edge_insertion_order),
            alias_map=dict(self._node_aliases),
            canonical_hash_index={k: tuple(v) for k, v in self._canonical_hash_index.items()},
        )

    def export_graph(self) -> Mapping[str, object]:
        return {
            "roots": list(self._root_node_ids),
            "nodes": [node.export_state() for node in self.iter_nodes()],
            "edges": [edge.export_state() for edge in self.iter_edges()],
            "aliases": dict(self._node_aliases),
            "canonical_hash_index": {k: list(v) for k, v in self._canonical_hash_index.items()},
        }

    def reset(self) -> None:
        self.__init__()

    def _merge_nodes(
        self,
        existing: ReasoningStateNode,
        incoming: ReasoningStateNode,
        *,
        compatibility: MergeCompatibility | None = None,
    ) -> ReasoningStateNode:
        prefer_incoming = compatibility is not None and compatibility.dominance == "other"

        merged_constraints = existing.constraints + tuple(c for c in incoming.constraints if c not in existing.constraints)
        merged_invariants = existing.invariants + tuple(i for i in incoming.invariants if i not in existing.invariants)
        merged_goals = existing.goals + tuple(g for g in incoming.goals if g not in existing.goals)
        merged_proof_obligations = existing.proof_obligations + tuple(
            item for item in incoming.proof_obligations if item not in existing.proof_obligations
        )
        merged_history = existing.operator_history + tuple(op for op in incoming.operator_history if op not in existing.operator_history)
        merged_evidence = existing.evidence + tuple(ev for ev in incoming.evidence if ev not in existing.evidence)

        merged_parent_ids = tuple(sorted(set(existing.provenance.parent_node_ids) | set(incoming.provenance.parent_node_ids)))
        merged_from = tuple(sorted(set(existing.provenance.merged_from_node_ids) | set(incoming.provenance.merged_from_node_ids) | {incoming.node_id}))
        merged_tags = tuple(sorted(set(existing.provenance.source_tags) | set(incoming.provenance.source_tags)))

        merged_lineage = existing.provenance.lineage
        if incoming.provenance.lineage and (not merged_lineage or len(incoming.provenance.lineage) < len(merged_lineage)):
            merged_lineage = incoming.provenance.lineage

        partial_solution = existing.partial_solution
        if prefer_incoming and incoming.partial_solution:
            partial_solution = incoming.partial_solution
        elif incoming.partial_solution and len(incoming.partial_solution) > len(partial_solution):
            partial_solution = incoming.partial_solution

        extracted_objects = dict(existing.extracted_objects)
        for key in sorted(incoming.extracted_objects.keys()):
            if key not in extracted_objects or prefer_incoming:
                extracted_objects[key] = incoming.extracted_objects[key]

        merged_metadata = dict(existing.metadata)
        for key in sorted(incoming.metadata.keys()):
            if key not in merged_metadata or prefer_incoming:
                merged_metadata[key] = incoming.metadata[key]

        return existing.with_updates(
            constraints=merged_constraints,
            invariants=merged_invariants,
            goals=merged_goals,
            proof_obligations=merged_proof_obligations,
            extracted_objects=extracted_objects,
            partial_solution=partial_solution,
            summary_text=incoming.summary_text if prefer_incoming and incoming.summary_text else existing.summary_text,
            operator_history=merged_history,
            evidence=merged_evidence,
            confidence=max(existing.confidence, incoming.confidence),
            verifier_score=max(existing.verifier_score, incoming.verifier_score),
            symbolic_valid=existing.symbolic_valid or incoming.symbolic_valid,
            merge_ready=existing.merge_ready or incoming.merge_ready,
            is_terminal=existing.is_terminal or incoming.is_terminal,
            parent_node_ids=merged_parent_ids,
            merged_from_node_ids=merged_from,
            repair_from_node_id=existing.provenance.repair_from_node_id or incoming.provenance.repair_from_node_id,
            source_tags=merged_tags,
            depth=min(existing.provenance.depth, incoming.provenance.depth),
            lineage=merged_lineage,
            metadata=merged_metadata,
        )

    @staticmethod
    def _append_unique(target: list[str], value: str) -> None:
        if value not in target:
            target.append(value)


__all__ = [
    "GraphInsertOutcome",
    "GraphEdgeInsertOutcome",
    "FrontierCandidate",
    "GraphSnapshot",
    "StateGraphStore",
]

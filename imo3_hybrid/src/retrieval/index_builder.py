from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.common.logging import get_logger
from src.common.schemas import BranchTrace, ParsedProblem, RetrievedTrace
from src.retrieval.embedder import MathEmbedder, _safe_normalize_text, _l2_normalize


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
logger = get_logger("imo3_hybrid.retrieval", module=__name__, component="index_builder")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(_safe_normalize_text(text).lower())


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class IndexedTraceRecord:
    trace_id: str
    problem_id: str
    problem_text: str
    solution_text: str
    answer: str = ""
    domain: str = "mixed"
    difficulty: str = "medium"
    archetypes: list[str] = field(default_factory=list)
    operators_used: list[str] = field(default_factory=list)
    source: str = "unknown"
    tags: list[str] = field(default_factory=list)
    subproblem_snippets: list[str] = field(default_factory=list)
    repair_snippets: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    proof_obligation_hints: list[str] = field(default_factory=list)
    repair_neighbors: list[str] = field(default_factory=list)
    continuation_operators: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def composite_text(self) -> str:
        tag_text = " ".join(self.tags)
        arch_text = " ".join(self.archetypes)
        op_text = " ".join(self.operators_used)
        return "\n".join(
            [
                f"domain: {self.domain}",
                f"difficulty: {self.difficulty}",
                f"archetypes: {arch_text}",
                f"operators: {op_text}",
                f"tags: {tag_text}",
                f"failure_modes: {' '.join(self.failure_modes)}",
                f"proof_obligations: {' '.join(self.proof_obligation_hints)}",
                f"repair_neighbors: {' '.join(self.repair_neighbors)}",
                f"problem: {self.problem_text}",
                f"solution: {self.solution_text}",
            ]
        ).strip()

    def lexical_text(self) -> str:
        return " ".join(
            [
                self.problem_text,
                self.solution_text[:1000],
                " ".join(self.archetypes),
                " ".join(self.operators_used),
                " ".join(self.tags),
                " ".join(self.subproblem_snippets[:6]),
                " ".join(self.repair_snippets[:4]),
                " ".join(self.failure_modes),
                " ".join(self.proof_obligation_hints[:4]),
                " ".join(self.repair_neighbors[:4]),
                " ".join(self.continuation_operators[:4]),
            ]
        ).strip()


@dataclass
class RetrievalArtifactMetadata:
    version: str
    built_from_count: int
    embedding_backend: str
    embedding_dimension: int
    artifact_type: str = "hybrid_retrieval_index"
    metadata: dict[str, object] = field(default_factory=dict)


class RetrievalIndexBuilder:
    """
    Hybrid retrieval artifact:
    - dense embeddings
    - deterministic lexical token store
    - structural metadata/tags
    Optional FAISS backing; safe numpy fallback.
    """

    def __init__(
        self,
        embedder: MathEmbedder,
        index_path: str = "data/interim/retrieval_index",
        ef_search: int = 64,
    ) -> None:
        self.embedder = embedder
        self.index_path = Path(index_path)
        self.ef_search = ef_search
        self._faiss_index = None
        self._dense_matrix: Optional[np.ndarray] = None
        self._records: list[IndexedTraceRecord] = []
        self._token_sets: list[set[str]] = []
        self._artifact_meta: Optional[RetrievalArtifactMetadata] = None

    @property
    def is_loaded(self) -> bool:
        return bool(self._records) and (self._dense_matrix is not None or self._faiss_index is not None)

    @property
    def records(self) -> list[IndexedTraceRecord]:
        return list(self._records)

    @property
    def artifact_metadata(self) -> Optional[RetrievalArtifactMetadata]:
        if self._artifact_meta is None:
            return None
        return RetrievalArtifactMetadata(**asdict(self._artifact_meta))

    def build_from_records(
        self,
        records: list[IndexedTraceRecord],
        *,
        save: bool = True,
    ) -> None:
        ordered = sorted(records, key=lambda r: (r.problem_id, r.trace_id, r.source))
        texts = [r.composite_text() for r in ordered]
        dense = self.embedder.embed(texts, normalize=True).astype(np.float32)

        self._records = ordered
        self._dense_matrix = dense
        self._token_sets = [set(_tokenize(r.lexical_text())) for r in ordered]
        self._artifact_meta = RetrievalArtifactMetadata(
            version="v2",
            built_from_count=len(ordered),
            embedding_backend=self.embedder.backend,
            embedding_dimension=int(dense.shape[1]) if len(dense) else self.embedder.dimension(),
            metadata={
                "deterministic_ordering": True,
                "record_fingerprint": self._record_fingerprint(ordered),
                "source_breakdown": self._source_breakdown(ordered),
            },
        )

        self._maybe_build_faiss(dense)

        if save:
            self.save()

        logger.info(
            "retrieval_index_built",
            message=(
                f"Built retrieval artifact with {len(ordered)} records "
                f"(backend={self.embedder.backend}, dim={self._artifact_meta.embedding_dimension})"
            ),
            payload={
                "record_count": len(ordered),
                "embedding_backend": self.embedder.backend,
                "embedding_dimension": self._artifact_meta.embedding_dimension,
                "index_path": str(self.index_path),
            },
        )

    def build(
        self,
        texts: list[str],
        metadata: list[dict],
        save: bool = True,
    ) -> None:
        if len(texts) != len(metadata):
            raise ValueError(
                f"texts/metadata length mismatch: {len(texts)} texts vs {len(metadata)} metadata rows"
            )
        records = []
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            records.append(
                IndexedTraceRecord(
                    trace_id=str(meta.get("trace_id", f"trace_{i}")),
                    problem_id=str(meta.get("problem_id", meta.get("trace_id", f"problem_{i}"))),
                    problem_text=str(meta.get("problem", text)),
                    solution_text=str(meta.get("solution", "")),
                    answer=str(meta.get("answer", "")),
                    domain=str(meta.get("domain", "mixed")),
                    difficulty=str(meta.get("difficulty", "medium")),
                    archetypes=list(meta.get("archetypes", [])),
                    operators_used=list(meta.get("operators_used", [])),
                    source=str(meta.get("source", "unknown")),
                    tags=list(meta.get("tags", [])),
                    subproblem_snippets=list(meta.get("subproblem_snippets", [])),
                    repair_snippets=list(meta.get("repair_snippets", [])),
                    failure_modes=list(meta.get("failure_modes", [])),
                    proof_obligation_hints=list(meta.get("proof_obligation_hints", [])),
                    repair_neighbors=list(meta.get("repair_neighbors", [])),
                    continuation_operators=list(meta.get("continuation_operators", [])),
                    metadata={k: v for k, v in meta.items() if k not in {
                        "trace_id", "problem_id", "problem", "solution", "answer",
                        "domain", "difficulty", "archetypes", "operators_used",
                        "source", "tags", "subproblem_snippets", "repair_snippets",
                        "failure_modes", "proof_obligation_hints", "repair_neighbors",
                        "continuation_operators",
                    }},
                )
            )
        self.build_from_records(records, save=save)

    def build_from_branch_traces(
        self,
        traces: list[BranchTrace],
        problem_lookup: Optional[dict[str, ParsedProblem]] = None,
        *,
        save: bool = True,
    ) -> None:
        records: list[IndexedTraceRecord] = []
        for trace in traces:
            problem = problem_lookup.get(trace.problem_id) if problem_lookup else None
            records.append(
                IndexedTraceRecord(
                    trace_id=trace.branch_id,
                    problem_id=trace.problem_id,
                    problem_text=problem.raw_text if problem else "",
                    solution_text=trace.full_reasoning,
                    answer=str(trace.answer or ""),
                    domain=problem.domain.value if problem else "mixed",
                    difficulty="medium",
                    archetypes=[trace.archetype_used] if trace.archetype_used else [],
                    operators_used=list(trace.operator_sequence),
                    source="branch_trace",
                    tags=self._derive_tags(problem, trace),
                    subproblem_snippets=self._derive_subproblem_snippets(problem, trace),
                    repair_snippets=self._derive_repair_snippets(trace),
                    failure_modes=self._derive_failure_modes(trace),
                    proof_obligation_hints=self._derive_proof_obligation_hints(trace),
                    repair_neighbors=self._derive_repair_neighbors(trace),
                    continuation_operators=self._derive_continuation_operators(trace),
                    metadata={
                        "branch_score": trace.branch_score,
                        "verifier_score": trace.verifier_score,
                        "symbolic_valid": trace.symbolic_valid,
                        "repaired": trace.repaired,
                    },
                )
            )
        self.build_from_records(records, save=save)

    def save(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)

        records_file = self.index_path / "records.pkl"
        tokens_file = self.index_path / "token_sets.pkl"
        dense_file = self.index_path / "dense.npy"
        meta_file = self.index_path / "artifact_meta.json"

        with records_file.open("wb") as f:
            pickle.dump(self._records, f)
        with tokens_file.open("wb") as f:
            pickle.dump(self._token_sets, f)
        if self._dense_matrix is not None:
            np.save(dense_file, self._dense_matrix)
        if self._artifact_meta is not None:
            meta_file.write_text(json.dumps(asdict(self._artifact_meta), indent=2, sort_keys=True), encoding="utf-8")

        if self._faiss_index is not None:
            try:
                import faiss

                faiss.write_index(self._faiss_index, str(self.index_path / "index.faiss"))
            except Exception as exc:
                logger.warning(
                    "retrieval_index_save_faiss_failed",
                    message="Failed to save FAISS index; numpy fallback still available.",
                    payload={"index_path": str(self.index_path), "reason": str(exc)},
                )

    def load(self) -> bool:
        try:
            records_file = self.index_path / "records.pkl"
            tokens_file = self.index_path / "token_sets.pkl"
            dense_file = self.index_path / "dense.npy"
            meta_file = self.index_path / "artifact_meta.json"

            if not records_file.exists():
                return False

            with records_file.open("rb") as f:
                self._records = pickle.load(f)
            if tokens_file.exists():
                with tokens_file.open("rb") as f:
                    self._token_sets = pickle.load(f)
            else:
                logger.warning(
                    "retrieval_token_sets_missing",
                    message="token_sets.pkl missing; rebuilding lexical token sets from records.",
                    payload={"index_path": str(self.index_path)},
                )
                self._token_sets = [set(_tokenize(record.lexical_text())) for record in self._records]
            if dense_file.exists():
                self._dense_matrix = np.load(dense_file).astype(np.float32)
            if meta_file.exists():
                self._artifact_meta = RetrievalArtifactMetadata(**json.loads(meta_file.read_text(encoding="utf-8")))
            elif self._records:
                self._artifact_meta = RetrievalArtifactMetadata(
                    version="v2",
                    built_from_count=len(self._records),
                    embedding_backend=self.embedder.backend,
                    embedding_dimension=(
                        int(self._dense_matrix.shape[1])
                        if self._dense_matrix is not None and self._dense_matrix.ndim == 2
                        else self.embedder.dimension()
                    ),
                    metadata={
                        "reconstructed": True,
                        "record_fingerprint": self._record_fingerprint(self._records),
                        "source_breakdown": self._source_breakdown(self._records),
                    },
                )

            self._load_faiss()
            return self.is_loaded
        except Exception as exc:
            logger.error(
                "retrieval_index_load_failed",
                message="Failed to load retrieval artifact.",
                payload={"index_path": str(self.index_path)},
                error=exc,
            )
            return False

    def search_dense(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[float, IndexedTraceRecord]]:
        if not self.is_loaded:
            return []

        top_k = max(1, int(top_k))
        q = _l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(-1))
        if self._faiss_index is not None:
            try:
                scores, indices = self._faiss_index.search(q.reshape(1, -1), top_k)
                out: list[tuple[float, IndexedTraceRecord]] = []
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(self._records):
                        out.append((float(score), self._records[int(idx)]))
                return out
            except Exception:
                pass

        dense = self._dense_matrix
        if dense is None:
            return []
        sims = dense @ q
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self._records[int(i)]) for i in order]

    def search_lexical(self, query_text: str, top_k: int = 5) -> list[tuple[float, IndexedTraceRecord]]:
        if not self.is_loaded:
            return []
        top_k = max(1, int(top_k))
        query_tokens = set(_tokenize(query_text))
        if not query_tokens:
            return []

        scores: list[tuple[float, IndexedTraceRecord]] = []
        for token_set, record in zip(self._token_sets, self._records):
            union = len(query_tokens | token_set)
            if union == 0:
                continue
            inter = len(query_tokens & token_set)
            score = inter / union
            if score > 0.0:
                scores.append((float(score), record))
        scores.sort(key=lambda x: (-x[0], x[1].trace_id))
        return scores[:top_k]

    def search_structural(
        self,
        *,
        domain: Optional[str] = None,
        archetypes: Optional[Iterable[str]] = None,
        operators: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        proof_obligations: Optional[Iterable[str]] = None,
        failure_modes: Optional[Iterable[str]] = None,
        branch_operators: Optional[Iterable[str]] = None,
        top_k: int = 5,
    ) -> list[tuple[float, IndexedTraceRecord]]:
        if not self.is_loaded:
            return []
        top_k = max(1, int(top_k))

        arch_set = {a for a in (archetypes or []) if a}
        op_set = {o for o in (operators or []) if o}
        tag_set = {t for t in (tags or []) if t}
        obligation_set = {item for item in (proof_obligations or []) if item}
        failure_set = {item for item in (failure_modes or []) if item}
        branch_operator_set = {item for item in (branch_operators or []) if item}
        out: list[tuple[float, IndexedTraceRecord]] = []

        for record in self._records:
            score = 0.0
            if domain and record.domain == domain:
                score += 0.35
            if arch_set:
                record_arch = set(record.archetypes)
                overlap = len(arch_set & record_arch)
                score += 0.20 * (overlap / max(len(arch_set | record_arch), 1))
            if op_set:
                record_ops = set(record.operators_used)
                overlap = len(op_set & record_ops)
                score += 0.25 * (overlap / max(len(op_set | record_ops), 1))
            if tag_set:
                record_tags = set(record.tags)
                overlap = len(tag_set & record_tags)
                score += 0.10 * (overlap / max(len(tag_set | record_tags), 1))
            if obligation_set:
                record_obligations = set(record.proof_obligation_hints)
                overlap = len(obligation_set & record_obligations)
                score += 0.20 * (overlap / max(len(obligation_set | record_obligations), 1))
            if failure_set:
                record_failures = set(record.failure_modes)
                overlap = len(failure_set & record_failures)
                score += 0.15 * (overlap / max(len(failure_set | record_failures), 1))
            if branch_operator_set:
                record_next = set(record.continuation_operators or record.operators_used)
                overlap = len(branch_operator_set & record_next)
                score += 0.12 * (overlap / max(len(branch_operator_set | record_next), 1))
            if score > 0.0:
                out.append((float(score), record))

        out.sort(key=lambda x: (-x[0], x[1].trace_id))
        return out[:top_k]

    def _maybe_build_faiss(self, dense: np.ndarray) -> None:
        try:
            import faiss

            dim = int(dense.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(dense.astype(np.float32))
            self._faiss_index = index
        except Exception as exc:
            logger.warning(
                "retrieval_faiss_unavailable",
                message="FAISS unavailable; using numpy retrieval fallback.",
                payload={"reason": str(exc)},
            )
            self._faiss_index = None

    def _load_faiss(self) -> None:
        idx_file = self.index_path / "index.faiss"
        if not idx_file.exists():
            self._faiss_index = None
            return
        try:
            import faiss

            self._faiss_index = faiss.read_index(str(idx_file))
        except Exception as exc:
            logger.warning(
                "retrieval_faiss_load_failed",
                message="Failed to load FAISS index; using numpy fallback.",
                payload={"index_path": str(idx_file), "reason": str(exc)},
            )
            self._faiss_index = None

    @staticmethod
    def _derive_tags(problem: Optional[ParsedProblem], trace: BranchTrace) -> list[str]:
        tags: set[str] = set()
        if problem is not None:
            tags.add(problem.domain.value)
            tags.update(problem.likely_archetypes[:4])
            if problem.target:
                tags.add(problem.target.lower().replace(" ", "_"))
            if problem.parity_cues:
                tags.add("parity")
            if problem.symmetries:
                tags.add("symmetry")
        if trace.failure_type is not None:
            tags.add(f"failure::{trace.failure_type.value}")
        if trace.repaired:
            tags.add("repaired")
        if trace.symbolic_valid:
            tags.add("symbolic_valid")
        return sorted(tags)

    @staticmethod
    def _derive_subproblem_snippets(problem: Optional[ParsedProblem], trace: BranchTrace) -> list[str]:
        snippets: list[str] = []
        if problem is not None:
            snippets.extend(problem.constraint_texts()[:4])
            snippets.extend(problem.unknowns[:2])
            snippets.extend(problem.knowns[:2])
        for step in trace.steps[:4]:
            if step.description:
                snippets.append(step.description[:180])
        return [s for s in snippets if s][:8]

    @staticmethod
    def _derive_repair_snippets(trace: BranchTrace) -> list[str]:
        snippets: list[str] = []
        if trace.failure_location:
            snippets.append(trace.failure_location)
        if trace.failure_type:
            snippets.append(trace.failure_type.value)
        if trace.repaired:
            snippets.append("repaired_branch")
        return snippets

    @staticmethod
    def _derive_failure_modes(trace: BranchTrace) -> list[str]:
        values: list[str] = []
        if trace.failure_type is not None:
            values.append(trace.failure_type.value)
        if trace.failure_obligation_id:
            values.append("obligation_targeted_repair")
        return sorted(dict.fromkeys(item for item in values if item))

    @staticmethod
    def _derive_proof_obligation_hints(trace: BranchTrace) -> list[str]:
        hints: list[str] = []
        for item in list(trace.proof_obligations or []):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "") or "").strip()
            evidence_kind = str(item.get("evidence_kind_required", "") or "").strip()
            status = str(item.get("status", "") or "").strip()
            if claim:
                hints.append(claim[:120])
            if evidence_kind:
                hints.append(evidence_kind)
            if status in {"open", "partial", "contradicted"}:
                hints.append(f"status::{status}")
        return sorted(dict.fromkeys(item for item in hints if item))[:8]

    @staticmethod
    def _derive_repair_neighbors(trace: BranchTrace) -> list[str]:
        sequence = [item for item in list(trace.operator_sequence or []) if item]
        if len(sequence) < 2:
            return []
        return list(dict.fromkeys(sequence[-2:]))

    @staticmethod
    def _derive_continuation_operators(trace: BranchTrace) -> list[str]:
        sequence = [item for item in list(trace.operator_sequence or []) if item]
        return list(dict.fromkeys(sequence[:4]))

    @staticmethod
    def _record_fingerprint(records: list[IndexedTraceRecord]) -> str:
        payload = [
            {
                "trace_id": record.trace_id,
                "problem_id": record.problem_id,
                "source": record.source,
                "operators": list(record.operators_used),
                "tags": list(record.tags),
            }
            for record in records
        ]
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    @staticmethod
    def _source_breakdown(records: list[IndexedTraceRecord]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            counts[record.source] = counts.get(record.source, 0) + 1
        return dict(sorted(counts.items()))

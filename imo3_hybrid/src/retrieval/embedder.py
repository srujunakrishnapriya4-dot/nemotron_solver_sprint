from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from src.common.logging import get_logger


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[=<>≡+\-*/^(){}\[\],.:;|]")
logger = get_logger("imo3_hybrid.retrieval", module=__name__, component="embedder")


def _safe_normalize_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    try:
        from src.parsing.latex_normalizer import LatexNormalizer

        norm = LatexNormalizer()
        maybe = norm.normalize(raw)
        if isinstance(maybe, str) and maybe.strip():
            return maybe.strip()
    except Exception:
        pass
    return re.sub(r"\s+", " ", raw).strip()


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


@dataclass(frozen=True)
class EmbeddingPayload:
    text: str
    normalized_text: str
    vector: np.ndarray
    method: str
    dimension: int
    metadata: dict[str, object] = field(default_factory=dict)


class DeterministicFallbackEmbedder:
    """
    Deterministic hashed token/character n-gram embedding.
    This is a stable fallback, not a production semantic-model claim.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = int(dimension)

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        rows = [self._encode_one(t, normalize=normalize) for t in texts]
        if not rows:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)

    def _encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        clean = _safe_normalize_text(text).lower()
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not clean:
            return vec

        tokens = _TOKEN_RE.findall(clean)
        features: list[str] = []
        features.extend(tokens)
        features.extend(f"bigram::{a}::{b}" for a, b in zip(tokens, tokens[1:]))
        compact = clean.replace(" ", "")
        features.extend(f"char3::{compact[i:i+3]}" for i in range(max(0, len(compact) - 2)))

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        if normalize:
            vec = _l2_normalize(vec)
        return vec


class MathEmbedder:
    """
    Embedding abstraction with:
    - sentence-transformers path when available
    - deterministic hashed fallback otherwise
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
        fallback_dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.fallback_dimension = int(fallback_dimension)
        self._model = None
        self._backend: Optional[str] = None
        self._fallback = DeterministicFallbackEmbedder(dimension=self.fallback_dimension)

    @property
    def backend(self) -> str:
        if self._backend is None:
            self._ensure_backend()
        return self._backend or "deterministic_fallback"

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "retrieval_embedder_load",
                message=f"Loading retrieval embedder: {self.model_name} on {self.device}",
                payload={"model_name": self.model_name, "device": self.device},
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._backend = "sentence_transformers"
        except Exception as exc:
            logger.warning(
                "retrieval_embedder_fallback",
                message="Sentence-transformer embedder unavailable; falling back to deterministic embedding.",
                payload={"model_name": self.model_name, "device": self.device, "reason": str(exc)},
            )
            self._model = None
            self._backend = "deterministic_fallback"

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        normalized = [_safe_normalize_text(t)[:4000] for t in texts]
        self._ensure_backend()

        if self._backend == "sentence_transformers" and self._model is not None:
            vectors = self._model.encode(
                normalized,
                batch_size=self.batch_size,
                show_progress_bar=len(normalized) > 256,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return np.asarray(vectors, dtype=np.float32)

        return self._fallback.encode(normalized, normalize=normalize)

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        vectors = self.embed([text], normalize=normalize)
        return vectors[0] if len(vectors) else np.zeros((self.dimension(),), dtype=np.float32)

    def payload(
        self,
        text: str,
        *,
        normalize: bool = True,
        metadata: Optional[dict[str, object]] = None,
    ) -> EmbeddingPayload:
        normalized = _safe_normalize_text(text)
        vector = self.embed_single(normalized, normalize=normalize)
        return EmbeddingPayload(
            text=text,
            normalized_text=normalized,
            vector=vector,
            method=self.backend,
            dimension=int(vector.shape[0]),
            metadata=metadata or {},
        )

    def dimension(self) -> int:
        self._ensure_backend()
        if self._backend == "sentence_transformers" and self._model is not None:
            try:
                return int(self._model.get_sentence_embedding_dimension())
            except Exception:
                pass
        return self.fallback_dimension


__all__ = [
    "EmbeddingPayload",
    "DeterministicFallbackEmbedder",
    "MathEmbedder",
    "_safe_normalize_text",
    "_l2_normalize",
]

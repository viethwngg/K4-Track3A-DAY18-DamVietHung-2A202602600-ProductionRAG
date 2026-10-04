"""M2: normalized Vietnamese BM25, Qdrant dense search and rank fusion."""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass
from functools import lru_cache

from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
    model_path,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


@lru_cache(maxsize=4096)
def segment_vietnamese(text: str) -> str:
    from underthesea import word_tokenize

    return word_tokenize(
        unicodedata.normalize("NFC", text).lower(), format="text"
    ).replace("_", " ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", segment_vietnamese(text), re.UNICODE)


class BM25Search:
    def __init__(self):
        self.corpus_tokens, self.documents, self.bm25 = [], [], None

    def index(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [_tokens(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if any(self.corpus_tokens) else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        scores = self.bm25.get_scores(_tokens(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(
                self.documents[i]["text"],
                float(scores[i]),
                dict(self.documents[i].get("metadata", {})),
                "bm25",
            )
            for i in indices
            if scores[i] > 0
        ][:top_k]


@lru_cache(maxsize=2)
def _encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path(model_name))


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        self._encoder = None
        self.indexed = set()
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
            self.client.get_collections()
            self.backend = "qdrant_server"
        except Exception as exc:  # noqa: BLE001 -- boundary with explicit local fallback
            warnings.warn(
                f"Qdrant server unavailable ({type(exc).__name__}); using explicit in-memory fallback."
            )
            self.client = QdrantClient(":memory:")
            self.backend = "qdrant_memory"

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _encoder(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        encoder = self._get_encoder()
        # Derive dimension from the actual model, including user-selected models.
        dim = encoder.get_sentence_embedding_dimension()
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        if chunks:
            vectors = encoder.encode(
                [c["text"] for c in chunks],
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=True,
            )
            points = [
                PointStruct(
                    id=i,
                    vector=v.tolist(),
                    payload={"text": c["text"], "metadata": c.get("metadata", {})},
                )
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ]
            for offset in range(0, len(points), 64):
                self.client.upsert(
                    collection, points=points[offset : offset + 64], wait=True
                )
        self.indexed.add(collection)

    def search(
        self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        if top_k <= 0 or not query.strip() or collection not in self.indexed:
            return []
        vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(
            collection, query=vector, limit=top_k, with_payload=True
        )
        return [
            SearchResult(
                p.payload["text"],
                float(p.score),
                dict(p.payload.get("metadata", {})),
                "dense",
            )
            for p in response.points
        ]


def _identity(result: SearchResult):
    # Distinguish identical text in different policy documents.
    return (
        result.metadata.get("source", ""),
        result.metadata.get("chunk_id", ""),
        result.text,
    )


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60, top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    if top_k <= 0:
        return []
    scores, originals = {}, {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            key = _identity(result)
            if key in seen:
                continue
            seen.add(key)
            originals.setdefault(key, result)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank + 1)
    keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        SearchResult(
            originals[key].text, scores[key], dict(originals[key].metadata), "hybrid"
        )
        for key in keys
    ]


class HybridSearch:
    def __init__(self):
        self.bm25, self.dense = BM25Search(), DenseSearch()
        self.parents = {}
        self.timings = {}

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        return reciprocal_rank_fusion(
            [
                self.bm25.search(query, BM25_TOP_K),
                self.dense.search(query, DENSE_TOP_K),
            ],
            top_k=top_k,
        )

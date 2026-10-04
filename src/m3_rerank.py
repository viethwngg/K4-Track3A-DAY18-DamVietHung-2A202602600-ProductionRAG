"""M3: cached CrossEncoder and optional FlashRank with measured latency."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from config import RERANK_MODEL, RERANK_TOP_K, model_path


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    from src.model_utils import optimize_cpu_model

    encoder = CrossEncoder(model_path(model_name), max_length=512)
    if model_name.startswith("BAAI/bge"):
        optimize_cpu_model(encoder.model)
    return encoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name, self._model = model_name, None

    def _load_model(self):
        if self._model is None:
            self._model = _cross_encoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        scores = np.asarray(
            self._load_model().predict(
                [(query, d["text"]) for d in documents],
                batch_size=8,
                show_progress_bar=False,
            )
        ).reshape(-1)
        if len(scores) != len(documents):
            raise ValueError("CrossEncoder must return one score per document")
        indices = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:top_k]
        return [
            RerankResult(
                documents[i]["text"],
                float(documents[i].get("score", 0)),
                float(scores[i]),
                dict(documents[i].get("metadata", {})),
                rank,
            )
            for rank, i in enumerate(indices)
        ]


class FlashrankReranker:
    def __init__(self):
        self._model = None

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        from flashrank import Ranker, RerankRequest

        if not documents or top_k <= 0:
            return []
        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))[
            :top_k
        ]
        return [
            RerankResult(
                documents[r["id"]]["text"],
                documents[r["id"]].get("score", 0),
                float(r["score"]),
                dict(documents[r["id"]].get("metadata", {})),
                i,
            )
            for i, r in enumerate(results)
        ]


def benchmark_reranker(
    reranker, query: str, documents: list[dict], n_runs: int = 5
) -> dict:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }

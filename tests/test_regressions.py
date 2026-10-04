"""Edge cases not covered by the provided happy-path tests (no API calls)."""

import json

import numpy as np
import pytest

from src.m1_chunking import chunk_hierarchical, chunk_structure_aware
from src.m2_search import DenseSearch, SearchResult, reciprocal_rank_fusion
from src.m4_eval import evaluate_ragas, save_report
from src.m5_enrichment import enrich_chunks


def test_hierarchy_bounds_and_document_namespaces():
    text = "longword" * 200
    first, children = chunk_hierarchical(text, 100, 30, {"source": "first.md"})
    second, _ = chunk_hierarchical(text, 100, 30, {"source": "second.md"})
    assert max(len(p.text) for p in first) <= 100
    assert max(len(c.text) for c in children) <= 30
    assert {p.metadata["parent_id"] for p in first}.isdisjoint(
        p.metadata["parent_id"] for p in second
    )
    assert "".join(c.text for c in children) == text


def test_markdown_code_headers_are_not_sections():
    text = "# Title\n```python\n# code comment\n```\n## Table\n| a | b |\n|---|---|\n| 1 | 2 |"
    chunks = chunk_structure_aware(text)
    assert len(chunks) == 2
    assert "# code comment" in chunks[0].text
    assert "| 1 | 2 |" in chunks[1].text


def test_rrf_duplicate_hit_counted_once_and_sources_distinct():
    a = SearchResult("same", 1, {"source": "a"}, "dense")
    b = SearchResult("same", 1, {"source": "b"}, "dense")
    result = reciprocal_rank_fusion([[a, a, b], [a]])
    assert len(result) == 2
    assert result[0].score == pytest.approx(2 / 61)
    assert a.method == "dense"


def test_dense_real_qdrant_roundtrip_with_small_encoder():
    from qdrant_client import QdrantClient

    class Encoder:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            def encode_one(text):
                return [1.0, 0.0] if "leave" in text else [0.0, 1.0]

            return np.array(
                [encode_one(t) for t in texts]
                if isinstance(texts, list)
                else encode_one(texts)
            )

    search = DenseSearch.__new__(DenseSearch)
    search.client, search._encoder, search.indexed = (
        QdrantClient(":memory:"),
        Encoder(),
        set(),
    )
    search.index(
        [
            {"text": "leave", "metadata": {"source": "hr"}},
            {"text": "vpn", "metadata": {"source": "it"}},
        ],
        "roundtrip",
    )
    hits = search.search("leave", 1, "roundtrip")
    assert hits[0].text == "leave"
    assert hits[0].metadata == {"source": "hr"}
    search.client.close()


def test_combined_one_call_and_ingestion_metadata_preserved(monkeypatch, tmp_path):
    from src import m5_enrichment as module

    calls = []
    monkeypatch.setattr(module, "ENRICHMENT_CACHE_DIR", str(tmp_path))

    def request(*args):
        calls.append(args)
        return json.dumps(
            {
                "summary": "summary",
                "questions": ["question?"],
                "context": "context",
                "metadata": {"source": "invented", "parent_id": "wrong"},
            }
        )

    monkeypatch.setattr(module, "_request", request)
    module._enrich_single_call.cache_clear()
    chunk = {"text": "original", "metadata": {"source": "truth.md", "parent_id": "p1"}}
    result = enrich_chunks([chunk])[0]
    assert len(calls) == 1
    assert result.auto_metadata["source"] == "truth.md"
    assert result.auto_metadata["parent_id"] == "p1"
    assert result.original_text in result.enriched_text
    module._enrich_single_call.cache_clear()


def test_no_key_report_retains_questions_without_invented_scores(monkeypatch, tmp_path):
    from src import m4_eval as module

    monkeypatch.setattr(module, "OPENAI_API_KEY", "")
    result = evaluate_ragas(["q"], ["a"], [["context"]], ["truth"])
    path = tmp_path / "report.json"
    save_report(result, [], str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["num_questions"] == 1
    assert report["evaluation_status"] == "unavailable"
    assert report["aggregate"]["faithfulness"] is None
    assert report["per_question"][0]["question"] == "q"


def test_evaluation_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_ragas(["q"], [], [], [])


def test_version_filter_preserves_unrelated_documents():
    from src.pipeline import _current_documents

    docs = [
        {"text": "policy", "metadata": {"source": name}}
        for name in (
            "leave_v2023.md",
            "leave_v2024.md",
            "password_v1.md",
            "password_v2.md",
            "salary.md",
        )
    ]
    assert [d["metadata"]["source"] for d in _current_documents(docs)] == [
        "leave_v2024.md",
        "password_v2.md",
        "salary.md",
    ]


def test_run_query_expands_and_deduplicates_parents(monkeypatch):
    from types import SimpleNamespace

    from src import pipeline
    from src.m1_chunking import Chunk
    from src.m3_rerank import RerankResult

    hits = [
        SearchResult("child", 1.0, {"parent_id": pid, "source": source}, "hybrid")
        for pid, source in (("a", "first.md"), ("a", "first.md"), ("b", "second.md"))
    ]
    search = SimpleNamespace(
        parents={"a": Chunk("FULL PARENT A"), "b": Chunk("FULL PARENT B")},
        search=lambda query: hits,
    )
    reranker = SimpleNamespace(
        rerank=lambda query, docs, top_k: [
            RerankResult(d["text"], 1.0, 1.0, d["metadata"], i)
            for i, d in enumerate(docs)
        ]
    )
    monkeypatch.setattr(
        pipeline, "generate_answer", lambda query, contexts: ("answer", "mock")
    )
    answer, contexts = pipeline.run_query("q", search, reranker)
    assert answer == "answer"
    assert len(contexts) == 2
    assert "FULL PARENT A" in contexts[0] and "FULL PARENT B" in contexts[1]
    assert search.last_query["sources"] == ["first.md", "second.md"]

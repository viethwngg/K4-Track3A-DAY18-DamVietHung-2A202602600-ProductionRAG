"""Naive baseline: basic chunks and dense-only retrieval on the full corpus."""

import sys
import time
from pathlib import Path

from config import EMBEDDING_MODEL, NAIVE_COLLECTION, REPORT_DIR
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.pipeline import generate_answer


def main():
    start = time.perf_counter()
    docs = load_documents()
    chunks = [
        {"text": c.text, "metadata": c.metadata}
        for doc in docs
        for c in chunk_basic(doc["text"], metadata=doc["metadata"])
    ]
    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)
    build_ms = (time.perf_counter() - start) * 1000
    questions, answers, contexts, truths, timings = [], [], [], [], []
    for item in load_test_set():
        start = time.perf_counter()
        hits = search.search(item["question"], 3, NAIVE_COLLECTION)
        retrieval_ms = (time.perf_counter() - start) * 1000
        evidence = [
            f"Nguồn: {r.metadata.get('source', 'unknown')}\n{r.text}" for r in hits
        ]
        start = time.perf_counter()
        answer, backend = generate_answer(item["question"], evidence)
        timings.append(
            {
                "question": item["question"],
                "retrieval_ms": retrieval_ms,
                "generation_ms": (time.perf_counter() - start) * 1000,
                "answer_backend": backend,
                "sources": [r.metadata.get("source") for r in hits],
            }
        )
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(evidence)
        truths.append(item["ground_truth"])
        print(f"Baseline [{len(questions)}/20] {item['question']}", flush=True)
    start = time.perf_counter()
    results = evaluate_ragas(questions, answers, contexts, truths)
    results["latency"] = {
        "build": {"total_ms": build_ms},
        "queries": timings,
        "evaluation_ms": (time.perf_counter() - start) * 1000,
    }
    results["runtime"] = {
        "embedding_model": EMBEDDING_MODEL,
        "qdrant_backend": search.backend,
        "documents_indexed": len(docs),
        "chunks": len(chunks),
        "strategy": "basic_dense_only",
    }
    save_report(
        results,
        failure_analysis(results["per_question"], 5),
        str(Path(REPORT_DIR) / "naive_baseline_report.json"),
    )
    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

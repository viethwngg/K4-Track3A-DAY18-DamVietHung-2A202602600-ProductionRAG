"""M4: genuine RAGAS evaluation, honest unavailable status and diagnostic tree."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from config import LLM_MODEL, OPENAI_API_KEY, REPORT_DIR, TEST_SET_PATH

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("Evaluation inputs must have equal lengths")
    records = [
        EvalResult(q, a, list(c), gt, 0, 0, 0, 0)
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
    ]
    base = {m: 0.0 for m in METRICS}
    base.update(
        per_question=records, evaluation_status="unavailable", evaluator="ragas"
    )
    if not questions:
        base["evaluation_error"] = "No evaluation questions"
        return base
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
        base["evaluation_error"] = "OPENAI_API_KEY is not configured"
        return base
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY),
            embeddings=OpenAIEmbeddings(
                model="text-embedding-3-small", api_key=OPENAI_API_KEY
            ),
            run_config=RunConfig(timeout=120, max_retries=2, max_workers=4),
            raise_exceptions=True,
        )
        frame = result.to_pandas()
        records = []
        for _, row in frame.iterrows():
            values = {m: float(row[m]) for m in METRICS}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError("RAGAS returned non-finite metrics")
            records.append(
                EvalResult(
                    row["question"],
                    row["answer"],
                    list(row["contexts"]),
                    row["ground_truth"],
                    **values,
                )
            )
        return {
            **{m: sum(getattr(r, m) for r in records) / len(records) for m in METRICS},
            "per_question": records,
            "evaluation_status": "completed",
            "evaluator": "ragas",
        }
    except Exception as exc:  # noqa: BLE001 -- preserve failed evaluation data
        # Keep data for reproducibility; never replace API failures with invented scores.
        base["evaluation_error"] = (
            f"{type(exc).__name__}: RAGAS evaluation failed; check API access, quota and dependencies."
        )
        print(f"  {base['evaluation_error']}", flush=True)
        return base


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 5) -> list[dict]:
    tree = {
        "faithfulness": (
            "Generation: answer contains unsupported claims",
            "Use context-only instructions, citations and temperature=0",
        ),
        "answer_relevancy": (
            "Generation: answer does not address the question",
            "Ask for direct answers; handle negation and arithmetic explicitly",
        ),
        "context_precision": (
            "Retrieval: irrelevant or obsolete context",
            "Rerank candidates and filter superseded policies",
        ),
        "context_recall": (
            "Retrieval: required evidence is missing",
            "Expand parent context, increase candidates and retrieve each sub-question",
        ),
    }
    ranked = []
    for item in eval_results:
        record = asdict(item) if isinstance(item, EvalResult) else dict(item)
        worst = min(METRICS, key=lambda m: record[m])
        diagnosis, fix = tree[worst]
        ranked.append(
            {
                **record,
                "worst_metric": worst,
                "score": record[worst],
                "average_score": sum(record[m] for m in METRICS) / 4,
                "diagnosis": diagnosis,
                "suggested_fix": fix,
                "error_tree": "Output incorrect -> evidence complete? -> retrieval relevant/current? -> generation grounded?",
            }
        )
    return sorted(ranked, key=lambda r: r["average_score"])[: max(0, bottom_n)]


def save_report(results: dict, failures: list[dict], path: str | None = None):
    path = Path(path) if path else Path(REPORT_DIR) / "ragas_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = results.get("evaluation_status") == "completed"
    records = [
        asdict(r) if isinstance(r, EvalResult) else dict(r)
        for r in results.get("per_question", [])
    ]
    if not completed:
        for record in records:
            for metric in METRICS:
                record[metric] = None
    report = {
        "aggregate": {m: results.get(m) if completed else None for m in METRICS},
        "num_questions": len(records),
        "per_question": records,
        "evaluation_status": results.get("evaluation_status", "unavailable"),
        "evaluation_error": results.get("evaluation_error"),
        "evaluator": "ragas",
        "failures": failures if completed else [],
        "latency": results.get("latency", {}),
        "runtime": results.get("runtime", {}),
    }
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Report saved to {path}")

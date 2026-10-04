"""Run both pipelines and print a comparable, honest RAGAS score table."""

import sys

from naive_baseline import main as run_baseline
from src.m4_eval import METRICS
from src.pipeline import build_pipeline, evaluate_pipeline


def main():
    baseline = run_baseline()
    search, reranker = build_pipeline()
    production = evaluate_pipeline(search, reranker)
    print(f"{'Metric':<24} {'Baseline':>10} {'Production':>12} {'Delta':>10}")
    for metric in METRICS:
        if (
            baseline["evaluation_status"]
            == production["evaluation_status"]
            == "completed"
        ):
            old, new = baseline[metric], production[metric]
            print(f"{metric:<24} {old:>10.4f} {new:>12.4f} {new - old:>+10.4f}")
        else:
            print(f"{metric:<24} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
    print(
        f"Evaluation: baseline={baseline['evaluation_status']}, production={production['evaluation_status']}"
    )
    from scripts.write_analysis import main as write_analysis

    write_analysis()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

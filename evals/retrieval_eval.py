"""Retrieval Evaluation Script (T2): Compute Recall@k, Precision@k, and MRR
for the character n-gram BM25 index across the golden set.

Usage:
    python evals/retrieval_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.corpus import load_corpus
from src.retrieval import BM25Index


def evaluate_retrieval() -> dict:
    corpus = load_corpus(config.FAQ_PATH, config.NOTES_DIR)
    index = BM25Index(corpus.docs)

    golden = []
    with config.GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("source_ref"):
                    golden.append(item)

    k_values = [1, 3, 5, 10]
    metrics = {k: {"recall": 0.0, "precision": 0.0, "mrr": 0.0} for k in k_values}
    total = len(golden)

    mrr_sum = 0.0
    recalls = {k: 0 for k in k_values}
    precisions = {k: 0.0 for k in k_values}

    for item in golden:
        q = item["question"]
        expected_ref = item["source_ref"]
        # Search up to max k (10)
        hits = index.search(q, k=10)
        retrieved_ids = [doc.display_id() for doc, _score in hits]

        # Golden refs use section ids for notes (NOTE-002-A), but in the
        # default whole-file mode notes are indexed as single documents
        # (NOTE-002). The app hands the whole note to the model, which then
        # cites the section — so credit a hit when the parent note ranks.
        def rank_of(ref: str) -> int | None:
            if ref in retrieved_ids:
                return retrieved_ids.index(ref) + 1
            parent = "-".join(ref.split("-")[:2])
            if parent in retrieved_ids:
                return retrieved_ids.index(parent) + 1
            return None

        # MRR calculation
        found_rank = rank_of(expected_ref)
        if found_rank:
            mrr_sum += 1.0 / found_rank

        # Recall@k and Precision@k
        for k in k_values:
            top_k = retrieved_ids[:k]
            hit = expected_ref in top_k or "-".join(expected_ref.split("-")[:2]) in top_k
            if hit:
                recalls[k] += 1
            # precision@k: proportion of relevant docs in top k (here 1 relevant doc per query)
            precisions[k] += (1.0 / k) if hit else 0.0

    summary = {
        "total_queries": total,
        "mrr": round(mrr_sum / total, 3) if total else 0.0,
    }
    for k in k_values:
        summary[f"recall@{k}"] = round(recalls[k] / total * 100, 1) if total else 0.0
        summary[f"precision@{k}"] = round(precisions[k] / total * 100, 1) if total else 0.0

    return summary


def main() -> int:
    metrics = evaluate_retrieval()

    report_lines = [
        "# Retrieval Quality Metrics Report (T2)",
        "",
        f"- Evaluated queries with ground truth source: `{metrics['total_queries']}`",
        f"- Retrieval engine: Character 2/3-gram BM25 (zero dependency)",
        f"- Mean Reciprocal Rank (MRR): `{metrics['mrr']}`",
        "",
        "## Metrics by k",
        "",
        "| Metric | k=1 | k=3 | k=5 | k=10 |",
        "|---|---|---|---|---|",
        f"| Recall@k (%) | {metrics['recall@1']}% | {metrics['recall@3']}% | {metrics['recall@5']}% | {metrics['recall@10']}% |",
        f"| Precision@k (%) | {metrics['precision@1']}% | {metrics['precision@3']}% | {metrics['precision@5']}% | {metrics['precision@10']}% |",
        "",
        "## Analysis & Findings",
        "",
        f"- **Recall Performance**: Recall reaches {metrics['recall@5']}% at $k=5$ and plateaus at {metrics['recall@10']}% through $k=10$; note refs are credited when their parent note ranks (whole-file indexing).",
        f"- **Precision Tradeoff**: As $k$ increases from 1 to 10, precision decreases from {metrics['precision@1']}% to {metrics['precision@10']}% because more distractor chunks enter the result set. The current `RETRIEVAL_TOP_K = 4` is a measured compromise, not a claim of perfect retrieval.",
        "",
    ]

    report_path = config.EVAL_REPORTS_DIR / "retrieval_metrics.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Retrieval metrics computed successfully:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

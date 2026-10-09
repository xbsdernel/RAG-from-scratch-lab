"""
Basic retrieval evaluation framework.

Computes Recall@k and MRR (Mean Reciprocal Rank) given:
  - A set of ground-truth relevant chunk IDs per question.
  - A retrieval function.

NOTE: Meaningful evaluation requires correct ground-truth labels.
      Test results reflect the quality of both labels AND the retrieval system.
      Passing evaluation != factual accuracy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class EvalQuestion:
    question: str
    relevant_chunk_ids: list[str]  # Ground-truth relevant chunk IDs


@dataclass
class EvalResult:
    question: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank for this question


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of first relevant item, or 0 if none found."""
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    questions: list[EvalQuestion],
    retrieval_fn: Callable[[str, int], list[str]],
    k: int = 5,
) -> dict[str, Any]:
    """
    Run evaluation over a set of questions.

    Args:
        questions: List of EvalQuestion with ground-truth relevant IDs.
        retrieval_fn: Function(query, k) → list of chunk_ids (ranked).
        k: Number of retrieved results to evaluate.

    Returns:
        Dict with per-question results and aggregate metrics.
    """
    results: list[EvalResult] = []

    for eq in questions:
        retrieved_ids = retrieval_fn(eq.question, k)
        relevant_set = set(eq.relevant_chunk_ids)
        rk = recall_at_k(retrieved_ids, relevant_set, k)
        rr = reciprocal_rank(retrieved_ids, relevant_set)
        results.append(EvalResult(
            question=eq.question,
            retrieved_ids=retrieved_ids,
            relevant_ids=eq.relevant_chunk_ids,
            recall_at_k=rk,
            mrr=rr,
        ))

    avg_recall = sum(r.recall_at_k for r in results) / len(results) if results else 0.0
    avg_mrr = sum(r.mrr for r in results) / len(results) if results else 0.0

    return {
        "k": k,
        "n_questions": len(results),
        "avg_recall_at_k": round(avg_recall, 4),
        "avg_mrr": round(avg_mrr, 4),
        "per_question": [
            {
                "question": r.question,
                "recall_at_k": round(r.recall_at_k, 4),
                "mrr": round(r.mrr, 4),
                "retrieved": r.retrieved_ids,
                "relevant": r.relevant_ids,
            }
            for r in results
        ],
    }


# ─── Sample evaluation dataset ────────────────────────────────────────────────
# NOTE: chunk_ids here are illustrative — they reference the sample documents
# in data/sample_docs/. You must index those documents first.
SAMPLE_EVAL_QUESTIONS = [
    EvalQuestion(
        question="What is Retrieval-Augmented Generation (RAG)?",
        relevant_chunk_ids=["sample_rag_0000", "sample_rag_0001"],
    ),
    EvalQuestion(
        question="How does vector similarity search work?",
        relevant_chunk_ids=["sample_rag_0002", "sample_rag_0003"],
    ),
    EvalQuestion(
        question="What is the purpose of chunking in RAG?",
        relevant_chunk_ids=["sample_rag_0001", "sample_rag_0004"],
    ),
]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from rag_app.core.vector_store import get_vector_store

    vs = get_vector_store()

    def retrieval_fn(query: str, k: int) -> list[str]:
        docs = vs.similarity_search(query=query, k=k)
        return [d.metadata.get("chunk_id", "") for d in docs]

    report = evaluate(SAMPLE_EVAL_QUESTIONS, retrieval_fn, k=5)
    print(json.dumps(report, indent=2))
    print(f"\n{'='*40}")
    print(f"Avg Recall@5: {report['avg_recall_at_k']}")
    print(f"Avg MRR:      {report['avg_mrr']}")
    print("\nNote: Results are meaningful only if ground-truth chunk_ids are correct for your indexed documents.")

"""
Mode 5: RAG Fusion with Reciprocal Rank Fusion.

1. Generate N query variations.
2. Retrieve a ranked list for each query.
3. Merge lists using RRF.
4. Select top-k by RRF score.
5. Generate the final answer.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from rag_app.core.llm_provider import LLMError, get_llm, safe_invoke
from rag_app.core.prompts import basic_rag_prompt, format_context
from rag_app.core.tracing import Trace
from rag_app.core.vector_store import VectorStore
from rag_app.pipelines.query_generator import generate_alternative_queries
from rag_app.pipelines.rrf import RRFResult, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class RAGFusionPipeline:
    """Mode 5: RAG Fusion with RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: str = "gemini",
        llm_model: str | None = None,
        rrf_c: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.rrf_c = rrf_c

    def run(
        self,
        query: str,
        top_k: int = 5,
        n_queries: int = 4,
        trace: Trace | None = None,
    ) -> dict[str, Any]:
        llm = get_llm(provider=self.llm_provider, model=self.llm_model)

        # ── Step 1: Generate queries ──────────────────────────────────────────
        alt_queries = generate_alternative_queries(query, llm, n_queries=n_queries)
        all_queries = [query] + alt_queries

        if trace:
            trace.record("query_generation", {"original": query, "variations": alt_queries})

        # ── Step 2: Retrieve ranked lists ─────────────────────────────────────
        ranked_lists: list[list[Document]] = []
        per_query_results: dict[str, list[Document]] = {}

        for q in all_queries:
            docs = self.vector_store.similarity_search(query=q, k=top_k)
            ranked_lists.append(docs)
            per_query_results[q] = docs

        if trace:
            trace.record(
                "ranked_retrieval",
                {
                    "queries": all_queries,
                    "per_query_counts": {q: len(d) for q, d in per_query_results.items()},
                },
            )

        # ── Step 3: Apply RRF ─────────────────────────────────────────────────
        rrf_results: list[RRFResult] = reciprocal_rank_fusion(ranked_lists, c=self.rrf_c)

        if trace:
            trace.record(
                "rrf_fusion",
                {
                    "rrf_constant": self.rrf_c,
                    "total_unique_docs": len(rrf_results),
                    "top_results": [r.to_dict() for r in rrf_results[:top_k]],
                },
            )

        # ── Step 4: Select top-k by RRF score ────────────────────────────────
        top_docs: list[Document] = [r.document for r in rrf_results[:top_k]]
        top_rrf: list[RRFResult] = rrf_results[:top_k]

        # ── Step 5: Generate final answer ─────────────────────────────────────
        context = format_context(top_docs)
        messages = basic_rag_prompt.format_messages(question=query, context=context)
        prompt_str = "\n\n".join(str(m.content) for m in messages)

        try:
            answer = safe_invoke(llm, messages)
        except LLMError as exc:
            answer = f"❌ LLM Error: {exc}"

        if trace:
            trace.record("generation", {"answer_preview": answer[:200]})

        return {
            "answer": answer,
            "original_query": query,
            "all_queries": all_queries,
            "per_query_results": per_query_results,
            "rrf_results": rrf_results,
            "top_docs": top_docs,
            "top_rrf": top_rrf,
            "prompt": prompt_str,
            "context": context,
            "rrf_c": self.rrf_c,
        }

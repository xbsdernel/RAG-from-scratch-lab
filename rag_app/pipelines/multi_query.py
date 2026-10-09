"""
Mode 4: Multi-Query Retrieval.

1. Generate N alternative query formulations using the LLM.
2. Retrieve top-k chunks for each query.
3. Deduplicate by chunk_id.
4. Pass the merged context + original question to the LLM.
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

logger = logging.getLogger(__name__)


class MultiQueryPipeline:
    """Mode 4: Multi-Query retrieval."""

    def __init__(self, vector_store: VectorStore, llm_provider: str = "gemini", llm_model: str | None = None) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def run(
        self,
        query: str,
        top_k: int = 5,
        n_queries: int = 3,
        trace: Trace | None = None,
    ) -> dict[str, Any]:
        llm = get_llm(provider=self.llm_provider, model=self.llm_model)

        # ── Step 1: Generate alternative queries ─────────────────────────────
        alt_queries = generate_alternative_queries(query, llm, n_queries=n_queries)
        all_queries = [query] + alt_queries

        if trace:
            trace.record("query_generation", {"original": query, "alternatives": alt_queries})

        # ── Step 2: Retrieve for each query ───────────────────────────────────
        per_query_results: dict[str, list[Document]] = {}
        seen_ids: set[str] = set()
        merged_docs: list[Document] = []

        for q in all_queries:
            docs = self.vector_store.similarity_search(query=q, k=top_k)
            per_query_results[q] = docs
            for doc in docs:
                cid = doc.metadata.get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged_docs.append(doc)

        if trace:
            trace.record(
                "multi_retrieval",
                {
                    "queries": list(per_query_results.keys()),
                    "per_query_counts": {q: len(d) for q, d in per_query_results.items()},
                    "merged_unique_docs": len(merged_docs),
                },
            )

        # ── Step 3: Generate final answer ─────────────────────────────────────
        context = format_context(merged_docs)
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
            "merged_docs": merged_docs,
            "prompt": prompt_str,
            "context": context,
        }

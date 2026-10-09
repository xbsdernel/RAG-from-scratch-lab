"""
Mode 1 & 3: Basic RAG pipeline and Generation/Prompt Inspection.

Pipeline steps:
1. Embed the query.
2. Retrieve top-k chunks from the vector store.
3. Format the prompt.
4. Call the LLM.
5. Return the answer with the full trace.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document

from rag_app.core.llm_provider import LLMError, get_llm, safe_invoke
from rag_app.core.prompts import basic_rag_prompt, format_context
from rag_app.core.tracing import Trace
from rag_app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class BasicRAGPipeline:
    """
    Implements Mode 1 (Basic RAG) and Mode 3 (Generation Inspection).
    The same underlying pipeline powers both; Mode 3 simply exposes more internals.
    """

    def __init__(self, vector_store: VectorStore, llm_provider: str = "gemini", llm_model: str | None = None) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def run(
        self,
        query: str,
        top_k: int = 5,
        trace: Trace | None = None,
    ) -> dict[str, Any]:
        """
        Execute the basic RAG pipeline.

        Returns:
            {
                "answer": str,
                "retrieved_docs": list[Document],
                "prompt": str,
                "context": str,
            }
        """
        t0 = time.time()

        # ── Step 1: Retrieve ─────────────────────────────────────────────────
        t_ret = time.time()
        docs_with_scores = self.vector_store.similarity_search_with_score(query=query, k=top_k)
        ret_ms = (time.time() - t_ret) * 1000
        docs = [d for d, _ in docs_with_scores]
        scores = [s for _, s in docs_with_scores]

        if trace:
            trace.record(
                "retrieval",
                {
                    "query": query,
                    "top_k": top_k,
                    "retrieved": [
                        {
                            "rank": i + 1,
                            "source": d.metadata.get("source"),
                            "page": d.metadata.get("page"),
                            "chunk_id": d.metadata.get("chunk_id"),
                            "score_note": "L2 distance (lower=more similar)",
                            "score": round(float(scores[i]), 4) if i < len(scores) else None,
                            "preview": d.page_content[:120] + "…",
                        }
                        for i, d in enumerate(docs)
                    ],
                },
                elapsed_ms=ret_ms,
            )

        # ── Step 2: Build context and prompt ─────────────────────────────────
        context = format_context(docs)
        formatted = basic_rag_prompt.format_messages(question=query, context=context)
        # Build readable prompt string (no API key exposure)
        prompt_str = "\n\n".join(str(m.content) for m in formatted)

        if trace:
            trace.record(
                "prompt_construction",
                {"formatted_prompt": prompt_str, "context_doc_count": len(docs)},
            )

        # ── Step 3: Generate ─────────────────────────────────────────────────
        t_gen = time.time()
        try:
            llm = get_llm(provider=self.llm_provider, model=self.llm_model)
            answer = safe_invoke(llm, formatted)
        except LLMError as exc:
            err = str(exc)
            if trace:
                trace.record("generation", {}, error=err)
            return {"answer": f"❌ LLM Error: {err}", "retrieved_docs": docs, "prompt": prompt_str, "context": context}
        gen_ms = (time.time() - t_gen) * 1000

        if trace:
            trace.record("generation", {"answer_preview": answer[:200]}, elapsed_ms=gen_ms)

        return {
            "answer": answer,
            "retrieved_docs": docs,
            "retrieved_with_scores": docs_with_scores,
            "prompt": prompt_str,
            "context": context,
        }

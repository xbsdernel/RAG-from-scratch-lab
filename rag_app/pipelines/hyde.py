"""
Mode 9: Hypothetical Document Embeddings (HyDE).

1. Ask the LLM to generate a hypothetical passage that would answer the question.
2. Embed the hypothetical passage using the local embedding model.
3. Use that embedding vector to retrieve REAL document chunks from Chroma.
4. Generate the final answer using the original question + REAL retrieved chunks.

CRITICAL: The hypothetical passage is NEVER inserted into the evidence context.
          It is only a retrieval signal. It is displayed separately in the UI.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from rag_app.core.embeddings import E5Embeddings
from rag_app.core.llm_provider import LLMError, get_llm, safe_invoke
from rag_app.core.prompts import format_context, hyde_answer_prompt, hyde_generation_prompt
from rag_app.core.tracing import Trace
from rag_app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HyDEPipeline:
    """Mode 9: Hypothetical Document Embeddings."""

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: E5Embeddings,
        llm_provider: str = "gemini",
        llm_model: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def _generate_hypothetical_passage(self, query: str, llm: Any) -> str:
        messages = hyde_generation_prompt.format_messages(question=query)
        try:
            return safe_invoke(llm, messages).strip()
        except LLMError as exc:
            logger.error("HyDE generation failed: %s", exc)
            # Fallback: use the original query directly
            return query

    def run(
        self,
        query: str,
        top_k: int = 5,
        trace: Trace | None = None,
    ) -> dict[str, Any]:
        llm = get_llm(provider=self.llm_provider, model=self.llm_model)

        # ── Step 1: Generate hypothetical passage ─────────────────────────────
        hypothetical_passage = self._generate_hypothetical_passage(query, llm)

        if trace:
            trace.record(
                "hyde_generation",
                {
                    "original_query": query,
                    "hypothetical_passage": hypothetical_passage,
                    "warning": "This passage is AI-generated and NOT verified factual evidence.",
                },
            )

        # ── Step 2: Embed the hypothetical passage (as a query) ───────────────
        # We use embed_query because E5's "query: " prefix is used for retrieval signals.
        try:
            hyp_embedding = self.embeddings.embed_query(hypothetical_passage)
        except Exception as exc:
            logger.error("Failed to embed hypothetical passage: %s", exc)
            hyp_embedding = self.embeddings.embed_query(query)  # fallback to original

        if trace:
            trace.record(
                "hyde_embedding",
                {
                    "embedded_text_preview": hypothetical_passage[:100],
                    "embedding_dim": len(hyp_embedding),
                },
            )

        # ── Step 3: Retrieve REAL documents using hypothetical embedding ───────
        real_docs = self.vector_store.similarity_search_by_vector(
            embedding=hyp_embedding,
            k=top_k,
        )

        if trace:
            trace.record(
                "vector_retrieval",
                {
                    "retrieved_real_docs": len(real_docs),
                    "sources": [d.metadata.get("source") for d in real_docs],
                },
            )

        # ── Step 4: Generate final answer using REAL docs only ─────────────────
        # The hypothetical passage is NOT included in the context.
        real_context = format_context(real_docs)
        messages = hyde_answer_prompt.format_messages(
            question=query,
            context=real_context,
        )
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
            "hypothetical_passage": hypothetical_passage,
            "retrieved_real_docs": real_docs,
            "real_context": real_context,
            "prompt": prompt_str,
        }

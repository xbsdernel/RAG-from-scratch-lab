"""
Mode 8: Step-Back Prompting.

1. Preserve the original specific question.
2. Ask the LLM to generate a more abstract "step-back" question.
3. Retrieve documents independently for BOTH the original and step-back queries.
4. Combine evidence (deduplicated, with provenance labels).
5. Generate final answer using both types of evidence.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from rag_app.core.llm_provider import LLMError, get_llm, safe_invoke
from rag_app.core.prompts import (
    format_context,
    step_back_answer_prompt,
    step_back_generation_prompt,
)
from rag_app.core.tracing import Trace
from rag_app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class StepBackPipeline:
    """Mode 8: Step-Back Prompting."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: str = "gemini",
        llm_model: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def _generate_step_back_query(self, original_query: str, llm: Any) -> str:
        messages = step_back_generation_prompt.format_messages(question=original_query)
        try:
            return safe_invoke(llm, messages).strip()
        except LLMError as exc:
            logger.error("Step-back generation failed: %s", exc)
            return original_query  # graceful fallback

    def run(
        self,
        query: str,
        top_k: int = 4,
        trace: Trace | None = None,
    ) -> dict[str, Any]:
        llm = get_llm(provider=self.llm_provider, model=self.llm_model)

        # ── Step 1: Generate step-back query ──────────────────────────────────
        step_back_query = self._generate_step_back_query(query, llm)

        if trace:
            trace.record(
                "step_back_generation",
                {"original": query, "step_back": step_back_query},
            )

        # ── Step 2: Retrieve for BOTH queries independently ───────────────────
        specific_docs = self.vector_store.similarity_search(query=query, k=top_k)
        background_docs = self.vector_store.similarity_search(query=step_back_query, k=top_k)

        if trace:
            trace.record(
                "dual_retrieval",
                {
                    "specific_query": query,
                    "specific_docs": len(specific_docs),
                    "step_back_query": step_back_query,
                    "background_docs": len(background_docs),
                },
            )

        # ── Step 3: Deduplicate with provenance ───────────────────────────────
        specific_ids = {d.metadata.get("chunk_id", "") for d in specific_docs}
        background_context = format_context(background_docs)
        specific_context = format_context(specific_docs)

        # Highlight in background_context which chunks were also in specific
        background_only_docs = [
            d for d in background_docs if d.metadata.get("chunk_id", "") not in specific_ids
        ]

        # ── Step 4: Generate final answer ─────────────────────────────────────
        messages = step_back_answer_prompt.format_messages(
            original_question=query,
            background_context=background_context,
            specific_context=specific_context,
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
            "step_back_query": step_back_query,
            "specific_docs": specific_docs,
            "background_docs": background_docs,
            "background_only_docs": background_only_docs,
            "specific_context": specific_context,
            "background_context": background_context,
            "prompt": prompt_str,
        }

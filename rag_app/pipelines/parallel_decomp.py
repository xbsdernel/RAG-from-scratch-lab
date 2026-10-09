"""
Mode 7: Parallel Query Decomposition.

Independent subquestion processing:
1. Decompose question into independent sub-questions.
2. Retrieve evidence for each sub-question independently.
3. Answer each sub-question independently (no shared history).
4. Run retrieval + answering concurrently via ThreadPoolExecutor.
5. Synthesize a final answer from all partial answers.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.documents import Document

from rag_app.core.llm_provider import LLMError, get_llm, safe_invoke
from rag_app.core.prompts import format_context, parallel_sub_prompt, synthesis_prompt
from rag_app.core.tracing import Trace
from rag_app.core.vector_store import VectorStore
from rag_app.pipelines.query_generator import generate_subquestions

logger = logging.getLogger(__name__)


class ParallelDecompositionPipeline:
    """Mode 7: Parallel (independent) query decomposition."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: str = "gemini",
        llm_model: str | None = None,
        max_concurrency: int = 3,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_concurrency = max_concurrency

    def _process_subquestion(
        self,
        sub_q: str,
        top_k: int,
        llm_provider: str,
        llm_model: str | None,
    ) -> dict[str, Any]:
        """Process a single sub-question independently (runs in thread)."""
        docs = self.vector_store.similarity_search(query=sub_q, k=top_k)
        context = format_context(docs)

        try:
            llm = get_llm(provider=llm_provider, model=llm_model)
            messages = parallel_sub_prompt.format_messages(
                sub_question=sub_q,
                context=context,
            )
            answer = safe_invoke(llm, messages)
        except LLMError as exc:
            answer = f"[Failed: {exc}]"
            logger.error("Parallel sub-question failed: %s → %s", sub_q, exc)

        return {"sub_question": sub_q, "docs": docs, "context": context, "answer": answer}

    def run(
        self,
        query: str,
        top_k: int = 4,
        n_subquestions: int = 3,
        trace: Trace | None = None,
    ) -> dict[str, Any]:
        llm = get_llm(provider=self.llm_provider, model=self.llm_model)

        # ── Step 1: Decompose ─────────────────────────────────────────────────
        subquestions = generate_subquestions(query, llm, n_subquestions=n_subquestions)

        if trace:
            trace.record("decomposition", {"original": query, "sub_questions": subquestions})

        # ── Step 2: Parallel retrieval + answering ────────────────────────────
        results: list[dict[str, Any]] = [{}] * len(subquestions)  # preserve order
        failed_indices: list[int] = []

        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(subquestions))) as executor:
            future_to_idx = {
                executor.submit(
                    self._process_subquestion,
                    sub_q,
                    top_k,
                    self.llm_provider,
                    self.llm_model,
                ): idx
                for idx, sub_q in enumerate(subquestions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Sub-question %d raised unexpected error: %s", idx, exc)
                    results[idx] = {
                        "sub_question": subquestions[idx],
                        "docs": [],
                        "context": "",
                        "answer": f"[Unexpected error: {exc}]",
                    }
                    failed_indices.append(idx)

        if trace:
            for idx, r in enumerate(results):
                trace.record(
                    f"parallel_step_{idx+1}",
                    {
                        "sub_question": r.get("sub_question"),
                        "docs_count": len(r.get("docs", [])),
                        "answer_preview": r.get("answer", "")[:150],
                    },
                )

        # ── Step 3: Synthesize ────────────────────────────────────────────────
        sub_answers_str = "\n\n".join(
            f"Sub-question {i+1}: {r['sub_question']}\nAnswer: {r['answer']}"
            for i, r in enumerate(results)
        )
        synthesis_messages = synthesis_prompt.format_messages(
            original_question=query,
            sub_answers=sub_answers_str,
        )
        try:
            final_answer = safe_invoke(llm, synthesis_messages)
        except LLMError as exc:
            final_answer = f"❌ Synthesis failed: {exc}"

        if failed_indices:
            final_answer = (
                f"⚠️ Warning: Sub-questions {[i+1 for i in failed_indices]} failed. "
                f"Answer may be incomplete.\n\n{final_answer}"
            )

        if trace:
            trace.record("synthesis", {"answer_preview": final_answer[:200]})

        return {
            "answer": final_answer,
            "original_query": query,
            "subquestions": subquestions,
            "sub_results": results,
            "failed_indices": failed_indices,
            "max_concurrency": self.max_concurrency,
        }

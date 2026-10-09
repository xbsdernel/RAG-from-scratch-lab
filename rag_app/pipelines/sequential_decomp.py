"""
Mode 6: Sequential Query Decomposition.

Dependent step-by-step decomposition:
1. Decompose the question into ordered sub-questions.
2. For each sub-question: retrieve evidence → generate partial answer.
3. Each step receives the previous Q&A history as context.
4. A final synthesis step combines all partial answers.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from rag_app.core.llm_provider import LLMError, get_llm, safe_invoke
from rag_app.core.prompts import (
    format_context,
    seq_decomp_prompt,
    synthesis_prompt,
)
from rag_app.core.tracing import Trace
from rag_app.core.vector_store import VectorStore
from rag_app.pipelines.query_generator import generate_subquestions

logger = logging.getLogger(__name__)

MAX_HISTORY_PAIRS = 10  # configurable upper bound on Q&A history


class SequentialDecompositionPipeline:
    """Mode 6: Sequential (dependent) query decomposition."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: str = "gemini",
        llm_model: str | None = None,
        max_history: int = MAX_HISTORY_PAIRS,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_history = max_history

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

        # ── Step 2: Sequential retrieval + answering ──────────────────────────
        history: list[dict[str, str]] = []  # [{question, answer, context}, ...]
        failed_steps: list[int] = []

        for step_idx, sub_q in enumerate(subquestions):
            # Retrieve evidence for this sub-question
            docs = self.vector_store.similarity_search(query=sub_q, k=top_k)
            context = format_context(docs)

            # Build history string (bounded)
            history_bounded = history[-self.max_history :]
            history_str = "\n\n".join(
                f"Q{i+1}: {h['question']}\nA{i+1}: {h['answer']}"
                for i, h in enumerate(history_bounded)
            ) if history_bounded else "None yet."

            messages = seq_decomp_prompt.format_messages(
                original_question=query,
                history=history_str,
                sub_question=sub_q,
                context=context,
            )
            try:
                sub_answer = safe_invoke(llm, messages)
            except LLMError as exc:
                sub_answer = f"[Step {step_idx+1} failed: {exc}]"
                failed_steps.append(step_idx)
                logger.error("Sequential step %d failed: %s", step_idx + 1, exc)

            history.append({"question": sub_q, "answer": sub_answer, "context": context, "docs": docs})

            if trace:
                trace.record(
                    f"step_{step_idx+1}",
                    {
                        "sub_question": sub_q,
                        "retrieved_docs": len(docs),
                        "answer_preview": sub_answer[:200],
                        "failed": step_idx in failed_steps,
                    },
                )

        # ── Step 3: Synthesize ────────────────────────────────────────────────
        sub_answers_str = "\n\n".join(
            f"Sub-question {i+1}: {h['question']}\nAnswer: {h['answer']}"
            for i, h in enumerate(history)
        )
        synthesis_messages = synthesis_prompt.format_messages(
            original_question=query,
            sub_answers=sub_answers_str,
        )
        try:
            final_answer = safe_invoke(llm, synthesis_messages)
        except LLMError as exc:
            final_answer = f"❌ Synthesis failed: {exc}"
            logger.error("Synthesis failed: %s", exc)

        if failed_steps:
            final_answer = (
                f"⚠️ Warning: Steps {[s+1 for s in failed_steps]} encountered errors. "
                f"The answer below may be incomplete.\n\n{final_answer}"
            )

        if trace:
            trace.record("synthesis", {"final_answer_preview": final_answer[:200]})

        return {
            "answer": final_answer,
            "original_query": query,
            "subquestions": subquestions,
            "history": history,
            "failed_steps": failed_steps,
        }

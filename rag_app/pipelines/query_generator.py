"""
Query generation service.

Used by Multi-Query (Mode 4) and RAG Fusion (Mode 5).
Calls the LLM to produce alternative formulations of the user's question.
"""
from __future__ import annotations

import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel

from rag_app.core.llm_provider import LLMError, safe_invoke
from rag_app.core.prompts import multi_query_generation_prompt

logger = logging.getLogger(__name__)


def generate_alternative_queries(
    original_query: str,
    llm: BaseChatModel,
    n_queries: int = 3,
) -> list[str]:
    """
    Generate n_queries alternative reformulations of the original query.

    Returns:
        List of query strings. Always includes a cleaned version of the original
        as the last item if the LLM output cannot produce n_queries alternatives.
    """
    messages = multi_query_generation_prompt.format_messages(
        question=original_query,
        n_queries=n_queries,
    )
    try:
        raw = safe_invoke(llm, messages)
    except LLMError as exc:
        logger.error("Failed to generate alternative queries: %s", exc)
        return [original_query]  # fall back to original

    # Parse: one query per line, strip numbering / bullets
    queries: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*[\d\.\-\*\)]+\s*", "", line).strip()
        if line and len(line) > 5:
            queries.append(line)

    if not queries:
        logger.warning("Query generation returned no parseable queries. Using original.")
        return [original_query]

    # Trim to n_queries
    queries = queries[:n_queries]
    logger.info("Generated %d alternative queries.", len(queries))
    return queries


def generate_subquestions(
    original_query: str,
    llm: BaseChatModel,
    n_subquestions: int = 3,
) -> list[str]:
    """
    Decompose a complex question into n_subquestions independent sub-questions.
    Used by both Sequential (Mode 6) and Parallel (Mode 7) decomposition.
    """
    from rag_app.core.prompts import decompose_prompt  # noqa: PLC0415

    messages = decompose_prompt.format_messages(
        question=original_query,
        n_subquestions=n_subquestions,
    )
    try:
        raw = safe_invoke(llm, messages)
    except LLMError as exc:
        logger.error("Failed to decompose question: %s", exc)
        return [original_query]

    subqs: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*[\d\.\-\*\)]+\s*", "", line).strip()
        if line and len(line) > 5:
            subqs.append(line)

    if not subqs:
        logger.warning("Decomposition returned no parseable sub-questions. Using original.")
        return [original_query]

    subqs = subqs[:n_subquestions]
    logger.info("Decomposed into %d sub-questions.", len(subqs))
    return subqs

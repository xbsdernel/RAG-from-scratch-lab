"""
Prompt templates for all nine RAG modes.

All templates are kept in one place so they can be tested and iterated
independently of the pipeline code. Secrets are never embedded here.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ─── Mode 1: Basic RAG ────────────────────────────────────────────────────────

BASIC_RAG_SYSTEM = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided context documents. "
    "If the context does not contain sufficient information to answer the question, "
    "say so explicitly — do not invent facts. "
    "When possible, cite the source document name and page number."
)

BASIC_RAG_HUMAN = (
    "Context documents:\n"
    "─────────────────\n"
    "{context}\n"
    "─────────────────\n\n"
    "Question: {question}\n\n"
    "Answer (based only on the context above):"
)

basic_rag_prompt = ChatPromptTemplate.from_messages(
    [("system", BASIC_RAG_SYSTEM), ("human", BASIC_RAG_HUMAN)]
)

# ─── Multi-Query generation ───────────────────────────────────────────────────

MULTI_QUERY_SYSTEM = (
    "You are an expert at reformulating information retrieval queries. "
    "Your task is to generate {n_queries} different versions of a user's question "
    "to retrieve a broader set of relevant documents from a vector database. "
    "Each version should approach the question from a slightly different angle or "
    "use different vocabulary, but all must ask for the SAME information as the original. "
    "Do NOT decompose the question into sub-questions — rephrase it as a whole. "
    "Output ONLY the queries, one per line, with no numbering, bullets, or extra text."
)

MULTI_QUERY_HUMAN = "Original question: {question}"

multi_query_generation_prompt = ChatPromptTemplate.from_messages(
    [("system", MULTI_QUERY_SYSTEM), ("human", MULTI_QUERY_HUMAN)]
)

# ─── Query Decomposition ──────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = (
    "You are an expert at breaking down complex questions into simpler sub-questions "
    "that can each be answered independently using a document retrieval system. "
    "Generate {n_subquestions} sub-questions that together fully cover the original question. "
    "Output ONLY the sub-questions, one per line, with no numbering, bullets, or extra text."
)

DECOMPOSE_HUMAN = "Original question: {question}"

decompose_prompt = ChatPromptTemplate.from_messages(
    [("system", DECOMPOSE_SYSTEM), ("human", DECOMPOSE_HUMAN)]
)

# ─── Sequential decomposition — subquestion answering ─────────────────────────

SEQ_DECOMP_SYSTEM = (
    "You are a precise research assistant. Answer the current sub-question using ONLY "
    "the retrieved context provided. Cite sources when possible. "
    "If evidence is insufficient, say so. Do not invent facts."
)

SEQ_DECOMP_HUMAN = (
    "Original question: {original_question}\n\n"
    "Previous answers:\n{history}\n\n"
    "Current sub-question: {sub_question}\n\n"
    "Retrieved context:\n{context}\n\n"
    "Answer to current sub-question:"
)

seq_decomp_prompt = ChatPromptTemplate.from_messages(
    [("system", SEQ_DECOMP_SYSTEM), ("human", SEQ_DECOMP_HUMAN)]
)

# ─── Synthesis (Decomposition modes) ─────────────────────────────────────────

SYNTHESIS_SYSTEM = (
    "You are a skilled analyst. Synthesize the provided sub-question answers into "
    "a coherent, complete answer to the original question. "
    "Base your synthesis ONLY on the sub-answers provided. Do not add external knowledge."
)

SYNTHESIS_HUMAN = (
    "Original question: {original_question}\n\n"
    "Sub-question answers:\n{sub_answers}\n\n"
    "Synthesized final answer:"
)

synthesis_prompt = ChatPromptTemplate.from_messages(
    [("system", SYNTHESIS_SYSTEM), ("human", SYNTHESIS_HUMAN)]
)

# ─── Step-Back Prompting ──────────────────────────────────────────────────────

STEP_BACK_GENERATION_SYSTEM = (
    "You are an expert at abstractive thinking. Given a specific question, "
    "generate a more general, abstract question about the underlying principle, "
    "concept, or topic. The abstract question should help retrieve background "
    "knowledge that supports answering the specific question. "
    "Output ONLY the abstract question with no extra text."
)

STEP_BACK_GENERATION_HUMAN = "Specific question: {question}"

step_back_generation_prompt = ChatPromptTemplate.from_messages(
    [("system", STEP_BACK_GENERATION_SYSTEM), ("human", STEP_BACK_GENERATION_HUMAN)]
)

STEP_BACK_ANSWER_SYSTEM = (
    "You are a helpful assistant answering a question using two types of retrieved evidence:\n"
    "1. Background context (retrieved using an abstract step-back question)\n"
    "2. Specific context (retrieved using the original question)\n\n"
    "Answer the ORIGINAL specific question. Use both context types where relevant. "
    "Cite sources. Do not invent facts not present in the context."
)

STEP_BACK_ANSWER_HUMAN = (
    "Original question: {original_question}\n\n"
    "Background context (from step-back query):\n{background_context}\n\n"
    "Specific context (from original query):\n{specific_context}\n\n"
    "Answer:"
)

step_back_answer_prompt = ChatPromptTemplate.from_messages(
    [("system", STEP_BACK_ANSWER_SYSTEM), ("human", STEP_BACK_ANSWER_HUMAN)]
)

# ─── HyDE — Hypothetical Document Generation ─────────────────────────────────

HYDE_GENERATION_SYSTEM = (
    "You are a helpful assistant. Generate a hypothetical passage that would "
    "ideally answer the given question. The passage should be written as if it "
    "comes from an authoritative document. It is used only as a retrieval signal — "
    "it will NOT be presented to the user as factual evidence. "
    "Aim for 2-4 sentences. Output ONLY the passage with no preamble."
)

HYDE_GENERATION_HUMAN = "Question: {question}"

hyde_generation_prompt = ChatPromptTemplate.from_messages(
    [("system", HYDE_GENERATION_SYSTEM), ("human", HYDE_GENERATION_HUMAN)]
)

# ─── Final generation with real context (used after HyDE retrieval) ───────────

HYDE_ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "retrieved document excerpts below. These are real documents from the index — "
    "not hypothetical text. Cite sources. Do not invent facts."
)

HYDE_ANSWER_HUMAN = (
    "Question: {question}\n\n"
    "Retrieved context:\n{context}\n\n"
    "Answer:"
)

hyde_answer_prompt = ChatPromptTemplate.from_messages(
    [("system", HYDE_ANSWER_SYSTEM), ("human", HYDE_ANSWER_HUMAN)]
)

# ─── Parallel subquestion answering ──────────────────────────────────────────

PARALLEL_SUB_SYSTEM = (
    "You are a research assistant. Answer the following sub-question using ONLY "
    "the retrieved context provided. Cite sources. If evidence is insufficient, say so."
)

PARALLEL_SUB_HUMAN = (
    "Sub-question: {sub_question}\n\n"
    "Retrieved context:\n{context}\n\n"
    "Answer:"
)

parallel_sub_prompt = ChatPromptTemplate.from_messages(
    [("system", PARALLEL_SUB_SYSTEM), ("human", PARALLEL_SUB_HUMAN)]
)


def format_context(docs: list) -> str:
    """Format a list of LangChain Documents into a readable context string."""
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(f"[{i}] Source: {source} (page {page})\n{doc.page_content}")
    return "\n\n".join(parts)

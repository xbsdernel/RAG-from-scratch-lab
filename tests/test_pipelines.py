"""
Tests for pipelines using MOCKED LLMs and MOCKED vector stores.
No API keys required. No network needed.

Mocking strategy:
- LLM responses are mocked via unittest.mock.
- VectorStore is replaced with a lightweight in-memory fake.
"""
from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_app.core.chunker import split_documents
from rag_app.core.loaders import load_text
from rag_app.core.tracing import Tracer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_doc(chunk_id: str, content: str = "test content", source: str = "doc.txt", page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"chunk_id": chunk_id, "source": source, "page": page, "doc_id": "abc"},
    )


def fake_llm_response(text: str) -> MagicMock:
    """Create a mock LLM that returns a fixed string."""
    mock_llm = MagicMock()
    mock_response = AIMessage(content=text)
    mock_llm.invoke.return_value = mock_response
    return mock_llm


class FakeVectorStore:
    """In-memory fake vector store for unit tests."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def similarity_search(self, query: str, k: int = 5, filter_metadata=None) -> list[Document]:
        return self._docs[:k]

    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        return [(d, 0.1 * (i + 1)) for i, d in enumerate(self._docs[:k])]

    def similarity_search_by_vector(self, embedding: list[float], k: int = 5) -> list[Document]:
        return self._docs[:k]

    def list_sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self._docs:
            src = d.metadata.get("source", "?")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def total_chunks(self) -> int:
        return len(self._docs)

    def is_document_indexed(self, source_filename: str) -> bool:
        return source_filename in self.list_sources()

    def add_documents(self, chunks: list[Document]) -> dict[str, Any]:
        new = [c for c in chunks if c.metadata.get("chunk_id") not in {d.metadata.get("chunk_id") for d in self._docs}]
        self._docs.extend(new)
        return {"added": len(new), "skipped": len(chunks) - len(new)}

    def delete_document(self, source_filename: str) -> int:
        before = len(self._docs)
        self._docs = [d for d in self._docs if d.metadata.get("source") != source_filename]
        return before - len(self._docs)

    def clear_all(self) -> int:
        count = len(self._docs)
        self._docs = []
        return count


# ─── Indexing tests ───────────────────────────────────────────────────────────

class TestIndexingService:
    def test_index_new_file(self):
        from rag_app.core.indexing import IndexingService
        vs = FakeVectorStore([])
        svc = IndexingService(vector_store=vs)
        content = b"This is some content for indexing. " * 20
        result = svc.index_file(io.BytesIO(content), "new_doc.txt")
        assert result["status"] == "indexed"
        assert result["added"] > 0

    def test_skip_duplicate(self):
        from rag_app.core.indexing import IndexingService
        # Pre-populate with a document from same source
        existing_doc = make_doc("abc_0000", source="duplicate.txt")
        vs = FakeVectorStore([existing_doc])
        svc = IndexingService(vector_store=vs)
        content = b"Some content. " * 30
        result = svc.index_file(io.BytesIO(content), "duplicate.txt")
        assert result["status"] == "skipped"

    def test_force_reindex(self):
        from rag_app.core.indexing import IndexingService
        existing_doc = make_doc("abc_0000", source="doc.txt")
        vs = FakeVectorStore([existing_doc])
        svc = IndexingService(vector_store=vs)
        content = b"Updated content. " * 30
        result = svc.index_file(io.BytesIO(content), "doc.txt", force_reindex=True)
        assert result["status"] == "indexed"

    def test_empty_file_returns_error(self):
        from rag_app.core.indexing import IndexingService
        vs = FakeVectorStore([])
        svc = IndexingService(vector_store=vs)
        result = svc.index_file(io.BytesIO(b"   "), "empty.txt")
        assert result["status"] == "error"


# ─── Basic RAG pipeline tests ─────────────────────────────────────────────────

class TestBasicRAG:
    def test_returns_answer(self):
        from rag_app.pipelines.basic_rag import BasicRAGPipeline
        docs = [make_doc(f"chunk_{i}", f"Content about topic {i}") for i in range(3)]
        vs = FakeVectorStore(docs)
        pipeline = BasicRAGPipeline(vector_store=vs)

        with patch("rag_app.pipelines.basic_rag.get_llm") as mock_get_llm:
            mock_get_llm.return_value = fake_llm_response("This is the answer.")
            result = pipeline.run(query="What is the topic?", top_k=3)

        assert "answer" in result
        assert result["answer"] == "This is the answer."

    def test_retrieves_docs(self):
        from rag_app.pipelines.basic_rag import BasicRAGPipeline
        docs = [make_doc(f"chunk_{i}", f"Content {i}") for i in range(5)]
        vs = FakeVectorStore(docs)
        pipeline = BasicRAGPipeline(vector_store=vs)

        with patch("rag_app.pipelines.basic_rag.get_llm") as mock_get_llm:
            mock_get_llm.return_value = fake_llm_response("Answer")
            result = pipeline.run(query="query", top_k=3)

        assert len(result["retrieved_docs"]) == 3

    def test_trace_recorded(self):
        from rag_app.pipelines.basic_rag import BasicRAGPipeline
        docs = [make_doc("c1", "Some content")]
        vs = FakeVectorStore(docs)
        pipeline = BasicRAGPipeline(vector_store=vs)
        trace = Tracer().new_trace("basic_rag", "test query")

        with patch("rag_app.pipelines.basic_rag.get_llm") as mock_get_llm:
            mock_get_llm.return_value = fake_llm_response("Answer")
            pipeline.run(query="test query", top_k=1, trace=trace)

        event_names = [e.name for e in trace.events]
        assert "retrieval" in event_names
        assert "generation" in event_names


# ─── Multi-Query tests ────────────────────────────────────────────────────────

class TestMultiQuery:
    def test_merges_results(self):
        from rag_app.pipelines.multi_query import MultiQueryPipeline
        docs = [make_doc(f"chunk_{i}", f"Content {i}") for i in range(3)]
        vs = FakeVectorStore(docs)
        pipeline = MultiQueryPipeline(vector_store=vs)

        llm_mock = fake_llm_response("This is the answer.")
        # First call: generate queries, subsequent calls: generate answer
        llm_mock.invoke.side_effect = [
            AIMessage(content="Alternative query 1\nAlternative query 2"),
            AIMessage(content="Final answer"),
        ]

        with patch("rag_app.pipelines.multi_query.get_llm") as mock_get_llm, \
             patch("rag_app.pipelines.query_generator.safe_invoke") as mock_safe:
            mock_safe.return_value = "Alt query 1\nAlt query 2"
            mock_get_llm.return_value = fake_llm_response("Final answer")
            result = pipeline.run(query="original query", top_k=3, n_queries=2)

        assert "answer" in result
        assert "all_queries" in result
        assert len(result["all_queries"]) >= 1

    def test_deduplication(self):
        """All retrieved chunks should appear at most once in merged_docs."""
        from rag_app.pipelines.multi_query import MultiQueryPipeline
        # All queries return the same docs
        docs = [make_doc("dup_chunk", "Duplicate content")]
        vs = FakeVectorStore(docs)
        pipeline = MultiQueryPipeline(vector_store=vs)

        with patch("rag_app.pipelines.query_generator.safe_invoke") as mock_qi, \
             patch("rag_app.pipelines.multi_query.get_llm") as mock_llm:
            mock_qi.return_value = "Query B\nQuery C"
            mock_llm.return_value = fake_llm_response("Answer")
            result = pipeline.run(query="test", top_k=5, n_queries=2)

        # merged_docs should have the chunk only once
        chunk_ids = [d.metadata["chunk_id"] for d in result["merged_docs"]]
        assert len(chunk_ids) == len(set(chunk_ids))


# ─── Sequential decomposition tests ──────────────────────────────────────────

class TestSequentialDecomposition:
    def test_history_accumulates(self):
        from rag_app.pipelines.sequential_decomp import SequentialDecompositionPipeline
        docs = [make_doc("c1", "Some info")]
        vs = FakeVectorStore(docs)
        pipeline = SequentialDecompositionPipeline(vector_store=vs)

        with patch("rag_app.pipelines.query_generator.safe_invoke") as mock_qi, \
             patch("rag_app.pipelines.sequential_decomp.get_llm") as mock_llm, \
             patch("rag_app.pipelines.sequential_decomp.safe_invoke") as mock_si:
            mock_qi.return_value = "Sub Q 1\nSub Q 2"
            mock_llm.return_value = MagicMock()
            mock_si.side_effect = ["Answer 1", "Answer 2", "Final synthesis"]
            result = pipeline.run(query="complex question", top_k=1, n_subquestions=2)

        assert len(result["history"]) == 2
        assert result["history"][0]["answer"] == "Answer 1"
        assert result["history"][1]["answer"] == "Answer 2"

    def test_failed_step_reported(self):
        from rag_app.pipelines.sequential_decomp import SequentialDecompositionPipeline
        from rag_app.core.llm_provider import LLMError
        docs = [make_doc("c1", "Info")]
        vs = FakeVectorStore(docs)
        pipeline = SequentialDecompositionPipeline(vector_store=vs)

        with patch("rag_app.pipelines.query_generator.safe_invoke") as mock_qi, \
             patch("rag_app.pipelines.sequential_decomp.get_llm") as mock_llm, \
             patch("rag_app.pipelines.sequential_decomp.safe_invoke") as mock_si:
            mock_qi.return_value = "Sub Q 1\nSub Q 2"
            mock_llm.return_value = MagicMock()
            mock_si.side_effect = [LLMError("Quota exceeded"), "Answer 2", "Final"]
            result = pipeline.run(query="question", top_k=1, n_subquestions=2)

        assert len(result["failed_steps"]) > 0
        assert "Warning" in result["answer"] or "failed" in result["history"][0]["answer"].lower()


# ─── HyDE tests ───────────────────────────────────────────────────────────────

class TestHyDE:
    def test_uses_hypothetical_embedding_not_passage(self):
        """Final answer context must use real docs, not the hypothetical passage."""
        from rag_app.pipelines.hyde import HyDEPipeline

        real_doc = make_doc("real_chunk", "Real document content about AI.")
        vs = FakeVectorStore([real_doc])

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 384

        pipeline = HyDEPipeline(vector_store=vs, embeddings=mock_emb)
        HYPO_PASSAGE = "This is the hypothetical passage generated by the AI."

        with patch("rag_app.pipelines.hyde.get_llm") as mock_llm, \
             patch("rag_app.pipelines.hyde.safe_invoke") as mock_si:
            mock_llm.return_value = MagicMock()
            mock_si.side_effect = [HYPO_PASSAGE, "Real answer based on real docs."]
            result = pipeline.run(query="What is AI?", top_k=1)

        # Hypothetical passage should be recorded separately
        assert result["hypothetical_passage"] == HYPO_PASSAGE
        # Real context should NOT contain the hypothetical passage
        assert HYPO_PASSAGE not in result["real_context"]
        # Real docs should be in context
        assert "Real document content" in result["real_context"]

    def test_hypothetical_passage_not_in_final_context(self):
        """Critical: hypothetical passage must never appear as evidence."""
        from rag_app.pipelines.hyde import HyDEPipeline

        docs = [make_doc("r1", "Actual factual content")]
        vs = FakeVectorStore(docs)
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.0] * 384

        pipeline = HyDEPipeline(vector_store=vs, embeddings=mock_emb)

        with patch("rag_app.pipelines.hyde.get_llm") as mock_llm, \
             patch("rag_app.pipelines.hyde.safe_invoke") as mock_si:
            mock_llm.return_value = MagicMock()
            mock_si.side_effect = ["HYPOTHETICAL_UNIQUE_STRING", "Answer"]
            result = pipeline.run(query="query", top_k=1)

        assert "HYPOTHETICAL_UNIQUE_STRING" not in result["real_context"]


# ─── Step-Back tests ──────────────────────────────────────────────────────────

class TestStepBack:
    def test_dual_retrieval(self):
        from rag_app.pipelines.step_back import StepBackPipeline
        docs = [make_doc("d1", "Specific fact"), make_doc("d2", "General principle")]
        vs = FakeVectorStore(docs)
        pipeline = StepBackPipeline(vector_store=vs)

        with patch("rag_app.pipelines.step_back.get_llm") as mock_llm, \
             patch("rag_app.pipelines.step_back.safe_invoke") as mock_si:
            mock_llm.return_value = MagicMock()
            mock_si.side_effect = ["What is the general principle?", "Final answer"]
            result = pipeline.run(query="Specific question", top_k=2)

        assert result["step_back_query"] == "What is the general principle?"
        assert len(result["specific_docs"]) > 0
        assert len(result["background_docs"]) > 0

    def test_original_query_preserved(self):
        from rag_app.pipelines.step_back import StepBackPipeline
        docs = [make_doc("d1", "Content")]
        vs = FakeVectorStore(docs)
        pipeline = StepBackPipeline(vector_store=vs)

        with patch("rag_app.pipelines.step_back.get_llm") as mock_llm, \
             patch("rag_app.pipelines.step_back.safe_invoke") as mock_si:
            mock_llm.return_value = MagicMock()
            mock_si.side_effect = ["Abstract query", "Answer"]
            result = pipeline.run(query="Very specific question", top_k=1)

        assert result["original_query"] == "Very specific question"


# ─── Tracing tests ────────────────────────────────────────────────────────────

class TestTracing:
    def test_trace_records_events(self):
        tracer = Tracer()
        trace = tracer.new_trace("basic_rag", "test query")
        trace.record("step1", {"key": "value"})
        trace.record("step2", {"key2": "value2"})
        assert len(trace.events) == 2
        assert trace.events[0].name == "step1"

    def test_trace_finish(self):
        tracer = Tracer()
        trace = tracer.new_trace("test", "query")
        trace.finish(answer="The answer", error=None)
        assert trace.final_answer == "The answer"
        assert trace.total_elapsed_ms is not None
        assert trace.total_elapsed_ms >= 0

    def test_trace_to_dict(self):
        tracer = Tracer()
        trace = tracer.new_trace("test", "q")
        trace.record("evt", {"data": 123})
        trace.finish("answer")
        d = trace.to_dict()
        assert d["strategy"] == "test"
        assert d["query"] == "q"
        assert len(d["events"]) == 1

    def test_error_recorded(self):
        tracer = Tracer()
        trace = tracer.new_trace("test", "q")
        trace.record("bad_step", {}, error="Something went wrong")
        assert trace.events[0].error == "Something went wrong"

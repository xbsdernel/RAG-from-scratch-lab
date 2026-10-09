"""
Standalone test runner for pure-stdlib parts of the RAG app.
Tests: chunking logic (pure Python, uses langchain_text_splitters),
       loaders (stdlib io), tracing (stdlib).

Run with:
    python tests/run_stdlib_tests.py
"""
import io
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_passed = 0
_failed = 0


def test(name: str, fn):
    global _passed, _failed
    try:
        fn()
        print(f"  ✅ {name}")
        _passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        _failed += 1


# ─── Tracing Tests (zero external deps) ──────────────────────────────────────

def run_tracing_tests():
    print("\n─── Tracing Tests ───────────────────────────────────────────────────")
    from rag_app.core.tracing import Tracer

    def t_create():
        t = Tracer().new_trace("basic_rag", "test query")
        assert t.strategy == "basic_rag"
        assert t.query == "test query"
        assert isinstance(t.run_id, str) and len(t.run_id) > 0

    test("Create trace with strategy and query", t_create)

    def t_record():
        t = Tracer().new_trace("test", "q")
        t.record("step1", {"key": "value"})
        t.record("step2", {"key2": 42})
        assert len(t.events) == 2
        assert t.events[0].name == "step1"
        assert t.events[1].name == "step2"

    test("Record events sequentially", t_record)

    def t_finish():
        t = Tracer().new_trace("test", "q")
        t.finish(answer="The answer", error=None)
        assert t.final_answer == "The answer"
        assert t.total_elapsed_ms is not None
        assert t.total_elapsed_ms >= 0

    test("Finish trace with answer", t_finish)

    def t_error():
        t = Tracer().new_trace("test", "q")
        t.record("bad", {}, error="Something failed")
        assert t.events[0].error == "Something failed"

    test("Record error event", t_error)

    def t_to_dict():
        t = Tracer().new_trace("basic_rag", "query")
        t.record("evt", {"data": 123})
        t.finish("answer")
        d = t.to_dict()
        assert d["strategy"] == "basic_rag"
        assert d["query"] == "query"
        assert len(d["events"]) == 1
        assert d["final_answer"] == "answer"

    test("to_dict() returns correct structure", t_to_dict)

    def t_run_id_unique():
        tracer = Tracer()
        ids = {tracer.new_trace("t", "q").run_id for _ in range(10)}
        assert len(ids) == 10, "Run IDs should be unique"

    test("Run IDs are unique", t_run_id_unique)


# ─── RRF Tests (only needs langchain_core.documents) ─────────────────────────

def run_rrf_tests():
    print("\n─── RRF Tests ───────────────────────────────────────────────────────")
    try:
        from langchain_core.documents import Document
    except ImportError:
        print("  ⏭️ Skipping RRF tests — langchain_core not installed")
        return

    from rag_app.pipelines.rrf import reciprocal_rank_fusion

    def doc(chunk_id: str) -> Document:
        return Document(page_content="test", metadata={"chunk_id": chunk_id, "source": "t.txt", "page": 1})

    def t_formula():
        results = reciprocal_rank_fusion([[doc("x")]], c=60)
        expected = 1.0 / (60 + 1)
        assert abs(results[0].rrf_score - expected) < 1e-9

    test("RRF formula: 1/(c+rank)", t_formula)

    def t_multi_list():
        list1, list2 = [doc("a"), doc("b")], [doc("a"), doc("c")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        a = next(r for r in results if r.chunk_id == "a")
        b = next(r for r in results if r.chunk_id == "b")
        assert a.rrf_score > b.rrf_score

    test("Doc in 2 lists scores higher than in 1", t_multi_list)

    def t_empty():
        assert reciprocal_rank_fusion([]) == []

    test("Empty input → empty output", t_empty)

    def t_ranks():
        results = reciprocal_rank_fusion([[doc("a"), doc("b"), doc("c")]])
        assert [r.final_rank for r in results] == [1, 2, 3]

    test("Final ranks are 1-based", t_ranks)

    def t_deterministic():
        r1 = reciprocal_rank_fusion([[doc("x")], [doc("y")]])
        r2 = reciprocal_rank_fusion([[doc("x")], [doc("y")]])
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]

    test("Ties resolved deterministically", t_deterministic)

    def t_aggregated_sum():
        list1, list2 = [doc("a"), doc("b")], [doc("b"), doc("a")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        a = next(r for r in results if r.chunk_id == "a")
        expected = 1/(60+1) + 1/(60+2)
        assert abs(a.rrf_score - expected) < 1e-9

    test("Aggregated score = sum of per-list contributions", t_aggregated_sum)

    def t_c_effect():
        d1, d2 = doc("top"), doc("low")
        lc = reciprocal_rank_fusion([[d1, d2]], c=1)
        hc = reciprocal_rank_fusion([[d1, d2]], c=1000)
        assert (lc[0].rrf_score - lc[1].rrf_score) > (hc[0].rrf_score - hc[1].rrf_score)

    test("Higher c → smaller rank score gaps", t_c_effect)


# ─── Loader Tests (stdlib only) ───────────────────────────────────────────────

def run_loader_tests():
    print("\n─── Loader Tests ────────────────────────────────────────────────────")
    try:
        from langchain_core.documents import Document
    except ImportError:
        print("  ⏭️ Skipping loader tests — langchain_core not installed")
        return

    from rag_app.core.loaders import load_text, load_document, SUPPORTED_EXTENSIONS

    def t_utf8():
        f = io.BytesIO(b"Hello UTF-8 content")
        docs = load_text(f, "test.txt")
        assert "Hello" in docs[0].page_content

    test("Load UTF-8 text", t_utf8)

    def t_arabic():
        content = "مرحبا بالعالم"
        f = io.BytesIO(content.encode("utf-8"))
        docs = load_text(f, "arabic.txt")
        assert "مرحبا" in docs[0].page_content

    test("Load Arabic text", t_arabic)

    def t_metadata_source():
        f = io.BytesIO(b"Content")
        docs = load_text(f, "my_file.txt")
        assert docs[0].metadata["source"] == "my_file.txt"

    test("Metadata source preserved", t_metadata_source)

    def t_stable_doc_id():
        f1 = io.BytesIO(b"Content A")
        f2 = io.BytesIO(b"Content B")
        d1 = load_text(f1, "same.txt")[0].metadata["doc_id"]
        d2 = load_text(f2, "same.txt")[0].metadata["doc_id"]
        assert d1 == d2

    test("Same filename → same doc_id", t_stable_doc_id)

    def t_empty_raises():
        f = io.BytesIO(b"   \n  ")
        try:
            load_text(f, "empty.txt")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    test("Empty file raises ValueError", t_empty_raises)

    def t_md_type():
        f = io.BytesIO(b"# Title\n\nContent")
        docs = load_text(f, "readme.md")
        assert docs[0].metadata["file_type"] == "markdown"

    test("Markdown file_type detected", t_md_type)

    def t_unsupported_raises():
        f = io.BytesIO(b"data")
        try:
            load_document(f, "file.xyz")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    test("Unsupported extension raises ValueError", t_unsupported_raises)

    def t_supported_extensions():
        for ext in [".pdf", ".docx", ".txt", ".md"]:
            assert ext in SUPPORTED_EXTENSIONS

    test("All expected extensions in SUPPORTED_EXTENSIONS", t_supported_extensions)


# ─── Chunker Tests ────────────────────────────────────────────────────────────

def run_chunker_tests():
    print("\n─── Chunker Tests ───────────────────────────────────────────────────")
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa
    except ImportError:
        print("  ⏭️ Skipping chunker tests — langchain_core / langchain_text_splitters not installed")
        return

    from rag_app.core.chunker import split_documents, describe_chunks

    def make(text, doc_id="d1", source="s.txt"):
        return Document(page_content=text, metadata={"source": source, "doc_id": doc_id, "page": 1, "file_type": "txt"})

    def t_basic():
        doc = make("A " * 500)
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    test("Long text splits into multiple chunks", t_basic)

    def t_metadata_preserved():
        doc = make("Hello " * 50, doc_id="myid", source="my.txt")
        chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)
        for c in chunks:
            assert c.metadata["source"] == "my.txt"
            assert c.metadata["doc_id"] == "myid"

    test("Source and doc_id preserved in chunks", t_metadata_preserved)

    def t_chunk_id():
        doc = make("X " * 100, doc_id="xid")
        chunks = split_documents([doc], chunk_size=80, chunk_overlap=0)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    test("Chunk IDs are unique", t_chunk_id)

    def t_chunk_id_prefix():
        doc = make("A " * 60, doc_id="myid")
        chunks = split_documents([doc], chunk_size=100, chunk_overlap=0)
        for c in chunks:
            assert c.metadata["chunk_id"].startswith("myid_")

    test("Chunk ID starts with doc_id", t_chunk_id_prefix)

    def t_empty():
        chunks = split_documents([], chunk_size=200, chunk_overlap=0)
        assert chunks == []

    test("Empty input returns empty list", t_empty)

    def t_small():
        doc = make("Short.", doc_id="sid")
        chunks = split_documents([doc], chunk_size=1000, chunk_overlap=50)
        assert len(chunks) == 1

    test("Short document → single chunk", t_small)

    def t_describe():
        doc = make("Word " * 100, source="s.txt")
        chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)
        info = describe_chunks(chunks)
        assert info["count"] == len(chunks)
        assert "s.txt" in info["sources"]

    test("describe_chunks returns correct summary", t_describe)

    def t_sequential_index():
        doc = make("W " * 200, doc_id="sq")
        chunks = split_documents([doc], chunk_size=100, chunk_overlap=0)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    test("Chunk indices are sequential", t_sequential_index)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Application — Standalone Test Runner")
    print("=" * 60)

    run_tracing_tests()
    run_rrf_tests()
    run_loader_tests()
    run_chunker_tests()

    print(f"\n{'='*60}")
    print(f"  TOTAL: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(0 if _failed == 0 else 1)

"""
Streamlit application entry point.

Run with:
    streamlit run app.py

Implements all nine RAG modes in a single tabbed interface.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st

# ─── Page config must be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="RAG From Scratch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Imports ──────────────────────────────────────────────────────────────────
from rag_app.core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    MAX_UPLOAD_SIZE_MB,
    validate,
)
from rag_app.core.indexing import IndexingService
from rag_app.core.tracing import Tracer

# ─── Strategy labels ──────────────────────────────────────────────────────────
STRATEGIES = {
    "1 — Basic RAG": "basic_rag",
    "2 — Indexing & Retrieval Inspection": "inspection",
    "3 — Generation & Prompt Inspection": "generation_inspection",
    "4 — Multi-Query Retrieval": "multi_query",
    "5 — RAG Fusion (RRF)": "rag_fusion",
    "6 — Sequential Query Decomposition": "sequential",
    "7 — Parallel Query Decomposition": "parallel",
    "8 — Step-Back Prompting": "step_back",
    "9 — Hypothetical Document Embeddings (HyDE)": "hyde",
}

tracer = Tracer()


# ─── Cached resource getters ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~120 MB)…")
def _get_vector_store():
    from rag_app.core.vector_store import get_vector_store
    return get_vector_store()


@st.cache_resource(show_spinner="Initializing embedding model…")
def _get_embeddings():
    from rag_app.core.embeddings import get_embeddings
    return get_embeddings()


def get_indexing_service() -> IndexingService:
    return IndexingService(vector_store=_get_vector_store())


# ─── Session state helpers ────────────────────────────────────────────────────

def init_session() -> None:
    defaults = {
        "chat_history": [],
        "last_trace": None,
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict[str, Any]:
    st.sidebar.title("🔍 RAG From Scratch")
    st.sidebar.markdown("---")

    # Config warnings
    warnings = validate()
    for w in warnings:
        st.sidebar.warning(w)

    st.sidebar.subheader("⚙️ Settings")

    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        ["gemini", "ollama", "groq"],
        index=0,
        help="Switch providers without changing the pipeline code.",
    )

    top_k = st.sidebar.slider("Top-K (retrieved chunks)", min_value=1, max_value=20, value=DEFAULT_TOP_K)

    strategy_label = st.sidebar.selectbox(
        "RAG Strategy",
        list(STRATEGIES.keys()),
        index=0,
    )
    strategy = STRATEGIES[strategy_label]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Chunking Settings")
    chunk_size = st.sidebar.number_input("Chunk size (chars)", 128, 4096, DEFAULT_CHUNK_SIZE, step=64)
    chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", 0, 512, DEFAULT_CHUNK_OVERLAP, step=16)

    # Strategy-specific options
    extra: dict[str, Any] = {}
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Strategy Options")

    if strategy in ("multi_query", "rag_fusion"):
        extra["n_queries"] = st.sidebar.slider("Alternative queries", 2, 6, 3)
    if strategy == "rag_fusion":
        extra["rrf_c"] = st.sidebar.number_input("RRF constant (c)", 10, 200, 60)
    if strategy in ("sequential", "parallel"):
        extra["n_subquestions"] = st.sidebar.slider("Sub-questions", 2, 6, 3)
    if strategy == "parallel":
        extra["max_concurrency"] = st.sidebar.slider("Max concurrency", 1, 5, 3)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**Embedding model:** `{EMBEDDING_MODEL}`  \n"
        f"**LLM model:** `{GEMINI_MODEL}`  \n"
        "⚠️ Do not change embedding model after indexing."
    )

    return {
        "strategy": strategy,
        "strategy_label": strategy_label,
        "llm_provider": llm_provider,
        "top_k": top_k,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        **extra,
    }


# ─── Document Management Tab ──────────────────────────────────────────────────

def render_document_management(settings: dict[str, Any]) -> None:
    st.header("📂 Document Management")

    vs = _get_vector_store()
    svc = get_indexing_service()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload & Index Documents")
        uploaded_files = st.file_uploader(
            f"Upload PDF, DOCX, TXT, or Markdown (max {MAX_UPLOAD_SIZE_MB} MB each)",
            type=["pdf", "docx", "txt", "md", "markdown"],
            accept_multiple_files=True,
        )

        force_reindex = st.checkbox("Force re-index (replaces existing chunks)", value=False)

        if uploaded_files and st.button("📥 Index Selected Files", type="primary"):
            for uf in uploaded_files:
                # Size check
                content = uf.read()
                size_mb = len(content) / (1024 * 1024)
                if size_mb > MAX_UPLOAD_SIZE_MB:
                    st.error(f"❌ {uf.name}: File too large ({size_mb:.1f} MB > {MAX_UPLOAD_SIZE_MB} MB limit)")
                    continue

                with st.spinner(f"Indexing {uf.name}…"):
                    result = svc.index_file(
                        file_like=io.BytesIO(content),
                        filename=uf.name,
                        chunk_size=settings["chunk_size"],
                        chunk_overlap=settings["chunk_overlap"],
                        force_reindex=force_reindex,
                    )

                if result["status"] == "indexed":
                    st.success(
                        f"✅ **{uf.name}**: {result['added']} chunks added "
                        f"({result['skipped']} skipped as duplicates) "
                        f"from {result['raw_pages']} page(s)"
                    )
                elif result["status"] == "skipped":
                    st.info(f"⏭️ **{uf.name}**: {result['reason']}")
                else:
                    st.error(f"❌ **{uf.name}**: {result['reason']}")

    with col2:
        st.subheader("Indexed Documents")
        sources = vs.list_sources()
        total = vs.total_chunks()
        st.metric("Total Chunks", total)

        if sources:
            for src, count in sorted(sources.items()):
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.text(src)
                c2.text(f"{count} chunks")
                if c3.button("🗑️", key=f"del_{src}", help=f"Delete {src}"):
                    deleted = vs.delete_document(src)
                    st.success(f"Deleted {deleted} chunks for '{src}'")
                    st.rerun()
        else:
            st.info("No documents indexed yet. Upload files to get started.")

        st.markdown("---")
        if st.button("🧹 Clear Entire Index", type="secondary"):
            if st.session_state.get("confirm_clear"):
                deleted = vs.clear_all()
                st.warning(f"Index cleared — {deleted} chunks deleted.")
                st.session_state["confirm_clear"] = False
                st.rerun()
            else:
                st.session_state["confirm_clear"] = True
                st.warning("Click again to confirm clearing the index.")

    # Explain update behavior
    with st.expander("ℹ️ How document updates work"):
        st.markdown(
            """
            - **First index**: All chunks are added. The document is marked as indexed.
            - **Re-upload same file**: Skipped automatically (duplicate detection by filename).
            - **To update a document**: Check **Force re-index**. Old chunks are deleted, then new ones are added.
            - **Changing chunk size**: Old chunks at the old size remain until you force re-index.
            - **Never silently delete**: The index persists across app restarts.
            """
        )


# ─── Chat Tab ─────────────────────────────────────────────────────────────────

def render_chat(settings: dict[str, Any]) -> None:
    st.header(f"💬 Chat — {settings['strategy_label']}")

    vs = _get_vector_store()
    embs = _get_embeddings()
    total = vs.total_chunks()

    if total == 0:
        st.warning("⚠️ No documents indexed yet. Go to the **Document Management** tab to upload and index files.")
        return

    st.caption(f"📚 {total} chunks indexed | Using strategy: **{settings['strategy_label']}**")

    # Chat history display
    for msg in st.session_state.chat_history:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

    # Input
    query = st.chat_input("Ask a question about your documents…")

    if query:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                t_start = time.time()
                trace = tracer.new_trace(strategy=settings["strategy"], query=query)
                result = run_strategy(query=query, settings=settings, trace=trace, vs=vs, embs=embs)
                elapsed = time.time() - t_start
                trace.finish(answer=result.get("answer", ""), error=result.get("error"))

            answer = result.get("answer", "No answer returned.")
            st.markdown(answer)
            st.caption(f"⏱️ {elapsed:.2f}s | Run ID: `{trace.run_id}`")

            # Store trace/result for inspection
            st.session_state.last_trace = trace
            st.session_state.last_result = result

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Clear conversation button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.last_trace = None
            st.session_state.last_result = None
            st.rerun()


def run_strategy(
    query: str,
    settings: dict[str, Any],
    trace: Any,
    vs: Any,
    embs: Any,
) -> dict[str, Any]:
    """Dispatch query to the appropriate pipeline based on strategy."""
    strategy = settings["strategy"]
    provider = settings["llm_provider"]
    top_k = settings["top_k"]

    if strategy in ("basic_rag", "inspection", "generation_inspection"):
        from rag_app.pipelines.basic_rag import BasicRAGPipeline
        pipeline = BasicRAGPipeline(vector_store=vs, llm_provider=provider)
        return pipeline.run(query=query, top_k=top_k, trace=trace)

    if strategy == "multi_query":
        from rag_app.pipelines.multi_query import MultiQueryPipeline
        pipeline = MultiQueryPipeline(vector_store=vs, llm_provider=provider)
        return pipeline.run(query=query, top_k=top_k, n_queries=settings.get("n_queries", 3), trace=trace)

    if strategy == "rag_fusion":
        from rag_app.pipelines.rag_fusion import RAGFusionPipeline
        pipeline = RAGFusionPipeline(
            vector_store=vs, llm_provider=provider, rrf_c=settings.get("rrf_c", 60)
        )
        return pipeline.run(query=query, top_k=top_k, n_queries=settings.get("n_queries", 4), trace=trace)

    if strategy == "sequential":
        from rag_app.pipelines.sequential_decomp import SequentialDecompositionPipeline
        pipeline = SequentialDecompositionPipeline(vector_store=vs, llm_provider=provider)
        return pipeline.run(query=query, top_k=top_k, n_subquestions=settings.get("n_subquestions", 3), trace=trace)

    if strategy == "parallel":
        from rag_app.pipelines.parallel_decomp import ParallelDecompositionPipeline
        pipeline = ParallelDecompositionPipeline(
            vector_store=vs,
            llm_provider=provider,
            max_concurrency=settings.get("max_concurrency", 3),
        )
        return pipeline.run(query=query, top_k=top_k, n_subquestions=settings.get("n_subquestions", 3), trace=trace)

    if strategy == "step_back":
        from rag_app.pipelines.step_back import StepBackPipeline
        pipeline = StepBackPipeline(vector_store=vs, llm_provider=provider)
        return pipeline.run(query=query, top_k=top_k, trace=trace)

    if strategy == "hyde":
        from rag_app.pipelines.hyde import HyDEPipeline
        pipeline = HyDEPipeline(vector_store=vs, embeddings=embs, llm_provider=provider)
        return pipeline.run(query=query, top_k=top_k, trace=trace)

    return {"answer": f"Unknown strategy: {strategy}"}


# ─── Inspection Tab ───────────────────────────────────────────────────────────

def render_inspection(settings: dict[str, Any]) -> None:
    st.header("🔬 Retrieval & Process Inspection")

    result = st.session_state.get("last_result")
    trace = st.session_state.get("last_trace")

    if not result:
        st.info("Run a query in the Chat tab first to see inspection details here.")
        return

    strategy = settings["strategy"]

    # ── Trace events ──────────────────────────────────────────────────────────
    if trace:
        st.subheader(f"🕵️ Trace — Run `{trace.run_id}` | Strategy: `{trace.strategy}`")
        st.caption(f"Total time: {trace.total_elapsed_ms:.0f} ms")
        for event in trace.events:
            with st.expander(f"📍 {event.name}" + (f" ⚠️ ERROR" if event.error else ""), expanded=False):
                if event.error:
                    st.error(event.error)
                if event.elapsed_ms:
                    st.caption(f"⏱️ {event.elapsed_ms:.0f} ms")
                st.json(event.to_dict()["data"])

    st.markdown("---")

    # ── Strategy-specific inspection ──────────────────────────────────────────
    if strategy in ("basic_rag", "inspection", "generation_inspection"):
        _inspect_basic(result, strategy)
    elif strategy == "multi_query":
        _inspect_multi_query(result)
    elif strategy == "rag_fusion":
        _inspect_rag_fusion(result)
    elif strategy == "sequential":
        _inspect_sequential(result)
    elif strategy == "parallel":
        _inspect_parallel(result)
    elif strategy == "step_back":
        _inspect_step_back(result)
    elif strategy == "hyde":
        _inspect_hyde(result)


def _inspect_basic(result: dict, strategy: str) -> None:
    if strategy == "generation_inspection":
        st.subheader("📝 Full Formatted Prompt")
        st.code(result.get("prompt", ""), language="text")

    docs_with_scores = result.get("retrieved_with_scores", [])
    docs = result.get("retrieved_docs", [])

    st.subheader(f"📄 Retrieved Chunks ({len(docs)})")

    if strategy == "inspection":
        st.info(
            "**About retrieval scores:** Chroma returns L2 (Euclidean) distances. "
            "Lower values = more similar. These are NOT calibrated probability scores — "
            "do not interpret them as confidence percentages."
        )

    for i, doc in enumerate(docs):
        score = docs_with_scores[i][1] if i < len(docs_with_scores) else None
        label = f"Chunk {i+1}: {doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
        if score is not None:
            label += f" | L2 dist: {score:.4f}"
        with st.expander(label):
            st.markdown(f"**Chunk ID:** `{doc.metadata.get('chunk_id', 'N/A')}`")
            st.markdown(f"**Size:** {len(doc.page_content)} chars")
            st.markdown("**Content:**")
            st.text(doc.page_content)
            st.json(doc.metadata)


def _inspect_multi_query(result: dict) -> None:
    st.subheader("🔀 Generated Query Variations")
    for i, q in enumerate(result.get("all_queries", [])):
        label = "🎯 Original" if i == 0 else f"🔄 Variation {i}"
        st.markdown(f"**{label}:** {q}")

    st.subheader("📚 Per-Query Results")
    per_q = result.get("per_query_results", {})
    for q, docs in per_q.items():
        with st.expander(f"Query: {q[:60]}… → {len(docs)} chunks"):
            for j, doc in enumerate(docs):
                st.markdown(f"  **{j+1}.** `{doc.metadata.get('chunk_id')}` — {doc.page_content[:100]}…")

    merged = result.get("merged_docs", [])
    st.subheader(f"🔗 Merged Unique Chunks: {len(merged)}")
    for doc in merged:
        st.caption(f"✅ {doc.metadata.get('chunk_id')} | {doc.metadata.get('source')}")


def _inspect_rag_fusion(result: dict) -> None:
    st.subheader("🔀 Query Variations")
    for i, q in enumerate(result.get("all_queries", [])):
        label = "🎯 Original" if i == 0 else f"🔄 Variation {i}"
        st.markdown(f"**{label}:** {q}")

    st.subheader(f"📊 RRF Rankings (c={result.get('rrf_c', 60)})")
    st.info(
        "RRF score = Σ 1/(c + rank). Higher score = better combined rank. "
        "Documents appearing in multiple result lists score higher."
    )
    rrf_results = result.get("top_rrf", [])
    for r in rrf_results:
        d = r.to_dict()
        with st.expander(f"#{d['final_rank']} — RRF: {d['rrf_score']:.5f} | {d['source']} p.{d['page']}"):
            st.markdown(f"**Chunk ID:** `{d['chunk_id']}`")
            st.markdown(f"**Rank per query:** {d['ranks_per_query']}")
            st.markdown(f"**Preview:** {d['preview']}")


def _inspect_sequential(result: dict) -> None:
    st.subheader("📋 Sequential Q&A History")
    history = result.get("history", [])
    failed = result.get("failed_steps", [])
    for i, h in enumerate(history):
        status = "⚠️ FAILED" if i in failed else "✅"
        with st.expander(f"{status} Step {i+1}: {h['question'][:60]}…"):
            st.markdown(f"**Sub-question:** {h['question']}")
            st.markdown(f"**Answer:** {h['answer']}")
            st.markdown(f"**Evidence chunks:** {len(h.get('docs', []))}")
            for doc in h.get("docs", []):
                st.caption(f"  • {doc.metadata.get('chunk_id')} | {doc.metadata.get('source')}")


def _inspect_parallel(result: dict) -> None:
    st.subheader("⚡ Parallel Sub-Questions")
    sub_results = result.get("sub_results", [])
    failed = result.get("failed_indices", [])
    for i, r in enumerate(sub_results):
        status = "⚠️ FAILED" if i in failed else "✅"
        with st.expander(f"{status} Sub-question {i+1}: {r.get('sub_question', '')[:60]}…"):
            st.markdown(f"**Sub-question:** {r.get('sub_question')}")
            st.markdown(f"**Answer:** {r.get('answer')}")
            st.markdown(f"**Evidence chunks:** {len(r.get('docs', []))}")

    st.caption(f"Max concurrency used: {result.get('max_concurrency', 'N/A')}")


def _inspect_step_back(result: dict) -> None:
    st.subheader("🪜 Step-Back Query")
    st.markdown(f"**Original:** {result.get('original_query')}")
    st.markdown(f"**Step-back (abstract):** {result.get('step_back_query')}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Specific Retrieval")
        for doc in result.get("specific_docs", []):
            st.caption(f"• {doc.metadata.get('chunk_id')} | {doc.metadata.get('source')}")
    with col2:
        st.subheader("🌐 Background Retrieval")
        for doc in result.get("background_docs", []):
            st.caption(f"• {doc.metadata.get('chunk_id')} | {doc.metadata.get('source')}")

    only_bg = result.get("background_only_docs", [])
    if only_bg:
        st.info(f"📌 {len(only_bg)} chunks retrieved only by the step-back query (not in specific results).")


def _inspect_hyde(result: dict) -> None:
    st.subheader("🔮 Hypothetical Passage (AI-generated, NOT factual evidence)")
    st.warning(
        "⚠️ This passage was generated by the AI as a retrieval signal only. "
        "It has NOT been verified and is NOT included in the final answer's evidence. "
        "Do not treat it as a factual source."
    )
    st.markdown(f"> {result.get('hypothetical_passage', 'N/A')}")

    st.subheader("📄 Real Retrieved Documents")
    st.success("These are the actual indexed documents retrieved using the hypothetical passage's embedding.")
    for i, doc in enumerate(result.get("retrieved_real_docs", [])):
        with st.expander(f"Real Doc {i+1}: {doc.metadata.get('source')} p.{doc.metadata.get('page')}"):
            st.text(doc.page_content)
            st.json(doc.metadata)


# ─── Mode 2: Indexing Inspection ──────────────────────────────────────────────

def render_indexing_inspection() -> None:
    st.header("🔬 Mode 2: Indexing & Retrieval Inspection")

    vs = _get_vector_store()
    sources = vs.list_sources()

    st.subheader("Index Summary")
    st.metric("Total indexed chunks", vs.total_chunks())
    st.metric("Unique sources", len(sources))

    if not sources:
        st.info("No documents indexed yet.")
        return

    st.subheader("Source Breakdown")
    for src, count in sorted(sources.items()):
        st.markdown(f"- **{src}**: {count} chunks")

    st.subheader("Ad-hoc Retrieval Test")
    test_query = st.text_input("Enter a test query to see what gets retrieved:")
    test_k = st.slider("K", 1, 20, 5, key="inspect_k")

    if test_query:
        with st.spinner("Retrieving…"):
            docs_scores = vs.similarity_search_with_score(query=test_query, k=test_k)

        st.markdown(
            "**Score note:** Chroma L2 distance — lower values = more similar to query. "
            "NOT a calibrated probability."
        )
        for i, (doc, score) in enumerate(docs_scores):
            with st.expander(
                f"Rank {i+1} | L2={score:.4f} | {doc.metadata.get('source')} p.{doc.metadata.get('page')}"
            ):
                st.markdown(f"**Chunk ID:** `{doc.metadata.get('chunk_id')}`")
                st.markdown(f"**Size:** {len(doc.page_content)} chars")
                st.text(doc.page_content)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    init_session()
    settings = render_sidebar()

    # Navigation tabs
    tab_chat, tab_docs, tab_inspect, tab_index_inspect = st.tabs(
        ["💬 Chat", "📂 Documents", "🔬 Inspect Last Run", "🧪 Index Inspection (Mode 2)"]
    )

    with tab_chat:
        render_chat(settings)

    with tab_docs:
        render_document_management(settings)

    with tab_inspect:
        render_inspection(settings)

    with tab_index_inspect:
        render_indexing_inspection()


if __name__ == "__main__":
    main()

"""
Text splitting / chunking with metadata preservation.

Uses LangChain's RecursiveCharacterTextSplitter.
Each chunk inherits its parent document's metadata plus:
  - chunk_index: position within the document
  - chunk_id:    stable unique ID (doc_id + chunk_index)
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def split_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Args:
        documents: Raw documents from loaders.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    chunks: list[Document] = []

    # Group by doc_id so chunk_index resets per document
    by_doc: dict[str, list[Document]] = {}
    for doc in documents:
        doc_id = doc.metadata.get("doc_id", "unknown")
        by_doc.setdefault(doc_id, []).append(doc)

    for doc_id, doc_pages in by_doc.items():
        page_chunks = splitter.split_documents(doc_pages)
        for idx, chunk in enumerate(page_chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = f"{doc_id}_{idx:04d}"
            chunk.metadata["chunk_size_actual"] = len(chunk.page_content)
            chunks.append(chunk)

    logger.info(
        "Split %d documents → %d chunks (chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def describe_chunks(chunks: list[Document]) -> dict:
    """Return summary statistics about a set of chunks (for the diagnostic UI)."""
    if not chunks:
        return {"count": 0}

    sizes = [len(c.page_content) for c in chunks]
    sources = list({c.metadata.get("source", "unknown") for c in chunks})
    return {
        "count": len(chunks),
        "sources": sources,
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "avg_chars": round(sum(sizes) / len(sizes), 1),
    }

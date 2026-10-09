"""
Indexing service: high-level orchestration of load → chunk → embed → store.
"""
from __future__ import annotations

import logging
from typing import Any, BinaryIO

from langchain_core.documents import Document

from rag_app.core.chunker import split_documents
from rag_app.core.loaders import load_document
from rag_app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Orchestrates the full indexing pipeline:
    1. Load raw document from file.
    2. Split into chunks with metadata.
    3. Add to persistent vector store (deduplication built-in).
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def index_file(
        self,
        file_like: BinaryIO,
        filename: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """
        Index a single file.

        Args:
            file_like: File-like object (binary mode).
            filename: Original filename (used for metadata and duplicate detection).
            chunk_size: Characters per chunk.
            chunk_overlap: Overlap characters between chunks.
            force_reindex: If True, delete existing chunks before indexing.

        Returns:
            Dict with indexing result details.
        """
        # Duplicate check
        if not force_reindex and self.vector_store.is_document_indexed(filename):
            logger.info("Document '%s' already indexed — skipping.", filename)
            return {
                "status": "skipped",
                "filename": filename,
                "reason": "Document already indexed. Use force_reindex=True to update.",
                "added": 0,
                "skipped": 0,
            }

        if force_reindex and self.vector_store.is_document_indexed(filename):
            deleted = self.vector_store.delete_document(filename)
            logger.info("Re-indexing: deleted %d old chunks for '%s'.", deleted, filename)

        # Load
        try:
            raw_docs: list[Document] = load_document(file_like, filename)
        except Exception as exc:
            logger.error("Failed to load '%s': %s", filename, exc)
            return {"status": "error", "filename": filename, "reason": str(exc), "added": 0, "skipped": 0}

        # Chunk
        chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            return {
                "status": "error",
                "filename": filename,
                "reason": "No text chunks produced from document.",
                "added": 0,
                "skipped": 0,
            }

        # Store
        result = self.vector_store.add_documents(chunks)
        return {
            "status": "indexed",
            "filename": filename,
            "raw_pages": len(raw_docs),
            "chunks_total": len(chunks),
            "added": result["added"],
            "skipped": result["skipped"],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

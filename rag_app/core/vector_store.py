"""
Vector store and indexing service using ChromaDB (persistent).

Responsibilities:
- Create / open the Chroma collection.
- Add document chunks, preventing duplicates via chunk_id.
- Retrieve top-k similar chunks for a given query embedding.
- List indexed sources and their chunk counts.
- Delete all chunks for a specific source document.
- Clear the entire index.

Duplicate detection strategy:
  Chunks are identified by metadata["chunk_id"] = "<doc_id>_<chunk_index>".
  Before adding, we check existing IDs and skip already-indexed chunks.
  To re-index a document, delete it first, then index again.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_app.core.embeddings import E5Embeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin wrapper around langchain-chroma's Chroma class.

    Uses chunk_id as the primary key for deduplication and
    similarity search with distance metadata.
    """

    def __init__(self, persist_directory: str, embeddings: E5Embeddings, collection_name: str = "rag_documents") -> None:
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.embeddings = embeddings

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info(
            "VectorStore opened: collection='%s' path='%s'",
            collection_name,
            persist_directory,
        )

    # ─── Indexing ─────────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[Document]) -> dict[str, Any]:
        """
        Add chunks to the vector store, skipping already-indexed chunks.

        Returns a dict with counts of added and skipped chunks.
        """
        if not chunks:
            return {"added": 0, "skipped": 0}

        # Determine which chunk_ids are already in the store
        existing_ids: set[str] = set()
        try:
            collection = self._store._collection
            all_ids = collection.get(include=[])["ids"]
            existing_ids = set(all_ids)
        except Exception:
            pass  # Empty collection — all are new

        new_chunks: list[Document] = []
        new_ids: list[str] = []
        skipped = 0

        for chunk in chunks:
            cid = chunk.metadata.get("chunk_id", "")
            if cid in existing_ids:
                skipped += 1
                continue
            new_chunks.append(chunk)
            new_ids.append(cid)

        if new_chunks:
            self._store.add_documents(documents=new_chunks, ids=new_ids)
            logger.info("Added %d chunks (%d skipped as duplicates).", len(new_chunks), skipped)
        else:
            logger.info("All %d chunks already indexed — nothing added.", skipped)

        return {"added": len(new_chunks), "skipped": skipped}

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[Document]:
        """Retrieve top-k chunks most similar to query."""
        return self._store.similarity_search(query=query, k=k, filter=filter_metadata)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve top-k chunks with distance scores.

        NOTE: Chroma returns L2 distance (lower = more similar).
        These are NOT calibrated confidence probabilities.
        """
        return self._store.similarity_search_with_score(query=query, k=k)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 5,
    ) -> list[Document]:
        """Retrieve top-k chunks most similar to a pre-computed embedding vector (for HyDE)."""
        return self._store.similarity_search_by_vector(embedding=embedding, k=k)

    # ─── Inspection ───────────────────────────────────────────────────────────

    def list_sources(self) -> dict[str, int]:
        """Return {source_filename: chunk_count} for all indexed documents."""
        try:
            collection = self._store._collection
            result = collection.get(include=["metadatas"])
            counts: dict[str, int] = {}
            for meta in result["metadatas"]:
                src = meta.get("source", "unknown")
                counts[src] = counts.get(src, 0) + 1
            return counts
        except Exception as exc:
            logger.error("Error listing sources: %s", exc)
            return {}

    def total_chunks(self) -> int:
        """Return total number of indexed chunks."""
        try:
            return self._store._collection.count()
        except Exception:
            return 0

    def is_document_indexed(self, source_filename: str) -> bool:
        """Check whether a document (by filename) is already in the index."""
        return source_filename in self.list_sources()

    # ─── Deletion ─────────────────────────────────────────────────────────────

    def delete_document(self, source_filename: str) -> int:
        """Delete all chunks from a specific source document. Returns chunks deleted."""
        try:
            collection = self._store._collection
            result = collection.get(include=["metadatas"], where={"source": source_filename})
            ids_to_delete = result["ids"]
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info("Deleted %d chunks for source '%s'.", len(ids_to_delete), source_filename)
            return len(ids_to_delete)
        except Exception as exc:
            logger.error("Error deleting document '%s': %s", source_filename, exc)
            return 0

    def clear_all(self) -> int:
        """Delete all chunks from the collection. Returns count deleted."""
        try:
            count = self.total_chunks()
            self._store._client.delete_collection(self.collection_name)
            # Re-create empty collection
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info("Cleared %d chunks from collection '%s'.", count, self.collection_name)
            return count
        except Exception as exc:
            logger.error("Error clearing index: %s", exc)
            return 0


# ─── Module-level singleton ────────────────────────────────────────────────────

_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the singleton VectorStore instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        from rag_app.core.config import CHROMA_DB_PATH, CHROMA_COLLECTION  # noqa: PLC0415
        from rag_app.core.embeddings import get_embeddings  # noqa: PLC0415

        embeddings = get_embeddings()
        _vector_store_instance = VectorStore(
            persist_directory=CHROMA_DB_PATH,
            embeddings=embeddings,
            collection_name=CHROMA_COLLECTION,
        )
    return _vector_store_instance

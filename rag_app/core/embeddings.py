"""
Embedding service wrapping HuggingFaceEmbeddings from langchain-huggingface.

Model: intfloat/multilingual-e5-small
- Supports English and Arabic (and 100+ other languages)
- Requires prefixes: "query: " for queries, "passage: " for documents
- First run downloads ~120 MB from HuggingFace (cached locally afterward)
- CPU inference only — no GPU required

IMPORTANT: Do not change EMBEDDING_MODEL after documents have been indexed.
The embedding vector space changes with the model, invalidating the stored index.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Singleton — we want exactly one model instance per process.
_embeddings_instance: "E5Embeddings | None" = None


class E5Embeddings(Embeddings):
    """
    Wrapper around HuggingFaceEmbeddings that enforces the
    'query: ' / 'passage: ' prefix convention required by E5 models.
    """

    def __init__(self, model_name: str) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415

        logger.info("Loading embedding model: %s (CPU — first run may download ~120 MB)", model_name)
        self.model_name = model_name

        # Document embeddings: prefix "passage: "
        self._doc_embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
        )
        # Query embeddings: prefix "query: "
        self._query_embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
        )
        logger.info("Embedding model loaded successfully.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages (adds 'passage: ' prefix)."""
        return self._doc_embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (adds 'query: ' prefix)."""
        return self._query_embedder.embed_query(text)

    def embed_text_as_query(self, text: str) -> list[float]:
        """Alias for embed_query — explicit for HyDE hypothetical passages."""
        return self.embed_query(text)


def get_embeddings(model_name: str | None = None) -> E5Embeddings:
    """Return the singleton E5Embeddings instance (lazy init)."""
    global _embeddings_instance
    from rag_app.core.config import EMBEDDING_MODEL  # noqa: PLC0415

    target_model = model_name or EMBEDDING_MODEL

    if _embeddings_instance is None:
        _embeddings_instance = E5Embeddings(model_name=target_model)
    elif _embeddings_instance.model_name != target_model:
        logger.warning(
            "Requested model '%s' differs from loaded '%s'. Using loaded model to preserve index compatibility.",
            target_model,
            _embeddings_instance.model_name,
        )
    return _embeddings_instance

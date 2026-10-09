"""
Application configuration loaded from environment variables / .env file.
All settings have sensible defaults so the app can start without a complete .env.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above rag_app/)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=False)


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool = False) -> bool:
    val = _get(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


# ─── LLM ──────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = _get("GOOGLE_API_KEY")
GEMINI_MODEL: str = _get("GEMINI_MODEL", "gemini-2.0-flash")

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# ─── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = _get("CHROMA_DB_PATH", str(_ROOT / "chroma_db"))
CHROMA_COLLECTION: str = _get("CHROMA_COLLECTION", "rag_documents")

# ─── File Upload ──────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = _get_int("MAX_UPLOAD_SIZE_MB", 50)
UPLOADS_DIR: Path = _ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ─── RAG defaults ─────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = _get_int("DEFAULT_TOP_K", 5)
DEFAULT_CHUNK_SIZE: int = _get_int("DEFAULT_CHUNK_SIZE", 512)
DEFAULT_CHUNK_OVERLAP: int = _get_int("DEFAULT_CHUNK_OVERLAP", 64)

# ─── LangSmith (optional) ─────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2: bool = _get_bool("LANGCHAIN_TRACING_V2", False)
LANGCHAIN_API_KEY: str = _get("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT: str = _get("LANGCHAIN_PROJECT", "rag-from-scratch")

# Apply LangSmith env vars so LangChain picks them up automatically
if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("LOG_LEVEL", "INFO").upper()


def validate() -> list[str]:
    """Return a list of configuration warnings (not errors) for the UI."""
    warnings: list[str] = []
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        warnings.append(
            "GOOGLE_API_KEY is not set. LLM calls will fail. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return warnings

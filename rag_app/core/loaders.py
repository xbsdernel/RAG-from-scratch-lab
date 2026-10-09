"""
Document loaders for PDF, DOCX, TXT, and Markdown files.

Each loader returns a list of LangChain Document objects with rich metadata:
  - source: original filename
  - doc_id:  stable identifier (derived from filename)
  - page:    page number when available, else 0
  - file_type: extension
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import BinaryIO

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _doc_id(filename: str) -> str:
    """Create a stable, filesystem-safe doc identifier from a filename."""
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def load_pdf(file_like: BinaryIO, filename: str) -> list[Document]:
    """Load a PDF using pypdf. Returns one Document per page."""
    try:
        import pypdf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pypdf is required: pip install pypdf") from exc

    reader = pypdf.PdfReader(file_like)
    if len(reader.pages) == 0:
        raise ValueError(f"PDF '{filename}' has no pages.")

    docs: list[Document] = []
    doc_id = _doc_id(filename)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            logger.debug("PDF page %d of '%s' has no extractable text — skipped.", page_num + 1, filename)
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "doc_id": doc_id,
                    "page": page_num + 1,
                    "file_type": "pdf",
                    "total_pages": len(reader.pages),
                },
            )
        )

    if not docs:
        raise ValueError(f"PDF '{filename}' contains no extractable text.")
    return docs


def load_docx(file_like: BinaryIO, filename: str) -> list[Document]:
    """Load a DOCX file using python-docx. Returns one Document for the full text."""
    try:
        import docx  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("python-docx is required: pip install python-docx") from exc

    doc = docx.Document(file_like)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError(f"DOCX '{filename}' contains no extractable text.")

    text = "\n\n".join(paragraphs)
    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "doc_id": _doc_id(filename),
                "page": 1,
                "file_type": "docx",
                "total_pages": 1,
            },
        )
    ]


def load_text(file_like: BinaryIO, filename: str) -> list[Document]:
    """Load a plain-text or Markdown file."""
    raw = file_like.read()
    # Try UTF-8, fall back to latin-1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    text = text.strip()
    if not text:
        raise ValueError(f"File '{filename}' is empty.")

    ext = Path(filename).suffix.lower().lstrip(".")
    file_type = "markdown" if ext in ("md", "markdown") else "txt"

    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "doc_id": _doc_id(filename),
                "page": 1,
                "file_type": file_type,
                "total_pages": 1,
            },
        )
    ]


SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
}


def load_document(file_like: BinaryIO, filename: str) -> list[Document]:
    """Dispatch to the appropriate loader based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_like, filename)
    if ext == ".docx":
        return load_docx(file_like, filename)
    if ext in (".txt", ".md", ".markdown"):
        return load_text(file_like, filename)
    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )

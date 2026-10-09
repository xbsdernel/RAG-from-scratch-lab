"""
Tests for document loaders.
No network, no LLM — uses in-memory file-like objects.
"""
import io
import pytest

from rag_app.core.loaders import load_document, load_text, SUPPORTED_EXTENSIONS


class TestTextLoader:
    def test_loads_utf8(self):
        content = "Hello, this is a UTF-8 text file.\nSecond line."
        f = io.BytesIO(content.encode("utf-8"))
        docs = load_text(f, "test.txt")
        assert len(docs) == 1
        assert "Hello" in docs[0].page_content

    def test_loads_arabic(self):
        content = "مرحبا بالعالم. هذا نص عربي."
        f = io.BytesIO(content.encode("utf-8"))
        docs = load_text(f, "arabic.txt")
        assert "مرحبا" in docs[0].page_content

    def test_metadata_source(self):
        f = io.BytesIO(b"Some content")
        docs = load_text(f, "myfile.txt")
        assert docs[0].metadata["source"] == "myfile.txt"

    def test_metadata_doc_id_stable(self):
        f1 = io.BytesIO(b"Content one")
        f2 = io.BytesIO(b"Content two")
        docs1 = load_text(f1, "same_name.txt")
        docs2 = load_text(f2, "same_name.txt")
        # Same filename → same doc_id
        assert docs1[0].metadata["doc_id"] == docs2[0].metadata["doc_id"]

    def test_empty_file_raises(self):
        f = io.BytesIO(b"   \n  ")
        with pytest.raises(ValueError, match="empty"):
            load_text(f, "empty.txt")

    def test_markdown_file_type(self):
        f = io.BytesIO(b"# Title\n\nSome content.")
        docs = load_text(f, "readme.md")
        assert docs[0].metadata["file_type"] == "markdown"

    def test_txt_file_type(self):
        f = io.BytesIO(b"Plain text.")
        docs = load_text(f, "data.txt")
        assert docs[0].metadata["file_type"] == "txt"

    def test_page_metadata(self):
        f = io.BytesIO(b"Content")
        docs = load_text(f, "file.txt")
        assert docs[0].metadata["page"] == 1


class TestDispatcher:
    def test_unsupported_extension_raises(self):
        f = io.BytesIO(b"content")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(f, "file.xyz")

    def test_txt_dispatched(self):
        f = io.BytesIO(b"Hello from txt")
        docs = load_document(f, "test.txt")
        assert len(docs) == 1

    def test_md_dispatched(self):
        f = io.BytesIO(b"# Heading\nContent")
        docs = load_document(f, "doc.md")
        assert len(docs) == 1

    def test_supported_extensions_dict(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS

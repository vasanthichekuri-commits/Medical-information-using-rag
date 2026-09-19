import pytest

from src.chunker import chunk_text


def test_chunk_text_preserves_content_with_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)
    assert chunks == ["abcdef", "efghij"]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=5, overlap=5)

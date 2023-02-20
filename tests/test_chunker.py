import pytest
from src.chunker import chunk_text, Chunk

def test_empty_string():
    assert chunk_text("") == []

def test_short_doc():
    text = "Hello world"
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    assert chunks[0].content == text or text in chunks[0].content

def test_chunks_cover_text():
    text = "word " * 3000
    chunks = chunk_text(text)
    assert len(chunks) > 1
    # Check no gap — each chunk should start before the previous ended
    for i in range(1, len(chunks)):
        assert chunks[i].start_char < chunks[i-1].end_char

def test_pdf_page_detection():
    text = "[PAGE 1]\nThis is page one.\n\n[PAGE 2]\nThis is page two."
    chunks = chunk_text(text, file_type="pdf")
    assert any(c.page_num is not None for c in chunks)

import pytest
from src.chunker import chunk_text, count_tokens, CHUNK_SIZE, OVERLAP


SAMPLE_TEXT = " ".join([f"word{i}" for i in range(2000)])  # ~2000 token-ish text


def test_chunk_count_is_reasonable():
    chunks = chunk_text(SAMPLE_TEXT, "txt")
    # Should produce multiple chunks
    assert len(chunks) >= 3


def test_chunk_token_size():
    chunks = chunk_text(SAMPLE_TEXT, "txt")
    for chunk in chunks:
        # Each chunk must not exceed CHUNK_SIZE tokens
        assert chunk.token_count <= CHUNK_SIZE


def test_overlap_content():
    """Last tokens of chunk N should appear in the beginning of chunk N+1."""
    chunks = chunk_text(SAMPLE_TEXT, "txt")
    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks to test overlap")

    # The tail of chunk 0 should share at least some words with the head of chunk 1
    words_end_of_first = set(chunks[0].content.split()[-OVERLAP:])
    words_start_of_second = set(chunks[1].content.split()[:OVERLAP])
    assert words_end_of_first & words_start_of_second, (
        "Expected overlap between consecutive chunks"
    )


def test_empty_text_returns_no_chunks():
    chunks = chunk_text("", "txt")
    assert chunks == []


def test_short_text_single_chunk():
    short = "This is a short document with just a few words."
    chunks = chunk_text(short, "txt")
    assert len(chunks) == 1
    assert short.strip() in chunks[0].content


def test_pdf_page_annotation():
    pdf_text = "[PAGE 1]\nFirst page content.\n\n[PAGE 2]\nSecond page content."
    chunks = chunk_text(pdf_text, "pdf")
    # At least one chunk should have a page_num
    page_nums = [c.page_num for c in chunks if c.page_num is not None]
    assert len(page_nums) > 0


def test_txt_no_page_annotation():
    chunks = chunk_text(SAMPLE_TEXT, "txt")
    for chunk in chunks:
        assert chunk.page_num is None


def test_char_offsets_are_ordered():
    chunks = chunk_text(SAMPLE_TEXT, "txt")
    for chunk in chunks:
        assert chunk.start_char < chunk.end_char


def test_count_tokens():
    text = "Hello world"
    count = count_tokens(text)
    assert count >= 2  # at minimum 2 tokens for 2 words

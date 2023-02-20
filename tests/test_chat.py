import pytest
from src.chat import _build_context
from src.chunker import Chunk

def _chunk(text):
    return Chunk(content=text, start_char=0, end_char=len(text))

def test_build_context_respects_budget():
    chunks = [_chunk("word " * 100) for _ in range(10)]
    ctx = _build_context(chunks, budget=50)
    assert len(ctx) < len("word " * 1000)

def test_build_context_empty():
    assert _build_context([], 1000) == ""

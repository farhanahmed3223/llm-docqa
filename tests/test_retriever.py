import json
import math
import pytest
from unittest.mock import patch, MagicMock

from src.retriever import _cosine_similarity, retrieve_relevant_chunks, ScoredChunk


# ---------------------------------------------------------------------------
# Pure math tests — no mocks needed
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert math.isclose(_cosine_similarity(v, v), 1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert math.isclose(_cosine_similarity(a, b), 0.0)


def test_cosine_similarity_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert math.isclose(_cosine_similarity(a, b), -1.0)


def test_cosine_similarity_zero_vector():
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    assert _cosine_similarity(a, b) == 0.0


def test_cosine_similarity_scaled_vector():
    """Scaling a vector should not change cosine similarity."""
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 6.0]  # b = 2 * a
    assert math.isclose(_cosine_similarity(a, b), 1.0)


# ---------------------------------------------------------------------------
# retrieve_relevant_chunks — mock DB and embedder
# ---------------------------------------------------------------------------

def _make_db_rows(vectors: list[list[float]]):
    """Simulate sqlite3.Row objects."""
    rows = []
    for i, vec in enumerate(vectors):
        row = {
            "id": i + 1,
            "content": f"chunk content {i}",
            "page_num": i + 1,
            "start_char": i * 100,
            "end_char": (i + 1) * 100,
            "embedding": json.dumps(vec),
        }
        rows.append(row)
    return rows


def _dict_row(d):
    """Simple dict proxy that supports item access like sqlite3.Row."""
    class Row(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)
    return Row(d)


@patch("src.retriever.get_db")
@patch("src.retriever.embed_query")
def test_top_k_returns_correct_count(mock_embed, mock_get_db):
    query_vec = [1.0, 0.0, 0.0]
    mock_embed.return_value = query_vec

    # 6 chunks with varying embeddings
    vectors = [
        [1.0, 0.0, 0.0],   # sim = 1.0  ← best
        [0.9, 0.1, 0.0],
        [0.7, 0.3, 0.0],
        [0.5, 0.5, 0.0],
        [0.2, 0.8, 0.0],
        [0.0, 1.0, 0.0],   # sim = 0.0  ← worst
    ]
    rows = [_dict_row(r) for r in _make_db_rows(vectors)]

    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = rows
    mock_get_db.return_value = mock_db

    results = retrieve_relevant_chunks(session_id=1, query="test", top_k=4)
    assert len(results) == 4


@patch("src.retriever.get_db")
@patch("src.retriever.embed_query")
def test_results_sorted_by_score_descending(mock_embed, mock_get_db):
    query_vec = [1.0, 0.0]
    mock_embed.return_value = query_vec

    vectors = [
        [0.0, 1.0],   # sim = 0.0
        [1.0, 0.0],   # sim = 1.0
        [0.5, 0.5],   # sim ~0.71
    ]
    rows = [_dict_row(r) for r in _make_db_rows(vectors)]
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = rows
    mock_get_db.return_value = mock_db

    results = retrieve_relevant_chunks(session_id=1, query="test", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@patch("src.retriever.get_db")
@patch("src.retriever.embed_query")
def test_empty_chunks_returns_empty_list(mock_embed, mock_get_db):
    mock_embed.return_value = [1.0, 0.0]
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = []
    mock_get_db.return_value = mock_db

    results = retrieve_relevant_chunks(session_id=1, query="anything", top_k=4)
    assert results == []

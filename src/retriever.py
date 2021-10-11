import json
import math
from dataclasses import dataclass

from src.embedder import embed_query
from src.history import get_db


@dataclass
class ScoredChunk:
    chunk_id: int
    content: str
    page_num: int | None
    start_char: int
    end_char: int
    score: float


def retrieve_relevant_chunks(
    session_id: int,
    query: str,
    top_k: int = 4,
) -> list[ScoredChunk]:
    """
    Return the *top_k* chunks most semantically similar to *query*.

    Steps
    -----
    1. Embed the query with the same model used during indexing.
    2. Load all chunk embeddings for this session from SQLite.
    3. Compute cosine similarity between the query vector and every chunk.
    4. Return the top-k results sorted by descending similarity.

    Why cosine similarity?
    ----------------------
    It measures the angle between two vectors, ignoring magnitude — so a
    short chunk and a long chunk are compared fairly as long as their content
    is semantically aligned with the question.
    """
    query_vec = embed_query(query)

    db = get_db()
    rows = db.execute(
        """
        SELECT id, content, page_num, start_char, end_char, embedding
        FROM   chunks
        WHERE  session_id = ?
        """,
        (session_id,),
    ).fetchall()

    if not rows:
        return []

    scored: list[ScoredChunk] = []
    for row in rows:
        chunk_vec = json.loads(row["embedding"])
        score = _cosine_similarity(query_vec, chunk_vec)
        scored.append(
            ScoredChunk(
                chunk_id=row["id"],
                content=row["content"],
                page_num=row["page_num"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                score=score,
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

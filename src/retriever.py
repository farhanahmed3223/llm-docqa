"""Retrieve top-k chunks by cosine similarity."""
import json, math
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

def retrieve_relevant_chunks(session_id: int, query: str, top_k: int = 4) -> list[ScoredChunk]:
    qvec = embed_query(query)
    db = get_db()
    rows = db.execute(
        "SELECT id,content,page_num,start_char,end_char,embedding FROM chunks WHERE session_id=?",
        (session_id,)
    ).fetchall()
    scored = []
    for r in rows:
        cvec = json.loads(r["embedding"])
        score = _cosine(qvec, cvec)
        scored.append(ScoredChunk(r["id"], r["content"], r["page_num"], r["start_char"], r["end_char"], score))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    mag = lambda v: math.sqrt(sum(x*x for x in v))
    ma, mb = mag(a), mag(b)
    return dot / (ma * mb) if ma and mb else 0.0

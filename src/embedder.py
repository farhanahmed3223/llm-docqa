"""Generate and persist OpenAI embeddings."""
import json, os
from src.chunker import Chunk
from src.history import get_db

EMBED_MODEL = "text-embedding-ada-002"
EMBED_BATCH = 100

def embed_chunks(session_id: int, chunks: list[Chunk]) -> None:
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            page_num INTEGER,
            embedding TEXT NOT NULL
        );
    """)
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i+EMBED_BATCH]
        resp = openai.Embedding.create(model=EMBED_MODEL, input=[c.content for c in batch])
        rows = [
            (session_id, c.content, c.start_char, c.end_char, c.page_num,
             json.dumps(e.embedding))
            for c, e in zip(batch, resp.data)
        ]
        db.executemany(
            "INSERT INTO chunks (session_id,content,start_char,end_char,page_num,embedding) VALUES (?,?,?,?,?,?)",
            rows
        )
        db.commit()

def embed_query(query: str) -> list[float]:
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    resp = openai.Embedding.create(model=EMBED_MODEL, input=[query])
    return resp.data[0].embedding

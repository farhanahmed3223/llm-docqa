import json
import os

from openai import OpenAI

from src.chunker import Chunk
from src.history import get_db

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100  # OpenAI allows up to 2048 inputs per request; keep batches small


def embed_chunks(session_id: int, chunks: list[Chunk]) -> None:
    """
    Generate embeddings for every chunk and persist them to SQLite.

    Chunks are sent to the API in batches to stay well within request limits.
    Embeddings are stored as JSON arrays of floats — simple and portable.
    """
    client = _get_client()
    db = get_db()

    for batch_start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[batch_start : batch_start + EMBED_BATCH]
        texts = [c.content for c in batch]

        response = client.embeddings.create(model=EMBED_MODEL, input=texts)

        rows = []
        for chunk, embed_obj in zip(batch, response.data):
            embedding_json = json.dumps(embed_obj.embedding)
            rows.append(
                (
                    session_id,
                    chunk.content,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.page_num,
                    embedding_json,
                )
            )

        db.executemany(
            """
            INSERT INTO chunks
                (session_id, content, start_char, end_char, page_num, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        db.commit()


def embed_query(query: str) -> list[float]:
    """Return the embedding vector for a single query string."""
    client = _get_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)

"""Split documents into overlapping chunks."""
from dataclasses import dataclass

CHUNK_CHARS = 1500
OVERLAP_CHARS = 200

@dataclass
class Chunk:
    content: str
    start_char: int
    end_char: int
    page_num: int | None = None

def chunk_text(text: str, file_type: str = "text") -> list[Chunk]:
    if not text or not text.strip():
        return []
    # hack: if doc is shorter than one chunk, return as-is
    if len(text) <= CHUNK_CHARS:
        return [Chunk(content=text, start_char=0, end_char=len(text))]
    chunks = []
    step = max(1, CHUNK_CHARS - OVERLAP_CHARS)
    i = 0
    while i < len(text):
        end = min(i + CHUNK_CHARS, len(text))
        chunks.append(Chunk(content=text[i:end], start_char=i, end_char=end))
        if end == len(text):
            break
        i += step
    return chunks

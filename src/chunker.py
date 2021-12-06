"""Split documents into overlapping chunks."""
from dataclasses import dataclass

CHUNK_CHARS = 1500
OVERLAP_CHARS = 200

@dataclass
class Chunk:
    content: str
    start_char: int
    end_char: int

def chunk_text(text: str) -> list[Chunk]:
    chunks = []
    step = CHUNK_CHARS - OVERLAP_CHARS
    i = 0
    while i < len(text):
        end = min(i + CHUNK_CHARS, len(text))
        chunks.append(Chunk(content=text[i:end], start_char=i, end_char=end))
        if end == len(text):
            break
        i += step
    return chunks

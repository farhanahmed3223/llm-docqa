"""Token-aware chunker with configurable overlap."""
import re
from dataclasses import dataclass, field

CHUNK_SIZE = 500
OVERLAP = 50
ENCODING_MODEL = "cl100k_base"

@dataclass
class Chunk:
    content: str
    start_char: int
    end_char: int
    page_num: int | None = None
    token_count: int = 0

def chunk_text(text: str, file_type: str = "text") -> list[Chunk]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(ENCODING_MODEL)
        return _token_chunk(text, enc, file_type)
    except ImportError:
        return _char_chunk(text, file_type)

def _token_chunk(text: str, enc, file_type: str) -> list[Chunk]:
    tokens = enc.encode(text)
    if not tokens:
        return []
    step = max(1, CHUNK_SIZE - OVERLAP)
    chunks = []
    for start in range(0, len(tokens), step):
        end = min(start + CHUNK_SIZE, len(tokens))
        window = tokens[start:end]
        content = enc.decode(window)
        start_char = len(enc.decode(tokens[:start]))
        end_char = start_char + len(content)
        page = _detect_page(content) if file_type == "pdf" else None
        chunks.append(Chunk(content=content, start_char=start_char,
                            end_char=end_char, page_num=page, token_count=len(window)))
        if end == len(tokens):
            break
    return chunks

def _char_chunk(text: str, file_type: str) -> list[Chunk]:
    """Fallback if tiktoken not installed."""
    SIZE, OVL = 1500, 150
    if len(text) <= SIZE:
        return [Chunk(content=text, start_char=0, end_char=len(text))]
    chunks, i = [], 0
    while i < len(text):
        end = min(i + SIZE, len(text))
        content = text[i:end]
        chunks.append(Chunk(content=content, start_char=i, end_char=end))
        if end == len(text): break
        i += SIZE - OVL
    return chunks

_PAGE_RE = re.compile(r"\[PAGE (\d+)\]")

def _detect_page(content: str) -> int | None:
    m = _PAGE_RE.search(content)
    return int(m.group(1)) if m else None

def count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding(ENCODING_MODEL).encode(text))
    except ImportError:
        return len(text) // 4

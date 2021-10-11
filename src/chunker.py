import re
from dataclasses import dataclass, field

import tiktoken

CHUNK_SIZE = 500      # target tokens per chunk
OVERLAP = 50          # overlap tokens between consecutive chunks
ENCODING_MODEL = "cl100k_base"  # matches GPT-4o-mini / text-embedding-3-small


@dataclass
class Chunk:
    content: str
    start_char: int
    end_char: int
    page_num: int | None = None
    token_count: int = 0


def chunk_text(text: str, file_type: str) -> list[Chunk]:
    """
    Split *text* into overlapping token-based chunks.

    Strategy
    --------
    1. Tokenise the entire document with tiktoken.
    2. Slide a window of CHUNK_SIZE tokens, stepping forward by
       (CHUNK_SIZE - OVERLAP) tokens each time.
    3. Decode each window back to a string and record character offsets.
    4. For PDFs, parse the [PAGE N] markers embedded by the extractor to
       annotate each chunk with its source page number.

    The overlap ensures that a sentence straddling a chunk boundary is
    present in full in at least one of the two neighbouring chunks.
    """
    enc = tiktoken.get_encoding(ENCODING_MODEL)
    tokens = enc.encode(text)

    if not tokens:
        return []

    step = max(1, CHUNK_SIZE - OVERLAP)
    chunks: list[Chunk] = []

    for start_tok in range(0, len(tokens), step):
        end_tok = min(start_tok + CHUNK_SIZE, len(tokens))
        window_tokens = tokens[start_tok:end_tok]
        content = enc.decode(window_tokens)

        # Approximate character offsets by scanning from last chunk boundary
        start_char = _token_to_char(text, tokens, start_tok, enc)
        end_char = _token_to_char(text, tokens, end_tok, enc)

        page_num = _detect_page(content) if file_type == "pdf" else None

        chunks.append(
            Chunk(
                content=content,
                start_char=start_char,
                end_char=end_char,
                page_num=page_num,
                token_count=len(window_tokens),
            )
        )

        if end_tok == len(tokens):
            break

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_to_char(text: str, tokens, token_index: int, enc) -> int:
    """Return the character offset in *text* corresponding to *token_index*."""
    if token_index == 0:
        return 0
    prefix = enc.decode(tokens[:token_index])
    return len(prefix)


_PAGE_RE = re.compile(r"\[PAGE (\d+)\]")


def _detect_page(content: str) -> int | None:
    """Return the first page number found in chunk content, or None."""
    match = _PAGE_RE.search(content)
    return int(match.group(1)) if match else None


def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding(ENCODING_MODEL)
    return len(enc.encode(text))

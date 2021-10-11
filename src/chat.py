import os
from typing import Any

from openai import OpenAI

from src.chunker import count_tokens
from src.retriever import ScoredChunk

MODEL = "gpt-4o-mini"
MAX_PROMPT_TOKENS = 12_000
SYSTEM_PROMPT = """\
You are a precise document assistant. Answer the user's question using ONLY
the context passages provided below. If the answer is not contained in the
context, say "I don't have enough information in the provided document to
answer that." Do not make up or infer facts beyond what is explicitly stated.
Format your answer clearly. Where helpful, quote short relevant phrases from
the context to support your answer.\
"""


def ask_question(
    question: str,
    chunks: list[ScoredChunk],
    history: list[dict[str, str]],
) -> tuple[str, int, list[str]]:
    """
    Assemble a prompt from the retrieved chunks + conversation history, call
    GPT-4o-mini, and return (answer, total_tokens_used, source_labels).

    Token budget management
    -----------------------
    We count tokens before sending. If the assembled prompt would exceed
    MAX_PROMPT_TOKENS we drop the lowest-scoring chunks one by one until it
    fits. This guarantees we never hit a context-limit API error at runtime.
    """
    client = _get_client()

    # Build context block — chunks sorted best-first
    working_chunks = list(chunks)  # copy so we don't mutate caller's list
    context_block, sources = _build_context(working_chunks)

    # Assemble messages
    messages = _build_messages(question, context_block, history)

    # Enforce token budget — drop least-relevant chunks if needed
    while count_tokens(_messages_to_text(messages)) > MAX_PROMPT_TOKENS and len(working_chunks) > 1:
        working_chunks.pop()  # remove lowest-score (last in sorted list)
        context_block, sources = _build_context(working_chunks)
        messages = _build_messages(question, context_block, history)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    return answer, tokens_used, sources


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: list[ScoredChunk]) -> tuple[str, list[str]]:
    parts = []
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        if chunk.page_num:
            label = f"Page {chunk.page_num}"
        else:
            label = f"chars {chunk.start_char}–{chunk.end_char}"
        sources.append(label)
        parts.append(f"[Context {i} — {label}]\n{chunk.content.strip()}")
    return "\n\n".join(parts), sources


def _build_messages(
    question: str,
    context_block: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    user_content = (
        f"Document context:\n\n{context_block}\n\n"
        f"Question: {question}"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    return " ".join(m["content"] for m in messages)


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )
    return OpenAI(api_key=api_key)

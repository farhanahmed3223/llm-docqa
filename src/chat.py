"""OpenAI chat wrapper with token budget management."""
import os
from src.chunker import count_tokens

DEFAULT_MODEL = "gpt-3.5-turbo"
CONTEXT_LIMIT = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "gpt-4-turbo-preview": 128000,
}
SYSTEM_OVERHEAD = 300

def answer(context_chunks, question: str, history: list[dict], model: str) -> str:
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    budget = CONTEXT_LIMIT.get(model, 4096) - SYSTEM_OVERHEAD - count_tokens(question)
    context = _build_context(context_chunks, budget)
    messages = [
        {"role": "system", "content": f"Answer questions using this document:\n\n{context}"},
        *history[-4:],  # keep last 2 exchanges
        {"role": "user", "content": question},
    ]
    resp = openai.ChatCompletion.create(model=model, messages=messages)
    return resp.choices[0].message.content

def _build_context(chunks, budget: int) -> str:
    parts, used = [], 0
    for chunk in chunks:
        t = count_tokens(chunk.content)
        if used + t > budget:
            break
        parts.append(chunk.content)
        used += t
    return "\n\n---\n\n".join(parts)

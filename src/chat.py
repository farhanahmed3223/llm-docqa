"""OpenAI chat wrapper — upgraded to openai>=1.0 SDK."""
import os
from src.chunker import count_tokens

CONTEXT_LIMIT = {
    "gpt-3.5-turbo": 16385,
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
}
SYSTEM_OVERHEAD = 400

def answer(context_chunks, question: str, history: list[dict], model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    budget = CONTEXT_LIMIT.get(model, 16385) - SYSTEM_OVERHEAD - count_tokens(question)
    context = _build_context(context_chunks, budget)
    messages = [
        {"role": "system", "content": f"Answer questions using only this document:\n\n{context}"},
        *history[-6:],
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
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

import openai, os, sys, argparse, textwrap

MAX_TOKENS = 3000

def chunk_text(text: str, max_chars: int = 12000):
    """Split text into rough chunks."""
    chunks = []
    while text:
        chunks.append(text[:max_chars])
        text = text[max_chars:]
    return chunks

def ask(chunks, question, model):
    # Use first chunk for now — will do proper retrieval later
    context = chunks[0] if chunks else ""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Answer using this text:\n\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--model", default="gpt-3.5-turbo")
    args = p.parse_args()
    openai.api_key = os.environ["OPENAI_API_KEY"]
    text = open(args.file).read()
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks")
    while True:
        q = input("> ")
        if q.lower() in ("exit", "q"):
            break
        print(ask(chunks, q, args.model))

if __name__ == "__main__":
    main()

"""CLI entry point."""
import argparse, os, sys
from pathlib import Path

def _load_env():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

def build_parser():
    p = argparse.ArgumentParser(prog="docqa", description="Ask questions about a document.")
    p.add_argument("file", help="Path to document (.txt, .pdf, .md)")
    p.add_argument("--model", default=os.environ.get("DOCQA_MODEL", "gpt-3.5-turbo"))
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--verbose", "-v", action="store_true")
    return p

def main():
    _load_env()
    args = build_parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    from src.extractor import extract_text
    from src.chunker import chunk_text
    from src.embedder import embed_chunks
    from src.retriever import retrieve_relevant_chunks
    from src.history import new_session, add_message, get_history
    from src.chat import answer
    from src.display import print_answer, print_header

    text, ftype = extract_text(args.file)
    chunks = chunk_text(text, ftype)
    session_id = new_session(args.file)
    if args.verbose:
        print(f"Embedding {len(chunks)} chunks...")
    embed_chunks(session_id, chunks)
    print_header(args.file, len(chunks))

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            break
        relevant = retrieve_relevant_chunks(session_id, q, top_k=args.top_k)
        hist = get_history(session_id)
        ans = answer(relevant, q, hist, args.model)
        add_message(session_id, "user", q)
        add_message(session_id, "assistant", ans)
        print_answer(ans)

if __name__ == "__main__":
    main()

"""CLI entry point."""
import argparse
import os
import sys
from src.extractor import extract_text
from src.chunker import chunk_text

def build_parser():
    p = argparse.ArgumentParser(
        prog="docqa",
        description="Ask questions about a document using GPT.",
    )
    p.add_argument("file", help="Path to document (.txt, .pdf, .md)")
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--verbose", "-v", action="store_true")
    return p

def main():
    args = build_parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not set.")
    text, ftype = extract_text(args.file)
    chunks = chunk_text(text, ftype)
    if args.verbose:
        print(f"[{ftype}] {len(text)} chars → {len(chunks)} chunks")
    # TODO: hook up actual QA loop
    print("Ready. Type your question:")
    while True:
        q = input("> ").strip()
        if not q or q.lower() in ("q", "quit", "exit"):
            break
        print(f"(stub) Would answer: {q!r} using {len(chunks)} chunks")

if __name__ == "__main__":
    main()

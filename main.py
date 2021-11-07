import openai, os, sys, argparse, textwrap

def build_parser():
    p = argparse.ArgumentParser(description="Ask questions about a document using GPT.")
    p.add_argument("file", help="Path to .txt or .pdf document")
    p.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    p.add_argument("--max-chars", type=int, default=4000, help="Max chars to send")
    return p

def ask(text, question, model, max_chars):
    import openai
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Answer questions using this document:\n\n{text[:max_chars]}"},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

def main():
    args = build_parser().parse_args()
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    text = open(args.file).read()
    print(f"Loaded: {args.file} ({len(text)} chars)\nType 'exit' to quit.\n")
    while True:
        q = input("> ")
        if q.lower() in ("q", "quit", "exit"):
            break
        print(textwrap.fill(ask(text, q, args.model, args.max_chars), 80))

if __name__ == "__main__":
    main()

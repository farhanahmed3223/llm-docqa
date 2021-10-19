import openai, os, sys, textwrap

openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    print("Set OPENAI_API_KEY first")
    sys.exit(1)

def ask(text: str, question: str) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Answer questions using this document:\n\n{text[:4000]}"},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "doc.txt"
    text = open(path).read()
    print(f"Loaded: {path} ({len(text)} chars)")
    while True:
        q = input("\n> ")
        if q.lower() in ("q", "quit", "exit"):
            break
        print(textwrap.fill(ask(text, q), 80))

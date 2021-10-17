import openai, os, sys

openai.api_key = os.environ["OPENAI_API_KEY"]

filepath = sys.argv[1] if len(sys.argv) > 1 else "doc.txt"
text = open(filepath).read()
print(f"Loaded {len(text)} chars from {filepath}")

while True:
    question = input("\nAsk: ")
    if question.lower() in ("quit", "exit"):
        break
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You answer questions using this document:\n\n" + text[:3000]},
            {"role": "user", "content": question},
        ],
    )
    print(resp.choices[0].message.content)

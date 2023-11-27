# llm-docqa

Ask questions about any document (.txt, .pdf, .md) using OpenAI embeddings + GPT.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Usage

```bash
python main.py path/to/document.pdf
# or after installing:
docqa path/to/document.pdf
```

## How it works

1. Extracts text from the document
2. Splits it into token-sized chunks with overlap
3. Embeds all chunks with `text-embedding-3-small`
4. For each question, retrieves the top-k most relevant chunks via cosine similarity
5. Sends those chunks + conversation history to GPT to generate an answer

## Models

Set `DOCQA_MODEL` in `.env` to switch models:
- `gpt-4o-mini` (default, fast and cheap)
- `gpt-4o` (better reasoning)
- `gpt-3.5-turbo` (legacy)

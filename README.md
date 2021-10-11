# llm-docqa

A command-line tool for asking questions about any PDF or plain-text document. Built to demonstrate real LLM engineering — not just API calls, but the context management decisions that make production systems work.

```
$ python main.py ask --file report.pdf "What are the key findings?"

  Answer ─────────────────────────────────────────────────────────
  The report identifies three key findings:

  1. **Revenue growth of 12%** in Q3, driven primarily by the APAC region.
  2. Operating costs fell 8% following the automation rollout in Q2.
  3. Customer retention improved to **94%**, up from 89% the prior year.
  ─────────────────────────────────────────────────────────────────
  Sources: Page 4, Page 7   Tokens: 1,842   Est. cost: $0.00028
```

---

## Why this project exists

Most people claiming "OpenAI API experience" on their CV have called `client.chat.completions.create()` once. The hard part is everything that surrounds that call:

- What do you do when the document is **larger than the context window**?
- How do you **maintain conversation history** without hitting token limits?
- How do you make **follow-up questions** work naturally?

This project answers those questions with a production-quality chunking and retrieval pipeline.

---

## Tech stack

| Library | Purpose |
|---|---|
| `openai` | GPT-4o-mini for answers, `text-embedding-3-small` for chunk ranking |
| `click` | CLI interface — clean commands with `--help` built in |
| `rich` | Coloured panels, progress spinners, markdown rendering |
| `pymupdf` | PDF text extraction, page by page |
| `tiktoken` | Token counting to stay within context limits |
| `sqlite3` | Stores sessions, messages, and chunk embeddings locally |
| `pytest` + `pytest-mock` | Tests mock the OpenAI client — no real API calls in CI |
| GitHub Actions | CI: lint (`ruff`) + tests on every push |

---

## Install and run

**Requirements:** Python 3.12+

```bash
# 1. Clone the repo
git clone https://github.com/yourname/llm-docqa.git
cd llm-docqa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...

# 4. Ask a question
python main.py ask --file report.pdf "What are the key findings?"
```

The database is stored at `~/.llm_docqa.db` by default. Override with:
```bash
export DOCQA_DB_PATH=/path/to/custom.db
```

---

## Commands

```bash
# Ask a question about a file (creates a new session automatically)
python main.py ask --file report.pdf "What are the key findings?"

# Continue an existing session (maintains conversation context)
python main.py ask --file report.pdf --session q3-review "What about the risks?"

# Ask about a plain text file
python main.py ask --file notes.txt "Summarise the action items"

# List all saved sessions
python main.py sessions

# Print full history of a session
python main.py history --session q3-review

# Delete a session and its stored data
python main.py delete --session q3-review

# Show token usage and cost estimate for a session
python main.py stats --session q3-review
```

---

## Terminal screenshots

**1. Asking a question (new session)**

```
$ python main.py ask --file annual_report.pdf "What was the revenue in Q3?"

› Session '3f9a1c2b' created — 47 chunks indexed.

  Answer ─────────────────────────────────────────────────────────
  Q3 revenue was **$4.2 billion**, representing a 12% year-over-year
  increase. The APAC region was the primary growth driver, contributing
  31% of total revenue compared to 24% in the prior year period.
  ─────────────────────────────────────────────────────────────────
  Sources: Page 4, Page 6   Tokens: 1,923   Est. cost: $0.00029
```

**2. Follow-up question (same session)**

```
$ python main.py ask --file annual_report.pdf --session 3f9a1c2b "What were the main risks?"

› Resuming session '3f9a1c2b'.

  Answer ─────────────────────────────────────────────────────────
  The report identifies three primary risks:

  1. **Supply chain disruption** — ongoing semiconductor shortages affecting
     hardware product lines (p. 18).
  2. **Regulatory exposure** in the EU under the proposed Digital Markets Act.
  3. **FX headwinds** — a strong USD reduced reported revenue by ~2% in EMEA.
  ─────────────────────────────────────────────────────────────────
  Sources: Page 18, Page 22   Tokens: 2,104   Est. cost: $0.00032
```

**3. Listing sessions**

```
$ python main.py sessions

  ╭─────────────────────────────────────────────────────────────────╮
  │ Saved Sessions                                                  │
  ├──────────────┬──────────────────────┬──────────┬───────────────┤
  │ Session      │ File                 │ Messages │ Last Used     │
  ├──────────────┼──────────────────────┼──────────┼───────────────┤
  │ 3f9a1c2b     │ annual_report.pdf    │ 4        │ 2024-11-01    │
  │ q3-review    │ q3_summary.pdf       │ 12       │ 2024-10-28    │
  │ notes-oct    │ meeting_notes.txt    │ 6        │ 2024-10-25    │
  ╰──────────────┴──────────────────────┴──────────┴───────────────╯
```

**4. Token stats**

```
$ python main.py stats --session q3-review

  Stats — q3-review
  ─────────────────────────────────────────
  Messages                  12
  Chunks indexed            53
  Total tokens used         18,430
  Estimated cost (GPT-4o-mini)   $0.00276
```

---

## How chunking and retrieval work

This is the important part. Here's exactly what happens when you ask a question.

### The problem

GPT-4o-mini has a 128k token context window, but sending an entire document on every question is wasteful, slow, and expensive. More importantly, **most of the document is irrelevant to any given question**. Injecting irrelevant text degrades answer quality — the model has to find the signal in the noise.

### Step 1 — Split the document into chunks

When a document is first loaded, it is split into overlapping chunks of approximately **500 tokens each**, with a **50-token overlap** between consecutive chunks.

```
Document tokens:  [──────────────────────────────────────────────]

Chunk 1:          [════════════════════]
Chunk 2:                         [════════════════════]
Chunk 3:                                        [════════════════════]
                                   ↑ overlap ↑
```

Why overlap? A sentence sitting exactly at a chunk boundary would be split in two, making it half-present in each chunk and fully present in neither. The overlap guarantees that any sentence appears complete in at least one chunk.

Token counting is done with **tiktoken** using the same encoding as the model (`cl100k_base`), so the chunk sizes are exact — not character-based estimates.

### Step 2 — Embed every chunk

Each chunk is sent to `text-embedding-3-small` via the OpenAI API. This converts the text into a **1536-dimensional vector** that captures its semantic meaning. These vectors are stored in SQLite as JSON arrays. This only happens once per session — not on every question.

### Step 3 — Embed the question and rank chunks

When a question arrives, it is also embedded using `text-embedding-3-small`. We then compute the **cosine similarity** between the question vector and every stored chunk vector.

Cosine similarity measures the angle between two vectors rather than their magnitude, so a short chunk and a long chunk are compared fairly — only the semantic content matters.

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (norm(a) * norm(b))
```

The top 4 chunks by similarity score are selected.

### Step 4 — Assemble the prompt with a token budget

The selected chunks, the last 4 turns of conversation history, and the question are assembled into a single prompt. Before sending, **tiktoken counts the total tokens**. If the assembled prompt would exceed 12,000 tokens, the lowest-scoring chunks are dropped one by one until it fits.

This guarantees the API call never fails with a context-limit error at runtime.

The system prompt instructs the model:

> "Answer using only the provided context. If the answer is not in the context, say so."

This prevents hallucination — the model cannot invent facts beyond what appears in the retrieved chunks.

### Step 5 — Conversation history

The last 4 turns of chat history (8 messages) are injected before the current question, so follow-up questions like "What about the risks?" refer naturally to the previous answer without needing to restate context.

---

## Supported file types

| Type | Support |
|---|---|
| `.pdf` — text-based | ✅ Full support. Text is extracted page-by-page; answers include page references. |
| `.txt` — plain text | ✅ Full support. Character offsets are used as source references. |
| `.pdf` — scanned / image-only | ❌ Not supported. The tool requires a real text layer in the PDF. If `pdfplumber` extracts no text, an error is raised with a clear message. Scanned PDFs require an OCR step first (e.g. `ocrmypdf`). |
| `.docx`, `.csv`, `.html` | ❌ Not yet supported. See roadmap below. |

---

## Cost estimate

**GPT-4o-mini:** ~$0.00015 per 1,000 tokens (blended input + output).

| Usage | Approx. tokens | Approx. cost |
|---|---|---|
| Single question | ~1,500–2,500 | $0.00023–$0.00038 |
| 10-question session | ~20,000 | ~$0.003 |
| Heavy research session (50 questions) | ~100,000 | ~$0.015 |

The embedding calls (`text-embedding-3-small`) cost ~$0.00002 per 1,000 tokens — negligible. A 100-page PDF produces roughly 50 chunks (~25,000 tokens total), costing about **$0.0005 to index** once.

---

## Project structure

```
llm-docqa/
├── src/
│   ├── cli.py          # Click commands — entry point
│   ├── extractor.py    # PDF → text (PyMuPDF), txt → text
│   ├── chunker.py      # Split text into overlapping token chunks
│   ├── embedder.py     # Generate + store chunk embeddings via OpenAI
│   ├── retriever.py    # Cosine similarity search, top-k chunk selection
│   ├── chat.py         # Assemble prompt, call GPT-4o-mini, enforce token budget
│   ├── history.py      # SQLite: save/load sessions and messages
│   └── display.py      # Rich formatting helpers
├── tests/
│   ├── test_chunker.py   # Chunk sizes, overlap, page annotation
│   ├── test_retriever.py # Cosine similarity ranking
│   └── test_chat.py      # Mock OpenAI client, prompt assembly, token budget
├── main.py
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml
└── README.md
```

---

## SQLite schema

```sql
sessions → id, name, file_path, created_at, last_used
messages → id, session_id, role (user/assistant), content, tokens_used, created_at
chunks   → id, session_id, content, start_char, end_char, page_num,
           embedding (JSON array of floats)
```

Sessions cascade-delete their messages and chunks when deleted.

---

## Running tests

```bash
pytest tests/ -v
```

All tests mock the OpenAI client — no API calls are made in CI. The test suite covers chunk sizing and overlap correctness, cosine similarity edge cases, prompt assembly, and the token-budget drop logic.

---

## What I'd add next

- **Streaming responses** — stream GPT-4o-mini output token-by-token with Rich's `Live` display for faster perceived response time.
- **Web UI** — a minimal FastAPI + HTMX interface so non-technical users can upload documents and chat without the CLI.
- **DOCX support** — `python-docx` for Word documents.
- **CSV / structured data** — route tabular questions through a pandas query layer rather than embedding.
- **Re-ranking** — use a cross-encoder (e.g. `ms-marco-MiniLM`) to re-score the top-20 retrieved chunks before selecting the top 4, improving precision on ambiguous queries.
- **Hybrid search** — combine BM25 keyword search with embedding similarity for better recall on proper nouns and technical terms that embeddings sometimes mishandle.

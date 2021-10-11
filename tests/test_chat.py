import pytest
from unittest.mock import patch, MagicMock

from src.chat import ask_question, _build_context, _build_messages, _messages_to_text
from src.retriever import ScoredChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chunk(content: str, page_num: int | None = None, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=1,
        content=content,
        page_num=page_num,
        start_char=0,
        end_char=len(content),
        score=score,
    )


CHUNKS = [
    make_chunk("The revenue grew by 12% in Q3.", page_num=2, score=0.95),
    make_chunk("Operating costs decreased due to automation.", page_num=3, score=0.80),
]

HISTORY = [
    {"role": "user", "content": "What is this document about?"},
    {"role": "assistant", "content": "It is a financial report."},
]


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

def test_build_context_labels_pages():
    context_block, sources = _build_context(CHUNKS)
    assert "Page 2" in sources
    assert "Page 3" in sources


def test_build_context_labels_char_range_when_no_page():
    chunk = make_chunk("Some text without page info.", page_num=None)
    _, sources = _build_context([chunk])
    assert sources[0].startswith("chars")


def test_build_context_includes_all_chunk_contents():
    context_block, _ = _build_context(CHUNKS)
    for chunk in CHUNKS:
        assert chunk.content.strip() in context_block


def test_build_context_empty_chunks():
    context_block, sources = _build_context([])
    assert context_block == ""
    assert sources == []


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

def test_build_messages_starts_with_system():
    messages = _build_messages("Question?", "Context here.", HISTORY)
    assert messages[0]["role"] == "system"


def test_build_messages_ends_with_user():
    messages = _build_messages("Question?", "Context here.", HISTORY)
    assert messages[-1]["role"] == "user"


def test_build_messages_includes_history():
    messages = _build_messages("Question?", "Context here.", HISTORY)
    roles = [m["role"] for m in messages]
    # system, user (history), assistant (history), user (current)
    assert roles.count("user") == 2
    assert roles.count("assistant") == 1


def test_build_messages_context_in_user_content():
    messages = _build_messages("Question?", "Important context.", [])
    user_msg = messages[-1]["content"]
    assert "Important context." in user_msg
    assert "Question?" in user_msg


# ---------------------------------------------------------------------------
# ask_question — mock the OpenAI client
# ---------------------------------------------------------------------------

def _make_openai_response(answer: str, total_tokens: int):
    response = MagicMock()
    response.choices[0].message.content = answer
    response.usage.total_tokens = total_tokens
    return response


@patch("src.chat._get_client")
def test_ask_question_returns_answer(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(
        "Revenue grew 12% in Q3.", 400
    )
    mock_get_client.return_value = mock_client

    answer, tokens, sources = ask_question("What was revenue growth?", CHUNKS, HISTORY)

    assert answer == "Revenue grew 12% in Q3."
    assert tokens == 400
    assert len(sources) == 2


@patch("src.chat._get_client")
def test_ask_question_passes_history_to_api(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("OK", 100)
    mock_get_client.return_value = mock_client

    ask_question("Follow up?", CHUNKS, HISTORY)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    # History messages should be present
    contents = [m["content"] for m in messages]
    assert any("financial report" in c for c in contents)


@patch("src.chat.count_tokens")
@patch("src.chat._get_client")
def test_ask_question_drops_chunks_when_over_token_limit(mock_get_client, mock_count_tokens):
    """When assembled prompt exceeds MAX_PROMPT_TOKENS, lowest-scored chunks are dropped."""
    # First call: over limit; subsequent calls: under limit
    mock_count_tokens.side_effect = [15_000, 15_000, 11_000]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("Trimmed answer.", 300)
    mock_get_client.return_value = mock_client

    answer, tokens, sources = ask_question("Test?", CHUNKS, [])

    assert answer == "Trimmed answer."
    # After dropping one chunk, only 1 source should remain
    assert len(sources) == 1


@patch("src.chat._get_client")
def test_ask_question_no_history(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("Direct answer.", 200)
    mock_get_client.return_value = mock_client

    answer, tokens, sources = ask_question("Standalone?", CHUNKS, [])
    assert answer == "Direct answer."

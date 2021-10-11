import click
from rich.console import Console
from rich.table import Table
from rich import box

from src.extractor import extract_text
from src.chunker import chunk_text
from src.embedder import embed_chunks
from src.retriever import retrieve_relevant_chunks
from src.chat import ask_question
from src.history import (
    get_or_create_session,
    save_message,
    list_sessions,
    get_history,
    delete_session,
    get_stats,
)
from src.display import (
    print_answer,
    print_error,
    print_info,
    print_sessions_table,
    print_history,
    print_stats,
)

console = Console()


@click.group()
def cli():
    """LLM Document Q&A — ask questions about any PDF or text file."""
    pass


@cli.command()
@click.option("--file", "-f", required=True, help="Path to PDF or .txt file.")
@click.option("--session", "-s", default=None, help="Session name (creates new if omitted).")
@click.argument("question")
def ask(file, session, question):
    """Ask a question about a document."""
    try:
        # Extract text from file
        with console.status("[bold cyan]Reading document…", spinner="dots"):
            text, file_type = extract_text(file)

        # Get or create session — chunks are recomputed if new
        session_id, session_name, is_new = get_or_create_session(session, file)

        if is_new:
            with console.status("[bold cyan]Chunking and embedding document…", spinner="dots"):
                chunks = chunk_text(text, file_type)
                embed_chunks(session_id, chunks)
            print_info(f"Session '[bold]{session_name}[/bold]' created — {len(chunks)} chunks indexed.")
        else:
            print_info(f"Resuming session '[bold]{session_name}[/bold]'.")

        # Retrieve relevant chunks
        with console.status("[bold cyan]Retrieving relevant context…", spinner="dots"):
            top_chunks = retrieve_relevant_chunks(session_id, question, top_k=4)

        # Load recent conversation history
        history = get_history(session_id, limit=4)

        # Call the model
        with console.status("[bold cyan]Thinking…", spinner="dots"):
            answer, tokens_used, sources = ask_question(question, top_chunks, history)

        # Persist messages
        save_message(session_id, "user", question, 0)
        save_message(session_id, "assistant", answer, tokens_used)

        print_answer(answer, sources, tokens_used)

    except FileNotFoundError:
        print_error(f"File not found: {file}")
    except Exception as e:
        print_error(str(e))


@cli.command("sessions")
def list_sessions_cmd():
    """List all saved sessions."""
    rows = list_sessions()
    if not rows:
        print_info("No sessions found. Run [bold]ask[/bold] to create one.")
        return
    print_sessions_table(rows)


@cli.command()
@click.option("--session", "-s", required=True, help="Session name to display.")
def history(session):
    """Print full Q&A history for a session."""
    rows = get_history(session_id=None, session_name=session)
    if rows is None:
        print_error(f"Session '{session}' not found.")
        return
    if not rows:
        print_info("No messages in this session yet.")
        return
    print_history(session, rows)


@cli.command()
@click.option("--session", "-s", required=True, help="Session name to delete.")
def delete(session):
    """Delete a session and all its stored data."""
    deleted = delete_session(session)
    if deleted:
        print_info(f"Session '[bold]{session}[/bold]' deleted.")
    else:
        print_error(f"Session '{session}' not found.")


@cli.command()
@click.option("--session", "-s", required=True, help="Session name for stats.")
def stats(session):
    """Show token usage and cost estimate for a session."""
    data = get_stats(session)
    if data is None:
        print_error(f"Session '{session}' not found.")
        return
    print_stats(session, data)

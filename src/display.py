from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# GPT-4o-mini pricing (per 1K tokens, blended input+output estimate)
COST_PER_1K = 0.00015


def print_answer(answer: str, sources: list[str], tokens_used: int) -> None:
    source_str = ", ".join(sources) if sources else "unknown"
    cost = (tokens_used / 1000) * COST_PER_1K

    console.print(
        Panel(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print(
        f"  [dim]Sources:[/dim] [cyan]{source_str}[/cyan]   "
        f"[dim]Tokens:[/dim] [yellow]{tokens_used:,}[/yellow]   "
        f"[dim]Est. cost:[/dim] [yellow]${cost:.5f}[/yellow]"
    )
    console.print()


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_info(message: str) -> None:
    console.print(f"[bold blue]›[/bold blue] {message}")


def print_sessions_table(rows: list[Any]) -> None:
    table = Table(
        title="Saved Sessions",
        box=box.ROUNDED,
        border_style="blue",
        header_style="bold cyan",
    )
    table.add_column("Session", style="bold white")
    table.add_column("File")
    table.add_column("Messages", justify="right")
    table.add_column("Last Used")

    for row in rows:
        table.add_row(
            row["name"],
            row["file_path"],
            str(row["message_count"]),
            row["last_used"],
        )
    console.print(table)


def print_history(session_name: str, rows: list[dict]) -> None:
    console.print(
        f"\n[bold blue]Session:[/bold blue] [bold]{session_name}[/bold]\n"
    )
    for msg in rows:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("created_at", "")
        if role == "user":
            console.print(f"[bold cyan]You[/bold cyan] [dim]{ts}[/dim]")
            console.print(f"  {content}\n")
        else:
            console.print(f"[bold green]Assistant[/bold green] [dim]{ts}[/dim]")
            console.print(Panel(Markdown(content), border_style="green", padding=(0, 1)))
            console.print()


def print_stats(session_name: str, data: dict) -> None:
    total_tokens = data["total_tokens"]
    cost = (total_tokens / 1000) * COST_PER_1K

    table = Table(
        title=f"Stats — {session_name}",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Messages", str(data["message_count"]))
    table.add_row("Chunks indexed", str(data["chunk_count"]))
    table.add_row("Total tokens used", f"{total_tokens:,}")
    table.add_row("Estimated cost (GPT-4o-mini)", f"${cost:.5f}")

    console.print(table)

import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = os.getenv("DOCQA_DB_PATH", str(Path.home() / ".llm_docqa.db"))

_connection: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(DB_PATH)
        _connection.row_factory = sqlite3.Row
        _migrate(_connection)
    return _connection


def _migrate(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    UNIQUE NOT NULL,
            file_path  TEXT    NOT NULL,
            created_at TEXT    NOT NULL,
            last_used  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
            content     TEXT    NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            content     TEXT    NOT NULL,
            start_char  INTEGER NOT NULL,
            end_char    INTEGER NOT NULL,
            page_num    INTEGER,
            embedding   TEXT    NOT NULL
        );
    """)
    db.commit()


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def get_or_create_session(
    name: str | None,
    file_path: str,
) -> tuple[int, str, bool]:
    """
    Return (session_id, session_name, is_new).
    If *name* is None a short random slug is generated.
    """
    db = get_db()
    now = _now()

    if name is None:
        name = uuid.uuid4().hex[:8]

    row = db.execute(
        "SELECT id FROM sessions WHERE name = ?", (name,)
    ).fetchone()

    if row:
        db.execute(
            "UPDATE sessions SET last_used = ? WHERE id = ?",
            (now, row["id"]),
        )
        db.commit()
        return row["id"], name, False

    cur = db.execute(
        "INSERT INTO sessions (name, file_path, created_at, last_used) VALUES (?, ?, ?, ?)",
        (name, file_path, now, now),
    )
    db.commit()
    return cur.lastrowid, name, True


def list_sessions() -> list[sqlite3.Row]:
    db = get_db()
    return db.execute("""
        SELECT s.name, s.file_path, s.last_used,
               COUNT(m.id) AS message_count
        FROM   sessions s
        LEFT JOIN messages m ON m.session_id = s.id
        GROUP BY s.id
        ORDER BY s.last_used DESC
    """).fetchall()


def delete_session(name: str) -> bool:
    db = get_db()
    row = db.execute(
        "SELECT id FROM sessions WHERE name = ?", (name,)
    ).fetchone()
    if not row:
        return False
    db.execute("DELETE FROM chunks   WHERE session_id = ?", (row["id"],))
    db.execute("DELETE FROM messages WHERE session_id = ?", (row["id"],))
    db.execute("DELETE FROM sessions WHERE id = ?", (row["id"],))
    db.commit()
    return True


# ---------------------------------------------------------------------------
# Message management
# ---------------------------------------------------------------------------

def save_message(
    session_id: int,
    role: str,
    content: str,
    tokens_used: int,
) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO messages (session_id, role, content, tokens_used, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, tokens_used, _now()),
    )
    db.commit()


def get_history(
    session_id: int | None,
    session_name: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]] | None:
    """
    Return messages for a session.

    If session_name is provided, look up by name first.
    If limit is set, return only the last *limit* pairs (user + assistant).
    Returns None if the session name was given but doesn't exist.
    """
    db = get_db()

    if session_name is not None:
        row = db.execute(
            "SELECT id FROM sessions WHERE name = ?", (session_name,)
        ).fetchone()
        if not row:
            return None
        session_id = row["id"]

    query = """
        SELECT role, content, tokens_used, created_at
        FROM   messages
        WHERE  session_id = ?
        ORDER BY id ASC
    """
    rows = db.execute(query, (session_id,)).fetchall()

    result = [dict(r) for r in rows]

    if limit is not None:
        # Keep last `limit` full turns (user + assistant = 2 messages per turn)
        result = result[-(limit * 2):]

    return result


def get_stats(session_name: str) -> dict[str, Any] | None:
    db = get_db()
    row = db.execute(
        "SELECT id FROM sessions WHERE name = ?", (session_name,)
    ).fetchone()
    if not row:
        return None

    stats = db.execute("""
        SELECT COUNT(*) AS message_count,
               SUM(tokens_used) AS total_tokens
        FROM   messages
        WHERE  session_id = ?
    """, (row["id"],)).fetchone()

    chunk_count = db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE session_id = ?", (row["id"],)
    ).fetchone()["n"]

    return {
        "message_count": stats["message_count"] or 0,
        "total_tokens": stats["total_tokens"] or 0,
        "chunk_count": chunk_count,
    }


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

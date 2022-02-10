"""Lightweight SQLite conversation history."""
import sqlite3
import os

_DB_PATH = os.path.expanduser("~/.docqa.db")
_db = None

def get_db():
    global _db
    if _db is None:
        _db = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _db.row_factory = sqlite3.Row
        _create_tables(_db)
    return _db

def _create_tables(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    db.commit()

def new_session(filepath: str) -> int:
    db = get_db()
    cur = db.execute("INSERT INTO sessions (filepath) VALUES (?)", (filepath,))
    db.commit()
    return cur.lastrowid

def add_message(session_id: int, role: str, content: str):
    db = get_db()
    db.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    db.commit()

def get_history(session_id: int) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

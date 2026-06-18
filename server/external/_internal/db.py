"""SQLite open helper: opens the DB, applies schema, returns a Connection."""
from __future__ import annotations
import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection at db_path, creating + migrating schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    _apply_in_place_migrations(conn)
    return conn


def _apply_in_place_migrations(conn: sqlite3.Connection) -> None:
    """ALTER TABLE for columns added after a brain.db was first created.
    Safe to run repeatedly — each step checks before applying."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(episodes)")}
    if "response_text" not in cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN response_text TEXT")

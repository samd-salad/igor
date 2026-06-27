import sqlite3
from server.external._internal.db import open_db


def test_open_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = open_db(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {row[0] for row in rows}
        assert {"episodes", "facts", "identity", "reflections", "feedback", "reminders"} <= names
    finally:
        conn.close()


def test_open_db_adds_response_text_column_to_legacy_episodes_table(tmp_path):
    """A brain.db created before response_text existed must gain the column
    automatically on next open — otherwise sqlite_persistence._row_to_episode
    fails on the missing column."""
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript("""
        CREATE TABLE episodes (
            episode_id TEXT PRIMARY KEY,
            occurred_at TEXT NOT NULL,
            speaker_id TEXT, participants TEXT, intent TEXT,
            raw_utterance TEXT NOT NULL, tool_calls TEXT,
            emotional_tone TEXT, summary TEXT, consolidated_at TEXT
        );
    """)
    legacy.commit()
    legacy.close()

    conn = open_db(db_path)
    try:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(episodes)")}
        assert "response_text" in cols
    finally:
        conn.close()


def test_open_db_loads_sqlite_vec_and_creates_vec_table(tmp_path):
    db_path = tmp_path / "brain.db"
    conn = open_db(db_path)
    try:
        # vec_version() only exists if extension loaded
        version_row = conn.execute("SELECT vec_version()").fetchone()
        assert version_row is not None
        # facts_vec virtual table exists
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='facts_vec'"
        ).fetchall()
        assert len(rows) == 1
    finally:
        conn.close()

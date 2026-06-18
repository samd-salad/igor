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

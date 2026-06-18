import json
from datetime import datetime, UTC
from pathlib import Path
from server.external.sqlite_persistence import SqlitePersistence
from server.external._internal.brain_json_migration import migrate_brain_json_if_needed


def _make_brain_json(p: Path):
    data = {
        "entries": [
            {
                "id": 1, "type": "memory", "created": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "data": {"category": "prefs", "key": "coffee", "value": "dark roast", "tags": []},
                "tags": ["beverage"],
            },
            {
                "id": 2, "type": "episode", "created": datetime(2026, 1, 1, 11, tzinfo=UTC).isoformat(),
                "data": {"raw_utterance": "what time is it", "summary": "user asked time",
                         "participants": ["sam", "igor"], "tool_calls": []},
            },
            {
                "id": 3, "type": "identity", "created": datetime(2026, 1, 1, 12, tzinfo=UTC).isoformat(),
                "data": {"narrative": "Sam is a homelab nerd."},
            },
        ],
    }
    p.write_text(json.dumps(data), encoding="utf-8")


def test_migration_copies_brain_into_sqlite(tmp_path):
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    _make_brain_json(bj)

    migrate_brain_json_if_needed(bj, db)

    sp = SqlitePersistence(db)
    assert sp.find_fact("prefs", "coffee") is not None
    assert len(sp.list_recent_episodes(10)) == 1
    assert sp.get_identity_narrative() == "Sam is a homelab nerd."
    assert not bj.exists()
    assert list(tmp_path.glob("brain.json.imported-*.bak"))


def test_migration_is_idempotent(tmp_path):
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    _make_brain_json(bj)
    migrate_brain_json_if_needed(bj, db)
    migrate_brain_json_if_needed(bj, db)  # second call no-op
    sp = SqlitePersistence(db)
    assert len(sp.list_recent_episodes(10)) == 1

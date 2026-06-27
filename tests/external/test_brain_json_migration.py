import json
from datetime import datetime, UTC
from pathlib import Path
import pytest
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


def test_summary_entries_become_consolidated_episodes(tmp_path):
    """summary entries are pre-schema-split user turns; map to Episode with raw_utterance=text.
    Mark consolidated_at because the existing identity narrative was already built on top of them
    — re-consolidating would burn Haiku calls on already-digested history."""
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    bj.write_text(json.dumps({"entries": [
        {"id": 10, "type": "summary",
         "created": datetime(2026, 3, 16, 23, 34, 21, tzinfo=UTC).isoformat(),
         "tags": ["conversation", "get_time"],
         "data": {"text": "Hey, good wake word. Um, ask me about myself. [get_time]"}},
    ]}), encoding="utf-8")

    migrate_brain_json_if_needed(bj, db)

    sp = SqlitePersistence(db)
    eps = sp.list_recent_episodes(10)
    assert len(eps) == 1
    assert eps[0].raw_utterance == "Hey, good wake word. Um, ask me about myself. [get_time]"
    assert eps[0].consolidated_at is not None
    assert sp.list_unconsolidated_episodes() == []


def test_reminder_entries_become_reminder_rows(tmp_path):
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    fire_at_ts = 1773720880.0  # 2026-03-17 unix ts
    bj.write_text(json.dumps({"entries": [
        {"id": 20, "type": "reminder",
         "created": datetime(2026, 3, 17, tzinfo=UTC).isoformat(),
         "status": "pending",
         "data": {"name": "trash day", "fire_at": fire_at_ts,
                  "duration_seconds": 30.0, "room_id": "kitchen"}},
        {"id": 21, "type": "reminder",
         "created": datetime(2026, 3, 17, tzinfo=UTC).isoformat(),
         "status": "fired",
         "data": {"name": "timer", "fire_at": fire_at_ts, "room_id": "living_room"}},
    ]}), encoding="utf-8")

    migrate_brain_json_if_needed(bj, db)

    sp = SqlitePersistence(db)
    pending = sp.list_pending_reminders()
    assert len(pending) == 1
    assert pending[0].name == "trash day"
    assert pending[0].room_id == "kitchen"
    # fire_at must be a real datetime, not a float
    assert isinstance(pending[0].fire_at, datetime)


def test_feedback_entries_become_feedback_rows(tmp_path):
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    bj.write_text(json.dumps({"entries": [
        {"id": 30, "type": "feedback",
         "created": datetime(2026, 3, 3, 16, 21, tzinfo=UTC).isoformat(),
         "status": "open",
         "data": {"id": 1, "issue": "Amazon Prime video won't open properly",
                  "suggestion": "", "context": "tv_launch prime video"}},
        {"id": 31, "type": "feedback",
         "created": datetime(2026, 3, 4, tzinfo=UTC).isoformat(),
         "status": "resolved",
         "data": {"id": 2, "issue": "TV sleep timer fired late"}},
    ]}), encoding="utf-8")

    migrate_brain_json_if_needed(bj, db)

    sp = SqlitePersistence(db)
    open_items = sp.list_feedback("open")
    assert len(open_items) == 1
    assert open_items[0].issue == "Amazon Prime video won't open properly"
    assert open_items[0].status == "open"
    assert len(sp.list_feedback("resolved")) == 1


def test_routine_entries_are_dropped_without_error(tmp_path):
    """Routines are tool-call frequency analytics with no schema table — drop them
    explicitly rather than silently skipping or crashing."""
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    bj.write_text(json.dumps({"entries": [
        {"id": 40, "type": "routine",
         "created": datetime(2026, 3, 12, tzinfo=UTC).isoformat(),
         "tags": ["tv_power", "night"],
         "data": {"command": "tv_power", "hour": 1, "day": 0}},
        {"id": 41, "type": "memory",
         "created": datetime(2026, 3, 12, tzinfo=UTC).isoformat(),
         "data": {"category": "prefs", "key": "tea", "value": "earl grey"},
         "tags": []},
    ]}), encoding="utf-8")

    migrate_brain_json_if_needed(bj, db)

    sp = SqlitePersistence(db)
    # routine dropped, memory survives
    assert sp.find_fact("prefs", "tea") is not None
    assert sp.find_fact("prefs", "tea").value == "earl grey"


def test_brain_json_is_backed_up_before_migration(tmp_path):
    """Safety: a verbatim copy of brain.json exists at .json.bak the moment migration
    starts touching the DB, in case anything downstream corrupts state."""
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    _make_brain_json(bj)
    original = bj.read_text(encoding="utf-8")

    migrate_brain_json_if_needed(bj, db)

    backups = list(tmp_path.glob("brain.json.pre-migration-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original


def test_migration_aborts_without_renaming_when_db_write_fails(tmp_path, monkeypatch):
    """If any persistence write raises, brain.json must NOT be renamed to .imported-*.bak —
    we need the source intact so the user can re-run after fixing the cause."""
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    _make_brain_json(bj)

    # Make save_fact blow up to simulate a partial-write failure.
    real_save_fact = SqlitePersistence.save_fact

    def boom(self, fact):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(SqlitePersistence, "save_fact", boom)

    with pytest.raises(RuntimeError):
        migrate_brain_json_if_needed(bj, db)

    # brain.json must still be there. .imported-*.bak must NOT exist.
    assert bj.exists()
    assert not list(tmp_path.glob("brain.json.imported-*.bak"))


def test_full_brain_json_migrates_with_correct_counts(tmp_path):
    """End-to-end: mixed brain.json with every supported type plus dropped routines.
    Validates that every retained type lands in its target table."""
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    bj.write_text(json.dumps({"entries": [
        # 2 memories
        {"id": 1, "type": "memory", "created": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
         "data": {"category": "prefs", "key": "coffee", "value": "dark roast"}, "tags": []},
        {"id": 2, "type": "memory", "created": datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
         "data": {"category": "prefs", "key": "tea", "value": "earl grey"}, "tags": []},
        # 1 episode
        {"id": 3, "type": "episode", "created": datetime(2026, 1, 3, tzinfo=UTC).isoformat(),
         "data": {"raw_utterance": "what time is it", "summary": "user asked time",
                  "participants": ["sam"], "tool_calls": []}},
        # 1 identity
        {"id": 4, "type": "identity", "created": datetime(2026, 1, 4, tzinfo=UTC).isoformat(),
         "data": {"narrative": "Sam likes dark roast and earl grey."}},
        # 1 summary -> episode (consolidated)
        {"id": 5, "type": "summary", "created": datetime(2026, 1, 5, tzinfo=UTC).isoformat(),
         "data": {"text": "Hey Igor, what's the weather"}, "tags": []},
        # 1 reminder
        {"id": 6, "type": "reminder", "created": datetime(2026, 1, 6, tzinfo=UTC).isoformat(),
         "status": "pending",
         "data": {"name": "feed cat", "fire_at": 1773720880.0, "room_id": "kitchen"}},
        # 1 feedback
        {"id": 7, "type": "feedback", "created": datetime(2026, 1, 7, tzinfo=UTC).isoformat(),
         "status": "open", "data": {"issue": "TV won't open Prime"}},
        # 2 routines (dropped)
        {"id": 8, "type": "routine", "created": datetime(2026, 1, 8, tzinfo=UTC).isoformat(),
         "data": {"command": "tv_power"}, "tags": []},
        {"id": 9, "type": "routine", "created": datetime(2026, 1, 9, tzinfo=UTC).isoformat(),
         "data": {"command": "set_light"}, "tags": []},
    ]}), encoding="utf-8")

    migrate_brain_json_if_needed(bj, db)

    sp = SqlitePersistence(db)
    # 2 facts
    assert sp.find_fact("prefs", "coffee") is not None
    assert sp.find_fact("prefs", "tea") is not None
    # 1 episode + 1 summary -> 2 episodes total
    assert len(sp.list_recent_episodes(10)) == 2
    # identity survived
    assert sp.get_identity_narrative() == "Sam likes dark roast and earl grey."
    # 1 reminder pending
    assert len(sp.list_pending_reminders()) == 1
    # 1 open feedback
    assert len(sp.list_feedback("open")) == 1

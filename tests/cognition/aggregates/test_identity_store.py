from datetime import datetime, UTC
from server.cognition.aggregates.identity import IdentityStore
from server.external.sqlite_persistence import SqlitePersistence


def test_narrative_default_empty(tmp_path):
    ids = IdentityStore(SqlitePersistence(tmp_path / "brain.db"))
    assert ids.get_narrative() == ""


def test_replace_narrative_and_track_consolidation(tmp_path):
    ids = IdentityStore(SqlitePersistence(tmp_path / "brain.db"))
    ids.replace_narrative("Sam likes coffee.",
                          last_consolidated_at=datetime(2026, 1, 1, tzinfo=UTC),
                          last_consolidated_episode_id="ep-5")
    assert ids.get_narrative() == "Sam likes coffee."
    assert ids.get_last_consolidated_episode_id() == "ep-5"


def test_log_reflection(tmp_path):
    ids = IdentityStore(SqlitePersistence(tmp_path / "brain.db"))
    ids.log_reflection("preambles too long",
                       at=datetime(2026, 1, 1, tzinfo=UTC),
                       source_episode_id="ep-1")
    assert any("preambles" in r.note for r in ids.list_recent_reflections(5))

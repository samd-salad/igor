from datetime import datetime, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.contracts import Episode
from server.external.sqlite_persistence import SqlitePersistence


def _ep(eid: str, minute: int = 0) -> Episode:
    return Episode(
        episode_id=eid,
        occurred_at=datetime(2026, 1, 1, 10, minute, tzinfo=UTC),
        speaker_id=None, participants=[], intent=None,
        raw_utterance="x", tool_calls=[], emotional_tone=None,
        summary=None, consolidated_at=None,
    )


def test_add_and_recent(tmp_path):
    es = EpisodeStore(SqlitePersistence(tmp_path / "brain.db"))
    es.add(_ep("ep-1", 0))
    es.add(_ep("ep-2", 5))
    recent = es.get_recent(10)
    assert [e.episode_id for e in recent] == ["ep-2", "ep-1"]


def test_consolidation_flow(tmp_path):
    es = EpisodeStore(SqlitePersistence(tmp_path / "brain.db"))
    for i in range(3):
        es.add(_ep(f"ep-{i}", i))
    assert len(es.get_unconsolidated()) == 3
    es.mark_consolidated(["ep-0", "ep-1"], at=datetime(2026, 1, 2, tzinfo=UTC))
    assert len(es.get_unconsolidated()) == 1

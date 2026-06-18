from datetime import datetime, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import Episode
from server.external.sqlite_persistence import SqlitePersistence


def _seed_episode(sp, episode_id: str, minute: int = 0) -> None:
    EpisodeStore(sp).add(Episode(
        episode_id=episode_id,
        occurred_at=datetime(2026, 1, 1, 10, minute, tzinfo=UTC),
        speaker_id=None, participants=[], intent=None,
        raw_utterance="seed", tool_calls=[], emotional_tone=None,
        summary=None, consolidated_at=None,
    ))


def test_save_uses_episode_provenance(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    _seed_episode(sp, "ep-1")
    mem = MemoryStore(sp)
    mem.save_fact(
        category="prefs", key="coffee", value="dark roast",
        tags=["beverage"], source_episode_id="ep-1",
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    found = mem.find_fact("prefs", "coffee")
    assert found is not None
    assert found.source_episode_id == "ep-1"


def test_invalidate_then_replace(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    mem = MemoryStore(sp)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 6, 1, tzinfo=UTC)
    mem.save_fact("prefs", "coffee", "milk only", [], None, t0)
    mem.update_fact("prefs", "coffee", "dark roast oat milk", [], None, t1)
    active = mem.list_active()
    matches = [f for f in active if f.category == "prefs" and f.key == "coffee"]
    assert len(matches) == 1
    assert matches[0].value == "dark roast oat milk"

from datetime import datetime, UTC
from server.cognition.aggregates.memory import MemoryStore
from server.external.sqlite_persistence import SqlitePersistence


def test_save_uses_episode_provenance(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
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
    mem.save_fact("prefs", "coffee", "milk only", [], "ep-1", t0)
    mem.update_fact("prefs", "coffee", "dark roast oat milk", [], "ep-2", t1)
    active = mem.list_active()
    matches = [f for f in active if f.category == "prefs" and f.key == "coffee"]
    assert len(matches) == 1
    assert matches[0].value == "dark roast oat milk"

from datetime import datetime, UTC
from server.cognition.contracts import Fact
from server.external.sqlite_persistence import SqlitePersistence


def _make_fact(fact_id="f-1", category="prefs", key="coffee",
               value="dark roast", invalid_at=None, episode_id=None):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Fact(
        fact_id=fact_id, category=category, key=key, value=value,
        tags=["beverage"], source_episode_id=episode_id,
        embedding=None,
        valid_at=now, invalid_at=invalid_at, created_at=now,
    )


def test_save_and_find_active_fact(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    sp.save_fact(_make_fact())
    found = sp.find_fact("prefs", "coffee")
    assert found is not None and found.value == "dark roast"


def test_invalidate_fact_excludes_from_active(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    sp.save_fact(_make_fact())
    sp.invalidate_fact("f-1", at=datetime(2026, 6, 1, tzinfo=UTC))
    active = sp.list_active_facts()
    assert not any(f.fact_id == "f-1" for f in active)
    raw = sp._conn.execute("SELECT invalid_at FROM facts WHERE fact_id='f-1'").fetchone()
    assert raw["invalid_at"] is not None

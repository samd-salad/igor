import struct
from datetime import datetime, UTC
import uuid
from server.cognition.contracts import Fact
from server.external.sqlite_persistence import SqlitePersistence
from server.external.vector_store import VectorStore


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


def test_naive_timestamps_in_db_normalize_to_utc_on_read(tmp_path):
    """Migration from brain.json wrote naive ISO strings. Reads must repair them
    so downstream code (e.g. TagRetrieval recency math) can subtract from
    datetime.now(UTC) without TypeError."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    sp._conn.execute(
        """INSERT INTO facts (fact_id, category, key, value, tags,
                              source_episode_id, embedding,
                              valid_at, invalid_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("f-naive", "prefs", "coffee", "dark roast", "[]", None, None,
         "2026-01-01T00:00:00",  # naive, no tzinfo
         None,
         "2026-01-01T00:00:00"),  # naive, no tzinfo
    )
    [f] = sp.list_active_facts()
    assert f.created_at.tzinfo is not None
    assert f.valid_at.tzinfo is not None
    now = datetime(2026, 6, 1, tzinfo=UTC)
    delta = now - f.created_at  # would TypeError pre-fix
    assert delta.days > 0


def test_save_fact_with_none_embedding_stores_null_and_skips_vec_mirror(tmp_path):
    """Persistence is encoder-agnostic — if Fact.embedding is None it stores
    null and does NOT touch the vec sidecar. The writer is responsible for
    encoding upstream; persistence just persists."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    fid = str(uuid.uuid4())
    now = datetime.now(UTC)
    sp.save_fact(Fact(
        fact_id=fid, category="prefs", key="coffee", value="dark roast",
        tags=[], source_episode_id=None, embedding=None,
        valid_at=now, invalid_at=None, created_at=now,
    ))
    stored = sp.find_fact("prefs", "coffee")
    assert stored is not None
    assert stored.embedding is None
    # Nothing landed in the vec sidecar — we never had bytes to mirror.
    raw = sp._conn.execute(
        "SELECT count(*) AS n FROM facts_vec WHERE fact_id = ?", (fid,)
    ).fetchone()
    assert raw["n"] == 0


def test_save_fact_with_embedding_mirrors_to_vec_index(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    pre = struct.pack("<384f", *([0.123] * 384))
    fid = str(uuid.uuid4())
    now = datetime.now(UTC)
    sp.save_fact(Fact(
        fact_id=fid, category="prefs", key="tea", value="earl grey",
        tags=[], source_episode_id=None, embedding=pre,
        valid_at=now, invalid_at=None, created_at=now,
    ))
    stored = sp.find_fact("prefs", "tea")
    assert stored.embedding == pre
    # vec sidecar mirrors the bytes
    vs = VectorStore(sp._conn)
    assert vs.search(pre, top_k=1) == [fid]


def test_find_fact_by_id_returns_the_fact(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    fid = str(uuid.uuid4())
    now = datetime.now(UTC)
    sp.save_fact(Fact(
        fact_id=fid, category="prefs", key="coffee", value="dark roast",
        tags=[], source_episode_id=None, embedding=None,
        valid_at=now, invalid_at=None, created_at=now,
    ))
    found = sp.find_fact_by_id(fid)
    assert found is not None
    assert found.fact_id == fid
    assert found.value == "dark roast"


def test_find_fact_by_id_returns_none_when_missing(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    assert sp.find_fact_by_id("nonexistent") is None

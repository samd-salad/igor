import struct
from datetime import datetime, UTC
import uuid
from server.cognition.contracts import Fact
from server.external.sqlite_persistence import SqlitePersistence
from server.external.embedding_encoder import EmbeddingEncoder
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


def test_save_fact_populates_embedding_blob_when_none_and_mirrors_to_vec_index(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db", encoder=EmbeddingEncoder())
    fact = Fact(
        fact_id=str(uuid.uuid4()),
        category="prefs",
        key="coffee",
        value="dark roast",
        tags=[],
        source_episode_id=None,
        embedding=None,   # encoder should fill this in
        valid_at=datetime.now(UTC),
        invalid_at=None,
        created_at=datetime.now(UTC),
    )
    sp.save_fact(fact)

    # canonical store populated
    stored = sp.find_fact("prefs", "coffee")
    assert stored is not None
    assert stored.embedding is not None
    assert len(stored.embedding) == 384 * 4

    # vec sidecar mirrors it
    vs = VectorStore(sp._conn)
    hits = vs.search(stored.embedding, top_k=1)
    assert hits[0][0] == fact.fact_id


def test_save_fact_respects_precomputed_embedding(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db", encoder=EmbeddingEncoder())
    pre = struct.pack("<384f", *([0.123] * 384))
    fact = Fact(
        fact_id=str(uuid.uuid4()),
        category="prefs",
        key="tea",
        value="earl grey",
        tags=[],
        source_episode_id=None,
        embedding=pre,
        valid_at=datetime.now(UTC),
        invalid_at=None,
        created_at=datetime.now(UTC),
    )
    sp.save_fact(fact)
    stored = sp.find_fact("prefs", "tea")
    assert stored.embedding == pre

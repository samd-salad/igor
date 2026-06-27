import struct
import time
import uuid
from datetime import datetime, UTC
import pytest

from server.cognition.contracts import Fact
from server.external.async_fact_writer import AsyncFactWriter
from server.external.embedding_encoder import EmbeddingEncoder
from server.external.sqlite_persistence import SqlitePersistence
from server.external.vector_store import VectorStore


def _fact(value: str, embedding: bytes | None = None) -> Fact:
    now = datetime.now(UTC)
    return Fact(
        fact_id=str(uuid.uuid4()),
        category="prefs", key=f"k_{uuid.uuid4().hex[:6]}", value=value, tags=[],
        source_episode_id=None, embedding=embedding,
        valid_at=now, invalid_at=None, created_at=now,
    )


@pytest.fixture
def sp(tmp_path):
    return SqlitePersistence(tmp_path / "brain.db")


def test_enqueue_returns_fast(sp):
    writer = AsyncFactWriter(sp, encoder=EmbeddingEncoder())
    try:
        facts = [_fact("dark roast coffee") for _ in range(10)]
        t0 = time.perf_counter()
        for f in facts:
            writer.enqueue(f)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50, f"enqueue too slow: {elapsed_ms} ms"
    finally:
        writer.shutdown()


def test_facts_eventually_land_in_persistence(sp):
    writer = AsyncFactWriter(sp, encoder=EmbeddingEncoder())
    f = _fact("dark roast coffee")
    writer.enqueue(f)
    writer.flush(timeout=10.0)
    writer.shutdown()
    assert sp.find_fact("prefs", f.key) is not None


def test_shutdown_drains_pending(sp):
    writer = AsyncFactWriter(sp, encoder=EmbeddingEncoder())
    facts = [_fact(f"value {i}") for i in range(5)]
    for f in facts:
        writer.enqueue(f)
    writer.shutdown(timeout=10.0)
    for f in facts:
        assert sp.find_fact("prefs", f.key) is not None


def test_writer_encodes_facts_when_encoder_provided(sp):
    """Writer owns the encoder. A Fact arriving with embedding=None gets
    encoded by the writer before persistence sees it, and the resulting
    bytes land in both the canonical store and the vec sidecar."""
    writer = AsyncFactWriter(sp, encoder=EmbeddingEncoder())
    f = _fact("dark roast coffee")
    writer.enqueue(f)
    writer.flush(timeout=10.0)
    writer.shutdown()

    stored = sp.find_fact("prefs", f.key)
    assert stored is not None
    assert stored.embedding is not None
    assert len(stored.embedding) == 384 * 4
    # vec sidecar got the same bytes
    assert VectorStore(sp._conn).search(stored.embedding, top_k=1) == [f.fact_id]


def test_writer_respects_precomputed_embedding(sp):
    """If Fact arrives with embedding already set, writer does NOT re-encode."""
    pre = struct.pack("<384f", *([0.5] * 384))
    writer = AsyncFactWriter(sp, encoder=EmbeddingEncoder())
    f = _fact("earl grey", embedding=pre)
    writer.enqueue(f)
    writer.flush(timeout=10.0)
    writer.shutdown()

    stored = sp.find_fact("prefs", f.key)
    assert stored.embedding == pre


def test_writer_without_encoder_persists_with_null_embedding(sp):
    """Encoder is optional. Without one, the writer just persists what it
    receives — embedding stays None, no vec mirror happens."""
    writer = AsyncFactWriter(sp)  # no encoder
    f = _fact("dark roast coffee")
    writer.enqueue(f)
    writer.flush(timeout=10.0)
    writer.shutdown()

    stored = sp.find_fact("prefs", f.key)
    assert stored is not None
    assert stored.embedding is None

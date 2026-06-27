import time
import uuid
from datetime import datetime, UTC
import pytest

from server.cognition.contracts import Fact
from server.external.sqlite_persistence import SqlitePersistence
from server.external.embedding_encoder import EmbeddingEncoder
from server.external.async_fact_writer import AsyncFactWriter


def _fact(value: str) -> Fact:
    now = datetime.now(UTC)
    return Fact(
        fact_id=str(uuid.uuid4()),
        category="prefs", key=f"k_{uuid.uuid4().hex[:6]}", value=value, tags=[],
        source_episode_id=None, embedding=None,
        valid_at=now, invalid_at=None, created_at=now,
    )


@pytest.fixture
def sp(tmp_path):
    return SqlitePersistence(tmp_path / "brain.db", encoder=EmbeddingEncoder())


def test_enqueue_returns_fast(sp):
    writer = AsyncFactWriter(sp)
    try:
        t0 = time.perf_counter()
        for _ in range(10):
            writer.enqueue(_fact("dark roast coffee"))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50, f"enqueue too slow: {elapsed_ms} ms"
    finally:
        writer.shutdown()


def test_facts_eventually_land_in_persistence(sp):
    writer = AsyncFactWriter(sp)
    f = _fact("dark roast coffee")
    writer.enqueue(f)
    writer.flush(timeout=10.0)
    writer.shutdown()
    assert sp.find_fact("prefs", f.key) is not None


def test_shutdown_drains_pending(sp):
    writer = AsyncFactWriter(sp)
    facts = [_fact(f"value {i}") for i in range(5)]
    for f in facts:
        writer.enqueue(f)
    writer.shutdown(timeout=10.0)
    for f in facts:
        assert sp.find_fact("prefs", f.key) is not None

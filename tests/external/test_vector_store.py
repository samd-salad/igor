import struct
from pathlib import Path
import pytest
from server.external._internal.db import open_db
from server.external.vector_store import VectorStore


def _vec(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _zero_vec(dim: int = 384, value: float = 0.0) -> bytes:
    return struct.pack(f"<{dim}f", *([value] * dim))


@pytest.fixture
def store(tmp_path):
    conn = open_db(tmp_path / "brain.db")
    return VectorStore(conn)


def test_upsert_then_search_returns_inserted_fact(store):
    emb = _zero_vec(value=0.5)
    store.upsert("fact-1", emb)
    results = store.search(emb, top_k=5)
    assert results == ["fact-1"]


def test_search_orders_by_distance_nearest_first(store):
    a = _zero_vec(value=0.5)
    b = _zero_vec(value=-0.5)
    store.upsert("near", a)
    store.upsert("far", b)
    # Query for 'a' — 'near' must come back before 'far'.
    assert store.search(a, top_k=2) == ["near", "far"]


def test_upsert_replaces_existing_fact_id(store):
    store.upsert("fact-1", _zero_vec(value=0.5))
    store.upsert("fact-1", _zero_vec(value=-0.5))  # overwrite
    assert store.search(_zero_vec(value=-0.5), top_k=1) == ["fact-1"]


def test_delete_removes_fact(store):
    store.upsert("fact-1", _zero_vec(value=0.5))
    store.delete("fact-1")
    assert store.search(_zero_vec(value=0.5), top_k=1) == []

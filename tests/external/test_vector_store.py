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
    assert len(results) == 1
    fact_id, distance = results[0]
    assert fact_id == "fact-1"
    assert distance == pytest.approx(0.0, abs=1e-3)


def test_search_orders_by_distance(store):
    a = _zero_vec(value=0.5)
    b = _zero_vec(value=-0.5)
    store.upsert("near", a)
    store.upsert("far", b)
    results = store.search(a, top_k=2)
    assert [fid for fid, _ in results] == ["near", "far"]
    assert results[0][1] < results[1][1]


def test_upsert_replaces_existing_fact_id(store):
    store.upsert("fact-1", _zero_vec(value=0.5))
    store.upsert("fact-1", _zero_vec(value=-0.5))  # overwrite
    results = store.search(_zero_vec(value=-0.5), top_k=1)
    assert results[0][0] == "fact-1"
    assert results[0][1] == pytest.approx(0.0, abs=1e-3)


def test_delete_removes_fact(store):
    store.upsert("fact-1", _zero_vec(value=0.5))
    store.delete("fact-1")
    results = store.search(_zero_vec(value=0.5), top_k=1)
    assert results == []

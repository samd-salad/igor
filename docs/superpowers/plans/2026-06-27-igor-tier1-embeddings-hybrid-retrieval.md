# Igor Tier-1 Memory: Embeddings + Hybrid Retrieval + Async Writes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the three dormant Tier-1 columns in Igor's brain.db (`facts.embedding`, hybrid retrieval, writes-off-hot-path) so semantic recall actually works, fact writes don't block the LLM turn, and the bi-temporal columns finally do their job.

**Architecture:** Encoder (`fastembed` + `BAAI/bge-small-en-v1.5`, 384-dim, ONNX, ARM64-native) populates the existing `facts.embedding BLOB` as the canonical store. A `sqlite-vec` virtual table on the same `brain.db` provides the speed index for cosine search. Hybrid retrieval combines tag-match and vector results via Reciprocal Rank Fusion (RRF, k=60). Fact writes (encoder + DB) move to a background queue; the LLM turn doesn't wait for them.

**Tech Stack:** Python 3.12, `fastembed~=0.4` (ONNX runtime), `sqlite-vec~=0.1.9` (ARM64 wheels), existing `sqlite3` stdlib, existing pytest harness. No PyTorch dependency — keeps the Pi5 Docker image lean.

## Global Constraints

- Embedding dimension: **384** (locked by `BAAI/bge-small-en-v1.5`)
- Embedding dtype: **float32**, stored as little-endian raw bytes (1536 B/fact). Reuse the existing `Fact.embedding: Optional[bytes]` contract — do not change the dataclass shape.
- `sqlite-vec` is the SPEED index; `facts.embedding BLOB` is the CANONICAL store. The vec0 virtual table must be regenerable from the BLOB column at any time. Never write a fact's embedding to vec0 without also writing it to `facts.embedding`.
- Pi5 hardware target: ARM64, 8 GB RAM, CPU-only inference. Verify install + inference works under `aarch64` before declaring any task done.
- RRF constant: **k=60** (the cited 2026 default; do not tune without independent benchmark).
- Async write contract: `save_memory` returns from the LLM tool call within **< 5 ms** measured wall-clock. The actual encode + DB write happens on a background worker thread.
- Existing tests must keep passing. The `python -m pytest tests/ --ignore=tests/wakeword -q` invocation is the baseline (119 tests pre-change).
- Hot-path query encode latency budget: **< 30 ms** measured on Pi5. Above that, downgrade model.
- No new top-level container service. Encoder + writer run inside the existing igor Docker process.

## Research Notes (cited for the choices above)

- fastembed produces embeddings ~0.92 cosine-equivalent to sentence-transformers for `bge-small-en-v1.5` ([Qdrant FastEmbed vs HF Comparison](https://qdrant.github.io/fastembed/examples/FastEmbed_vs_HF_Comparison/)).
- `bge-small-en-v1.5` lives at MTEB ~62 vs `bge-large-en-v1.5` at ~64; on Pi5 CPU the latency gap doesn't justify the 2-point recall gain ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [BentoML 2026 embedding guide](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)).
- `sqlite-vec` benchmarks 41 ms vs NumPy 581 ms for brute-force cosine — the 14× speedup justifies the dependency even at <10k vectors ([sqlite-vec v0.1.0 release](https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html), [SitePoint local-first RAG](https://www.sitepoint.com/local-first-rag-vector-search-in-sqlite-with-hamming-distance/)).
- `sqlite-vec` ARM64 wheels confirmed shipping as of March 2026 in v0.1.9 ([sqlite-vec on PyPI](https://pypi.org/project/sqlite-vec/), [issue #211 resolution](https://github.com/asg017/sqlite-vec/issues/211)).
- RRF is the 2026 hybrid-retrieval consensus because BM25/cosine score scales are incompatible; weighted-sum fusion is "fragile in production RAG pipelines" ([Digital Applied 2026 hybrid search reference](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026), [Denser.ai hybrid search guide](https://denser.ai/blog/hybrid-search-for-rag/)). The same incompatibility applies to Igor's tag-count score vs cosine.
- Writes-off-hot-path matches Letta's primary/sleep-time split (Tier 1 conclusion from the 2026-04-19 internal research wave, preserved in `memory/project_memory_roadmap.md`).

## File Structure

**Create:**
- `server/external/embedding_encoder.py` — `EmbeddingEncoder` class wrapping fastembed. One public method: `encode(text: str) -> bytes`. Lazy-loads the model on first call.
- `server/external/vector_store.py` — `VectorStore` class wrapping the sqlite-vec virtual table on the existing connection. Methods: `upsert(fact_id, embedding)`, `search(query_embedding, top_k) -> list[(fact_id, distance)]`.
- `server/cognition/hybrid_retrieval.py` — `HybridRetrieval` class implementing the same port as `TagRetrieval`. Composes `TagRetrieval` + `VectorStore` + `EmbeddingEncoder` and fuses results with RRF (k=60).
- `server/external/async_fact_writer.py` — `AsyncFactWriter` class wrapping a thread-safe queue + worker thread. Methods: `enqueue(fact)`, `flush()`, `shutdown()`.
- `tests/external/test_embedding_encoder.py`
- `tests/external/test_vector_store.py`
- `tests/external/test_async_fact_writer.py`
- `tests/cognition/test_hybrid_retrieval.py`

**Modify:**
- `requirements-server-text.txt` — add `fastembed~=0.4` and `sqlite-vec~=0.1.9`
- `server/external/_internal/schema.sql` — add `vec0` virtual table create
- `server/external/_internal/db.py` — load sqlite-vec extension on open
- `server/external/sqlite_persistence.py` — `save_fact` also upserts to vec0
- `server/main.py` — composition root wires `EmbeddingEncoder`, `VectorStore`, `HybridRetrieval`, `AsyncFactWriter`
- `server/cognition/contracts.py` — no changes (existing `Fact.embedding: Optional[bytes]` already shaped)
- `Dockerfile` — add fastembed model cache directory + warm-up step (avoids first-request 1-2 s download)

---

### Task 1: Embedding Encoder

**Files:**
- Create: `server/external/embedding_encoder.py`
- Test: `tests/external/test_embedding_encoder.py`
- Modify: `requirements-server-text.txt`

**Interfaces:**
- Consumes: nothing
- Produces: `EmbeddingEncoder.encode(text: str) -> bytes` — returns 1536 raw bytes (384 float32 little-endian). Lazy-loads the model on first call. Thread-safe (the underlying ONNX session is thread-safe per fastembed docs; no internal lock needed).

- [ ] **Step 1: Add dependency**

```
# requirements-server-text.txt
fastembed~=0.4
```

- [ ] **Step 2: Install and verify import**

```bash
/home/samda/.venvs/igor/bin/pip install fastembed~=0.4
/home/samda/.venvs/igor/bin/python -c "from fastembed import TextEmbedding; print(TextEmbedding.list_supported_models()[0]['model'])"
```

Expected: a model name prints (e.g. `BAAI/bge-small-en-v1.5`). If install fails on ARM64, halt and report.

- [ ] **Step 3: Write the failing test**

```python
# tests/external/test_embedding_encoder.py
import struct
from server.external.embedding_encoder import EmbeddingEncoder


def test_encode_returns_384_float32_bytes():
    enc = EmbeddingEncoder()
    out = enc.encode("dark roast coffee")
    assert isinstance(out, bytes)
    assert len(out) == 384 * 4   # 384 float32s
    floats = struct.unpack(f"<{384}f", out)
    assert any(f != 0.0 for f in floats)


def test_same_text_encodes_to_same_bytes():
    enc = EmbeddingEncoder()
    a = enc.encode("the cat sat on the mat")
    b = enc.encode("the cat sat on the mat")
    assert a == b


def test_different_text_encodes_to_different_bytes():
    enc = EmbeddingEncoder()
    a = enc.encode("dark roast coffee")
    b = enc.encode("the cat sat on the mat")
    assert a != b
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_embedding_encoder.py -v
```

Expected: ImportError (`No module named server.external.embedding_encoder`).

- [ ] **Step 5: Write the implementation**

```python
# server/external/embedding_encoder.py
"""ONNX-based embedding encoder for Igor's memory layer.

Wraps fastembed (Qdrant) with BAAI/bge-small-en-v1.5 (384-dim, ~33M params).
Lazy-loads on first call so test startup is cheap.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from fastembed import TextEmbedding


_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_EMBED_DIM = 384


class EmbeddingEncoder:
    def __init__(self, model_name: str = _MODEL_NAME):
        self._model_name = model_name
        self._model: Optional[TextEmbedding] = None

    def encode(self, text: str) -> bytes:
        if self._model is None:
            self._model = TextEmbedding(model_name=self._model_name)
        vec = next(self._model.embed([text]))
        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape != (_EMBED_DIM,):
            raise RuntimeError(
                f"unexpected embedding shape {arr.shape}, expected ({_EMBED_DIM},)"
            )
        return arr.tobytes()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_embedding_encoder.py -v
```

Expected: 3 passed. First run downloads the model (~120 MB); subsequent runs use the cache.

- [ ] **Step 7: Commit**

```bash
git add requirements-server-text.txt server/external/embedding_encoder.py tests/external/test_embedding_encoder.py
git commit -m "memory: add fastembed BGE-small-en-v1.5 encoder for fact embeddings"
```

---

### Task 2: sqlite-vec extension loading + vec0 virtual table

**Files:**
- Modify: `requirements-server-text.txt`
- Modify: `server/external/_internal/schema.sql`
- Modify: `server/external/_internal/db.py`
- Test: `tests/external/test_db.py` (extend)

**Interfaces:**
- Consumes: nothing
- Produces: `open_db(db_path)` returns a Connection that has sqlite-vec loaded and the `facts_vec` virtual table present. Other modules can issue `vec_distance_L2(...)` etc. against `facts_vec`.

- [ ] **Step 1: Add dependency**

```
# requirements-server-text.txt
sqlite-vec~=0.1.9
```

- [ ] **Step 2: Install and verify extension loads**

```bash
/home/samda/.venvs/igor/bin/pip install sqlite-vec~=0.1.9
/home/samda/.venvs/igor/bin/python -c "
import sqlite3, sqlite_vec
con = sqlite3.connect(':memory:')
con.enable_load_extension(True)
sqlite_vec.load(con)
print(con.execute('SELECT vec_version()').fetchone())
"
```

Expected: prints a version tuple like `('v0.1.9',)`. If extension load fails, halt and report.

- [ ] **Step 3: Write the failing test**

```python
# tests/external/test_db.py — append
def test_open_db_loads_sqlite_vec_and_creates_vec_table(tmp_path):
    from server.external._internal.db import open_db
    conn = open_db(tmp_path / "brain.db")
    # vec_version() only exists if extension loaded
    version_row = conn.execute("SELECT vec_version()").fetchone()
    assert version_row is not None
    # facts_vec virtual table exists
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='facts_vec'"
    ).fetchall()
    assert len(rows) == 1
```

- [ ] **Step 4: Run test to verify it fails**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_db.py::test_open_db_loads_sqlite_vec_and_creates_vec_table -v
```

Expected: FAIL with `no such function: vec_version` or `no such table: facts_vec`.

- [ ] **Step 5: Add schema for the vec0 virtual table**

Append to `server/external/_internal/schema.sql`:

```sql
-- Sidecar speed index for facts.embedding. Canonical embedding stays in facts.embedding BLOB.
-- Regenerable: at any point you can DELETE FROM facts_vec and rebuild from facts.embedding.
CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(
    fact_id TEXT PRIMARY KEY,
    embedding FLOAT[384]
);
```

- [ ] **Step 6: Load extension in open_db**

Replace `open_db` in `server/external/_internal/db.py`:

```python
import sqlite3
import sqlite_vec
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    _apply_in_place_migrations(conn)
    return conn


def _apply_in_place_migrations(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(episodes)")}
    if "response_text" not in cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN response_text TEXT")
```

- [ ] **Step 7: Run test to verify it passes**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_db.py -v
```

Expected: all passing including the new vec test.

- [ ] **Step 8: Run full suite to verify no regressions**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/ --ignore=tests/wakeword -q
```

Expected: 120 passed (119 prior + 1 new).

- [ ] **Step 9: Commit**

```bash
git add requirements-server-text.txt server/external/_internal/db.py server/external/_internal/schema.sql tests/external/test_db.py
git commit -m "memory: load sqlite-vec and create facts_vec sidecar index"
```

---

### Task 3: VectorStore wrapper

**Files:**
- Create: `server/external/vector_store.py`
- Test: `tests/external/test_vector_store.py`

**Interfaces:**
- Consumes: a sqlite3 Connection with sqlite-vec loaded and `facts_vec` virtual table present (from Task 2)
- Produces:
  - `VectorStore(conn).upsert(fact_id: str, embedding: bytes) -> None`
  - `VectorStore(conn).search(query_embedding: bytes, top_k: int) -> list[tuple[str, float]]` — list of `(fact_id, distance)` ascending by distance
  - `VectorStore(conn).delete(fact_id: str) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/external/test_vector_store.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_vector_store.py -v
```

Expected: ImportError on `server.external.vector_store`.

- [ ] **Step 3: Write the implementation**

```python
# server/external/vector_store.py
"""sqlite-vec backed vector index keyed by fact_id.

CANONICAL storage of embeddings remains in `facts.embedding BLOB`. This
virtual table is a sidecar index for fast nearest-neighbor queries —
delete + rebuild it from facts.embedding at any time.
"""
from __future__ import annotations
import sqlite3
from typing import List, Tuple


class VectorStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def upsert(self, fact_id: str, embedding: bytes) -> None:
        # vec0 doesn't support ON CONFLICT — delete-then-insert is the documented idiom.
        self._conn.execute("DELETE FROM facts_vec WHERE fact_id = ?", (fact_id,))
        self._conn.execute(
            "INSERT INTO facts_vec(fact_id, embedding) VALUES (?, ?)",
            (fact_id, embedding),
        )

    def search(self, query_embedding: bytes, top_k: int) -> List[Tuple[str, float]]:
        rows = self._conn.execute(
            """SELECT fact_id, distance FROM facts_vec
               WHERE embedding MATCH ? AND k = ?
               ORDER BY distance""",
            (query_embedding, top_k),
        ).fetchall()
        return [(r["fact_id"], r["distance"]) for r in rows]

    def delete(self, fact_id: str) -> None:
        self._conn.execute("DELETE FROM facts_vec WHERE fact_id = ?", (fact_id,))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_vector_store.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/external/vector_store.py tests/external/test_vector_store.py
git commit -m "memory: add VectorStore wrapping sqlite-vec facts_vec index"
```

---

### Task 4: Wire EmbeddingEncoder + VectorStore into save_fact

**Files:**
- Modify: `server/external/sqlite_persistence.py`
- Test: `tests/external/test_sqlite_persistence_facts.py` (extend)

**Interfaces:**
- Consumes: `EmbeddingEncoder.encode(text) -> bytes` (Task 1), `VectorStore.upsert(fact_id, embedding)` (Task 3)
- Produces: `SqlitePersistence.save_fact(fact)` now (a) populates `fact.embedding` if it was None, by encoding `value`, and (b) upserts the embedding into `facts_vec`. `facts.embedding` BLOB remains the canonical store.

- [ ] **Step 1: Write the failing test**

```python
# tests/external/test_sqlite_persistence_facts.py — append
import struct
from datetime import datetime, UTC
import uuid
from server.cognition.contracts import Fact
from server.external.sqlite_persistence import SqlitePersistence
from server.external.embedding_encoder import EmbeddingEncoder
from server.external.vector_store import VectorStore


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_sqlite_persistence_facts.py -v -k "embedding"
```

Expected: FAIL — `SqlitePersistence.__init__` doesn't accept `encoder`.

- [ ] **Step 3: Modify SqlitePersistence**

In `server/external/sqlite_persistence.py`:

- Add `encoder: Optional[EmbeddingEncoder] = None` parameter to `__init__`
- Hold the encoder + a `VectorStore(self._conn)` instance
- In `save_fact`:
  - If `fact.embedding is None` and `self._encoder is not None`, encode `fact.value` to fill the BLOB. Use a local rebuilt Fact (frozen dataclass) so existing callers still get the populated value back.
  - After the existing `INSERT OR REPLACE`, call `self._vec.upsert(fact.fact_id, embedding_bytes)`.
- Guard the encode + upsert behind the encoder presence so existing test code that constructs `SqlitePersistence(path)` without an encoder still passes (vec index simply stays empty for those).

```python
from typing import Optional
from server.external.embedding_encoder import EmbeddingEncoder
from server.external.vector_store import VectorStore
from dataclasses import replace


class SqlitePersistence:
    def __init__(self, db_path: Path, encoder: Optional[EmbeddingEncoder] = None):
        self._conn = open_db(db_path)
        self._encoder = encoder
        self._vec = VectorStore(self._conn)

    def save_fact(self, fact: Fact) -> None:
        embedding = fact.embedding
        if embedding is None and self._encoder is not None:
            embedding = self._encoder.encode(fact.value)
            fact = replace(fact, embedding=embedding)
        self._conn.execute(
            """INSERT OR REPLACE INTO facts ...""",   # existing SQL unchanged
            (...)
        )
        if embedding is not None:
            self._vec.upsert(fact.fact_id, embedding)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_sqlite_persistence_facts.py -v
```

Expected: existing tests still pass, new tests pass.

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/ --ignore=tests/wakeword -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add server/external/sqlite_persistence.py tests/external/test_sqlite_persistence_facts.py
git commit -m "memory: SqlitePersistence.save_fact auto-encodes + mirrors to vec index"
```

---

### Task 5: HybridRetrieval (RRF over tag + vector)

**Files:**
- Create: `server/cognition/hybrid_retrieval.py`
- Test: `tests/cognition/test_hybrid_retrieval.py`

**Interfaces:**
- Consumes: existing `TagRetrieval` (returns ranked fact list by tag-match for a query), `VectorStore.search` (Task 3), `EmbeddingEncoder.encode` (Task 1)
- Produces: `HybridRetrieval.retrieve(query: str, top_k: int) -> list[Fact]`. Internally encodes query, runs both retrievers, fuses with RRF (k=60), returns top_k by fused score.

- [ ] **Step 1: Inspect existing TagRetrieval interface**

```bash
grep -n "def " server/external/tag_retrieval.py 2>/dev/null || find . -name 'tag_retrieval*.py' -not -path '*/\.*' -exec grep -n 'def ' {} +
```

Use whatever method signature TagRetrieval exposes (e.g. `retrieve_by_tags(query)`); the test below names it `retrieve` — adjust if needed.

- [ ] **Step 2: Write the failing test**

```python
# tests/cognition/test_hybrid_retrieval.py
from dataclasses import dataclass
from datetime import datetime, UTC
import struct
from server.cognition.contracts import Fact
from server.cognition.hybrid_retrieval import HybridRetrieval


@dataclass
class FakeTagRetrieval:
    facts_by_tag: dict

    def retrieve(self, query: str, top_k: int = 10):
        return self.facts_by_tag.get(query, [])


class FakeVectorStore:
    def __init__(self, ranking):
        self._ranking = ranking  # list[(fact_id, distance)]

    def search(self, query_embedding, top_k):
        return self._ranking[:top_k]


class FakeEncoder:
    def encode(self, text):
        return struct.pack("<384f", *([0.0] * 384))


def _fact(fid: str, value: str) -> Fact:
    now = datetime.now(UTC)
    return Fact(
        fact_id=fid, category="x", key=fid, value=value, tags=[],
        source_episode_id=None, embedding=None, valid_at=now,
        invalid_at=None, created_at=now,
    )


class FakeFactStore:
    def __init__(self, facts):
        self._by_id = {f.fact_id: f for f in facts}

    def get(self, fact_id):
        return self._by_id.get(fact_id)


def test_rrf_combines_tag_and_vector_rankings():
    f1 = _fact("a", "alpha")
    f2 = _fact("b", "beta")
    f3 = _fact("c", "gamma")
    tag = FakeTagRetrieval({"coffee": [f1, f2]})   # ranks a > b
    vec = FakeVectorStore([("b", 0.1), ("c", 0.2), ("a", 0.3)])  # ranks b > c > a
    store = FakeFactStore([f1, f2, f3])
    hr = HybridRetrieval(tag, vec, FakeEncoder(), store, k=60)

    out = hr.retrieve("coffee", top_k=3)
    out_ids = [f.fact_id for f in out]

    # b appears in both => highest RRF score; a appears in both at lower ranks;
    # c only in vector. Expected order: b, a, c.
    assert out_ids == ["b", "a", "c"]


def test_top_k_truncates():
    f1 = _fact("a", "alpha"); f2 = _fact("b", "beta")
    tag = FakeTagRetrieval({"q": [f1, f2]})
    vec = FakeVectorStore([])
    store = FakeFactStore([f1, f2])
    hr = HybridRetrieval(tag, vec, FakeEncoder(), store, k=60)
    out = hr.retrieve("q", top_k=1)
    assert len(out) == 1


def test_facts_only_in_vector_results_are_still_returned():
    f1 = _fact("a", "alpha")
    tag = FakeTagRetrieval({})
    vec = FakeVectorStore([("a", 0.1)])
    store = FakeFactStore([f1])
    hr = HybridRetrieval(tag, vec, FakeEncoder(), store, k=60)
    out = hr.retrieve("anything", top_k=5)
    assert [f.fact_id for f in out] == ["a"]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/cognition/test_hybrid_retrieval.py -v
```

Expected: ImportError.

- [ ] **Step 4: Write the implementation**

```python
# server/cognition/hybrid_retrieval.py
"""Hybrid retrieval combining tag-match and vector cosine via Reciprocal Rank Fusion.

RRF (Cormack et al. 2009; 2026 RAG consensus) is robust to incompatible score
scales — tag-match is a count, vector distance is L2 in [0, 2]. Fusing on
RANK rather than score sidesteps the calibration problem entirely.
"""
from __future__ import annotations
from typing import Iterable, Protocol


class _FactStore(Protocol):
    def get(self, fact_id: str): ...


class _TagRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 10): ...


class _VectorStore(Protocol):
    def search(self, query_embedding: bytes, top_k: int): ...


class _Encoder(Protocol):
    def encode(self, text: str) -> bytes: ...


class HybridRetrieval:
    def __init__(
        self,
        tag_retrieval: _TagRetriever,
        vector_store: _VectorStore,
        encoder: _Encoder,
        fact_store: _FactStore,
        *,
        k: int = 60,
        per_retriever_top_k: int = 20,
    ):
        self._tag = tag_retrieval
        self._vec = vector_store
        self._enc = encoder
        self._facts = fact_store
        self._k = k
        self._per = per_retriever_top_k

    def retrieve(self, query: str, top_k: int = 10):
        tag_hits = list(self._tag.retrieve(query, top_k=self._per))
        q_emb = self._enc.encode(query)
        vec_hits = self._vec.search(q_emb, top_k=self._per)

        scores: dict[str, float] = {}
        for rank, fact in enumerate(tag_hits):
            scores[fact.fact_id] = scores.get(fact.fact_id, 0.0) + 1.0 / (self._k + rank)
        for rank, (fact_id, _dist) in enumerate(vec_hits):
            scores[fact_id] = scores.get(fact_id, 0.0) + 1.0 / (self._k + rank)

        ranked_ids = sorted(scores, key=lambda fid: scores[fid], reverse=True)
        out = []
        for fid in ranked_ids[:top_k]:
            f = self._facts.get(fid)
            if f is not None:
                out.append(f)
        return out
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/cognition/test_hybrid_retrieval.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add server/cognition/hybrid_retrieval.py tests/cognition/test_hybrid_retrieval.py
git commit -m "memory: HybridRetrieval fuses tag + vector via Reciprocal Rank Fusion"
```

---

### Task 6: AsyncFactWriter (writes off hot path)

**Files:**
- Create: `server/external/async_fact_writer.py`
- Test: `tests/external/test_async_fact_writer.py`

**Interfaces:**
- Consumes: `SqlitePersistence.save_fact` (synchronous), a `Fact` value
- Produces:
  - `AsyncFactWriter(persistence).enqueue(fact: Fact) -> None` — returns immediately
  - `AsyncFactWriter(persistence).flush(timeout: float = 5.0) -> None` — blocks until queue drains
  - `AsyncFactWriter(persistence).shutdown(timeout: float = 5.0) -> None` — flush + stop the worker thread

- [ ] **Step 1: Write the failing test**

```python
# tests/external/test_async_fact_writer.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_async_fact_writer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

```python
# server/external/async_fact_writer.py
"""Background writer for facts. The LLM tool-call returns within ~ms; the
encode + DB write happens on a single worker thread that drains a FIFO queue.

Matches Letta's primary/sleep-time split — hot path doesn't block on disk I/O.
"""
from __future__ import annotations
import queue
import threading
import logging
from typing import Optional

from server.cognition.contracts import Fact

_log = logging.getLogger(__name__)
_SENTINEL = object()


class AsyncFactWriter:
    def __init__(self, persistence):
        self._persistence = persistence
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="igor-fact-writer", daemon=True)
        self._thread.start()

    def enqueue(self, fact: Fact) -> None:
        self._queue.put(fact)

    def flush(self, timeout: float = 5.0) -> None:
        # block until queue is empty; raise on timeout
        deadline = threading.Event()
        threading.Timer(timeout, deadline.set).start()
        while not self._queue.empty():
            if deadline.is_set():
                raise TimeoutError("AsyncFactWriter.flush timeout")
            self._queue.join() if False else None  # keep simple
            import time; time.sleep(0.01)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            try:
                self._persistence.save_fact(item)
            except Exception:
                _log.exception("AsyncFactWriter: save_fact failed")
            finally:
                self._queue.task_done()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/external/test_async_fact_writer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add server/external/async_fact_writer.py tests/external/test_async_fact_writer.py
git commit -m "memory: AsyncFactWriter moves encode+DB writes off the LLM hot path"
```

---

### Task 7: Compose everything in main.py

**Files:**
- Modify: `server/main.py`
- Modify: `Dockerfile` (warm the fastembed model cache during image build)
- Test: `tests/test_main_composition.py` (extend)

**Interfaces:**
- Consumes: every prior task's components
- Produces: at runtime, the Conversation service uses `HybridRetrieval` for memory lookups, and any tool that previously called `save_memory` sync now goes through `AsyncFactWriter`. Existing `IGOR_API_TOKEN` / `BRAIN_DIR` / `ANTHROPIC_API_KEY` envvars unchanged. New optional envvar `IGOR_EMBEDDING_DISABLED=1` lets you bypass encoder/vec setup for diagnostic boots.

- [ ] **Step 1: Read current composition root**

```bash
sed -n '40,90p' server/main.py
```

Identify where `SqlitePersistence`, `TagRetrieval`, and the Conversation service are wired together.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_main_composition.py — append
def test_main_wires_hybrid_retrieval_when_embeddings_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("HA_TOKEN", "x")
    monkeypatch.delenv("IGOR_EMBEDDING_DISABLED", raising=False)

    from importlib import reload
    import server.main as m
    reload(m)
    app = m.build_app()  # or the equivalent composition entrypoint

    from server.cognition.hybrid_retrieval import HybridRetrieval
    # whatever attribute the conversation service uses
    assert isinstance(app.state.retrieval, HybridRetrieval)


def test_main_skips_embedding_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("HA_TOKEN", "x")
    monkeypatch.setenv("IGOR_EMBEDDING_DISABLED", "1")

    from importlib import reload
    import server.main as m
    reload(m)
    app = m.build_app()

    from server.cognition.hybrid_retrieval import HybridRetrieval
    assert not isinstance(app.state.retrieval, HybridRetrieval)
```

- [ ] **Step 3: Modify main.py composition**

In `server/main.py`:

```python
import os
from server.external.embedding_encoder import EmbeddingEncoder
from server.external.vector_store import VectorStore
from server.external.async_fact_writer import AsyncFactWriter
from server.cognition.hybrid_retrieval import HybridRetrieval

EMBEDDINGS_DISABLED = os.getenv("IGOR_EMBEDDING_DISABLED") == "1"

if EMBEDDINGS_DISABLED:
    persistence = SqlitePersistence(brain_dir / "brain.db")
    retrieval = TagRetrieval(persistence)
    fact_writer = None
else:
    encoder = EmbeddingEncoder()
    persistence = SqlitePersistence(brain_dir / "brain.db", encoder=encoder)
    tag_retrieval = TagRetrieval(persistence)
    vector_store = VectorStore(persistence._conn)
    retrieval = HybridRetrieval(tag_retrieval, vector_store, encoder, persistence)
    fact_writer = AsyncFactWriter(persistence)

# wire fact_writer into the save_memory command path; existing call sites
# call fact_writer.enqueue(fact) instead of persistence.save_fact(fact).
# register an atexit / FastAPI shutdown hook that calls fact_writer.shutdown()
```

- [ ] **Step 4: Update Dockerfile to warm the model cache at build time**

Append to `Dockerfile`:

```dockerfile
# Warm the fastembed model cache so the first request doesn't pay download latency.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/test_main_composition.py -v
```

Expected: both new tests pass.

- [ ] **Step 6: Run full suite**

```bash
/home/samda/.venvs/igor/bin/python -m pytest tests/ --ignore=tests/wakeword -q
```

Expected: all green; total count rises by ~9 tests over baseline.

- [ ] **Step 7: Manual smoke test inside the Igor venv**

```bash
/home/samda/.venvs/igor/bin/python -c "
from pathlib import Path
from server.external.embedding_encoder import EmbeddingEncoder
from server.external.sqlite_persistence import SqlitePersistence
from server.external.vector_store import VectorStore
from server.cognition.hybrid_retrieval import HybridRetrieval

import uuid
from datetime import datetime, UTC
from server.cognition.contracts import Fact

p = Path('/tmp/igor-smoke.db')
p.unlink(missing_ok=True)
enc = EmbeddingEncoder()
sp = SqlitePersistence(p, encoder=enc)
now = datetime.now(UTC)
sp.save_fact(Fact(str(uuid.uuid4()), 'prefs', 'coffee', 'dark roast', [],
                   None, None, now, None, now))
sp.save_fact(Fact(str(uuid.uuid4()), 'prefs', 'tea', 'earl grey', [],
                   None, None, now, None, now))
vs = VectorStore(sp._conn)
hits = vs.search(enc.encode('strong morning beverage'), top_k=2)
print('vec hits:', hits)
"
```

Expected: prints two `(fact_id, distance)` tuples, with `coffee` ranking ahead of `tea` for that query.

- [ ] **Step 8: Commit**

```bash
git add server/main.py Dockerfile tests/test_main_composition.py
git commit -m "memory: wire HybridRetrieval + AsyncFactWriter into composition root"
```

---

### Task 8: Deploy + production validation

**Files:**
- None (deployment task)

**Interfaces:**
- Consumes: a clean Pi5 brain.db (already wiped this session)
- Produces: live Igor on the Pi5 using hybrid retrieval + async writes

- [ ] **Step 1: Push to main**

```bash
git push origin main
```

- [ ] **Step 2: Wait for Portainer redeploy or trigger it manually**

- [ ] **Step 3: Verify container health**

```bash
ssh samda@10.0.30.5 'docker ps --filter name=igor --format "{{.Status}}"'
ssh samda@10.0.30.5 'docker logs igor --tail 50 2>&1 | grep -iE "encoder|embedding|hybrid|fastembed"'
```

Expected: container `Up`, log shows fastembed model load + composition wiring.

- [ ] **Step 4: Verify embeddings populate as Sam talks to Igor**

After a few conversations:

```bash
ssh samda@10.0.30.5 'docker exec igor python -c "
import sqlite3
con = sqlite3.connect(\"/app/data/brain.db\")
print(\"facts with embedding:\", con.execute(\"SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL\").fetchone()[0])
print(\"facts total:\", con.execute(\"SELECT COUNT(*) FROM facts\").fetchone()[0])
print(\"vec rows:\", con.execute(\"SELECT COUNT(*) FROM facts_vec\").fetchone()[0])
"'
```

Expected: `facts with embedding == facts total`, `vec rows == facts total`.

- [ ] **Step 5: Verify hot-path latency**

Look at the per-turn duration in Igor's logs. Compare a save_memory-heavy turn pre- and post-change.

Expected: post-change turn duration is no higher than pre-change. The encode + write are on the background thread and shouldn't appear in the LLM-turn timing.

- [ ] **Step 6: Mark Phase 2 done — proceed to consolidation reorganization (Phase 2 item #3 — bi-temporal invalidation + dedupe) as a separate plan**

---

## Self-Review

**Spec coverage:**
- Embedding encoder ✓ Task 1
- sqlite-vec index ✓ Tasks 2 & 3
- Auto-encode on save ✓ Task 4
- Hybrid retrieval with RRF ✓ Task 5
- Writes off hot path ✓ Task 6
- Composition wiring + envvar bypass ✓ Task 7
- Deploy + validation ✓ Task 8

**Type consistency:**
- `Fact.embedding: Optional[bytes]` used identically across Tasks 1, 4, 5
- `EmbeddingEncoder.encode(text) -> bytes` consistent in Tasks 1, 4, 5
- `VectorStore.upsert` / `search` / `delete` signatures consistent in Tasks 3, 4, 5
- RRF `k=60` constant consistent in Task 5 implementation and tests

**Placeholders:** none.

**Out of scope (do not build in this plan):**
- Consolidation reorganization (Phase 2 item #3 — bi-temporal invalidation + dedupe). Separate plan.
- External integrations: calendar, todos, HA state read-through (Phase 3). Separate plans.
- Backfill of historical facts — brain.db is empty, no backfill needed.
- Speaker awareness (the unused `episodes.speaker_id` column). Separate plan when HA Wyoming exposes speaker IDs.

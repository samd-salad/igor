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

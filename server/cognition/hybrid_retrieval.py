"""Hybrid retrieval combining tag-match and vector cosine via Reciprocal Rank Fusion.

RRF (Cormack et al. 2009; 2026 RAG consensus) is robust to incompatible score
scales — tag-match is a count, vector distance is L2 in [0, 2]. Fusing on
RANK rather than score sidesteps the calibration problem entirely.

Implements the same `query(turn, k)` port shape as TagRetrieval so it's a
drop-in composition swap.
"""
from __future__ import annotations
from typing import Protocol, Optional

from server.cognition.contracts import VoiceTurn, Fact


class _FactLookup(Protocol):
    def find_fact_by_id(self, fact_id: str) -> Optional[Fact]: ...


class _TagRetriever(Protocol):
    def query(self, turn: VoiceTurn, k: int = 10) -> list[Fact]: ...


class _VectorStore(Protocol):
    def search(self, query_embedding: bytes, top_k: int) -> list[str]: ...


class _Encoder(Protocol):
    def encode(self, text: str) -> bytes: ...


class HybridRetrieval:
    def __init__(
        self,
        tag_retrieval: _TagRetriever,
        vector_store: _VectorStore,
        encoder: _Encoder,
        fact_lookup: _FactLookup,
        *,
        k: int = 60,
        per_retriever_top_k: int = 20,
    ):
        self._tag = tag_retrieval
        self._vec = vector_store
        self._enc = encoder
        self._lookup = fact_lookup
        self._k = k
        self._per = per_retriever_top_k

    def query(self, turn: VoiceTurn, k: int = 10) -> list[Fact]:
        tag_hits = list(self._tag.query(turn, k=self._per))
        q_emb = self._enc.encode(turn.input_text)
        vec_hits = self._vec.search(q_emb, top_k=self._per)

        scores: dict[str, float] = {}
        for rank, fact in enumerate(tag_hits):
            scores[fact.fact_id] = scores.get(fact.fact_id, 0.0) + 1.0 / (self._k + rank)
        for rank, fact_id in enumerate(vec_hits):
            scores[fact_id] = scores.get(fact_id, 0.0) + 1.0 / (self._k + rank)

        ranked_ids = sorted(scores, key=lambda fid: scores[fid], reverse=True)
        out: list[Fact] = []
        for fid in ranked_ids[:k]:
            f = self._lookup.find_fact_by_id(fid)
            if f is not None:
                out.append(f)
        return out

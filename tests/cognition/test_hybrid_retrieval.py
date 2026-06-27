from dataclasses import dataclass
from datetime import datetime, UTC
import struct
import uuid

from server.cognition.contracts import Fact, VoiceTurn, RoomConfig
from server.cognition.hybrid_retrieval import HybridRetrieval


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id=str(uuid.uuid4()),
        started_at=datetime.now(UTC),
        device_id=None,
        room=RoomConfig(room_id="test", display_name="Test"),
        input_text=text,
        speaker_id=None,
    )


@dataclass
class FakeTagRetrieval:
    facts_by_text: dict   # input_text -> list[Fact]

    def query(self, turn: VoiceTurn, k: int = 10):
        return self.facts_by_text.get(turn.input_text, [])[:k]


class FakeVectorStore:
    def __init__(self, ranking):
        self._ranking = ranking  # list[fact_id], nearest first

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


class FakeFactLookup:
    def __init__(self, facts):
        self._by_id = {f.fact_id: f for f in facts}

    def find_fact_by_id(self, fact_id):
        return self._by_id.get(fact_id)


def test_rrf_combines_tag_and_vector_rankings():
    f1 = _fact("a", "alpha")
    f2 = _fact("b", "beta")
    f3 = _fact("c", "gamma")
    tag = FakeTagRetrieval({"coffee": [f1, f2]})   # ranks a > b
    vec = FakeVectorStore(["b", "c", "a"])  # ranks b > c > a
    store = FakeFactLookup([f1, f2, f3])
    hr = HybridRetrieval(tag, vec, FakeEncoder(), store, k=60)

    out = hr.query(_turn("coffee"), k=3)
    out_ids = [f.fact_id for f in out]

    # b appears at rank 0 in vec + rank 1 in tag => highest RRF;
    # a appears at rank 0 in tag + rank 2 in vec => second;
    # c only in vec at rank 1 => third.
    assert out_ids == ["b", "a", "c"]


def test_top_k_truncates():
    f1 = _fact("a", "alpha"); f2 = _fact("b", "beta")
    tag = FakeTagRetrieval({"q": [f1, f2]})
    vec = FakeVectorStore([])
    store = FakeFactLookup([f1, f2])
    hr = HybridRetrieval(tag, vec, FakeEncoder(), store, k=60)
    out = hr.query(_turn("q"), k=1)
    assert len(out) == 1


def test_facts_only_in_vector_results_are_still_returned():
    f1 = _fact("a", "alpha")
    tag = FakeTagRetrieval({})
    vec = FakeVectorStore(["a"])
    store = FakeFactLookup([f1])
    hr = HybridRetrieval(tag, vec, FakeEncoder(), store, k=60)
    out = hr.query(_turn("anything"), k=5)
    assert [f.fact_id for f in out] == ["a"]

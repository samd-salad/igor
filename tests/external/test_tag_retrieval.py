from datetime import datetime, UTC
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.external.sqlite_persistence import SqlitePersistence
from server.external.sqlite_retrieval import TagRetrieval


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t1", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig(room_id="default", display_name="Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_tag_overlap_ranks_matching_facts_higher(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    mem = MemoryStore(sp)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    mem.save_fact("prefs", "coffee", "dark roast oat milk",
                  tags=["coffee", "beverage"], source_episode_id=None, now=now)
    mem.save_fact("prefs", "music", "jazz",
                  tags=["music"], source_episode_id=None, now=now)

    retr = TagRetrieval(sp)
    hits = retr.query(_turn("what coffee do I like"), k=2)
    assert hits[0].key == "coffee"

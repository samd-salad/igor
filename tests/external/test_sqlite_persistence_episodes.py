from datetime import datetime, UTC
from server.cognition.contracts import Episode, ToolCallRecord
from server.external.sqlite_persistence import SqlitePersistence


def test_save_and_load_episode(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    ep = Episode(
        episode_id="ep-1",
        occurred_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        speaker_id=None,
        participants=["sam", "igor"],
        intent="time_query",
        raw_utterance="what time is it",
        tool_calls=[ToolCallRecord(name="get_time", args={"include_date": True}, result="3 PM")],
        emotional_tone=None,
        summary=None,
        consolidated_at=None,
    )
    sp.save_episode(ep)
    loaded = sp.load_episode("ep-1")
    assert loaded is not None
    assert loaded.raw_utterance == "what time is it"
    assert len(loaded.tool_calls) == 1
    assert loaded.tool_calls[0].name == "get_time"


def test_list_unconsolidated(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    for i in range(3):
        sp.save_episode(Episode(
            episode_id=f"ep-{i}", occurred_at=datetime(2026, 1, 1, 10, i, tzinfo=UTC),
            speaker_id=None, participants=[], intent=None,
            raw_utterance="x", tool_calls=[], emotional_tone=None,
            summary=None, consolidated_at=None,
        ))
    assert len(sp.list_unconsolidated_episodes()) == 3
    sp.mark_episodes_consolidated(["ep-0", "ep-1"], at=datetime(2026, 1, 2, tzinfo=UTC))
    assert len(sp.list_unconsolidated_episodes()) == 1

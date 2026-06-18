from datetime import datetime, UTC
from server.cognition.contracts import (
    Episode, Fact, Reflection, FeedbackEntry, Reminder, ToolCallRecord,
)


def test_episode_structured_fields():
    now = datetime.now(UTC)
    ep = Episode(
        episode_id="ep-1", occurred_at=now, speaker_id=None,
        participants=["sam", "igor"], intent="time_query",
        raw_utterance="what time is it", tool_calls=[
            ToolCallRecord(name="get_time", args={}, result="3 PM"),
        ],
        emotional_tone="neutral", summary=None, consolidated_at=None,
    )
    assert ep.episode_id == "ep-1"
    assert len(ep.tool_calls) == 1


def test_fact_bi_temporal_columns():
    now = datetime.now(UTC)
    fact = Fact(
        fact_id="f-1", category="preferences", key="coffee", value="dark roast oat milk",
        tags=["beverage", "morning"], source_episode_id="ep-1",
        embedding=None,
        valid_at=now, invalid_at=None, created_at=now,
    )
    assert fact.invalid_at is None
    assert fact.embedding is None


def test_reflection_minimal():
    r = Reflection(reflection_id="r-1", occurred_at=datetime.now(UTC),
                   note="user got frustrated by long preamble", source_episode_id="ep-1")
    assert r.note.startswith("user got")


def test_feedback_and_reminder():
    now = datetime.now(UTC)
    fb = FeedbackEntry(feedback_id="fb-1", occurred_at=now,
                       issue="time format should be 24h", status="open",
                       source_episode_id=None)
    assert fb.status == "open"
    rm = Reminder(reminder_id="rm-1", name="pasta", fire_at=now,
                  room_id="kitchen", status="pending", source_episode_id="ep-1")
    assert rm.status == "pending"

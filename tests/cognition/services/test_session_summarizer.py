from datetime import datetime, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import (
    Episode, VoiceTurn, RoomConfig, ConversationResult,
)
from server.cognition.ports.llm import ChatResult
from server.cognition.services.session_summarizer import SessionSummarizer
from server.external.sqlite_persistence import SqlitePersistence
from server.external.system_clock import SystemClock


class _StubLLM:
    def chat(self, system_prompt, user_text, tool_schemas, tool_executor, history=None):
        return ChatResult(text="user asked about coffee preferences",
                          commands_executed=[], input_tokens=5, output_tokens=2)


def _seed_episode(sp, eid="ep-1"):
    sp.save_episode(Episode(
        episode_id=eid, occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        speaker_id=None, participants=[], intent="llm",
        raw_utterance="what coffee do I like", tool_calls=[],
        emotional_tone=None, summary=None, consolidated_at=None,
    ))


def test_summarizer_stamps_summary(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    _seed_episode(sp, "ep-1")
    summarizer = SessionSummarizer(
        episodes=EpisodeStore(sp), memory=MemoryStore(sp),
        llm=_StubLLM(), clock=SystemClock(),
    )
    summarizer.start()
    summarizer.enqueue(VoiceTurn(
        correlation_id="ep-1", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text="what coffee do I like", speaker_id=None, metadata={},
    ), ConversationResult(correlation_id="ep-1", response_text="dark roast",
                          commands_executed=[], end_conversation=True))
    summarizer.shutdown(timeout=2.0)
    loaded = sp.load_episode("ep-1")
    assert loaded.summary == "user asked about coffee preferences"

from datetime import datetime, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.cognition.ports.llm import ChatResult
from server.cognition.services.conversation import Conversation
from server.external.sqlite_persistence import SqlitePersistence
from server.external.sqlite_retrieval import TagRetrieval
from server.external.system_clock import SystemClock


class _StubLLM:
    def chat(self, system_prompt, user_text, tool_schemas, tool_executor, history=None):
        return ChatResult(text="hi back", commands_executed=[], input_tokens=10, output_tokens=2)


class _StubExecutor:
    def list_schemas(self):
        return []
    def execute(self, name, args, turn):
        return ""


def _turn(text: str, correlation_id: str = "t-1") -> VoiceTurn:
    return VoiceTurn(
        correlation_id=correlation_id, started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("default", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def _build_conversation(sp):
    return Conversation(
        memory=MemoryStore(sp), episodes=EpisodeStore(sp),
        identity=IdentityStore(sp), user_state=UserState(sp),
        retrieval=TagRetrieval(sp), llm=_StubLLM(),
        tools=_StubExecutor(), clock=SystemClock(),
    )


def test_conversation_writes_episode_with_correlation_id(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    conv = _build_conversation(sp)
    result = conv.process(_turn("hello there", correlation_id="t-hello"))
    assert result.correlation_id == "t-hello"
    assert result.response_text == "hi back"
    ep = sp.load_episode("t-hello")
    assert ep is not None
    assert ep.raw_utterance == "hello there"


def test_conversation_short_circuits_tier1(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    conv = _build_conversation(sp)
    result = conv.process(_turn("pause", correlation_id="t-pause"))
    assert result.response_text == "Paused."
    assert result.commands_executed == ["play_pause"]
